"""
Dataset and data loading utilities for QR code multi-label classification.
Handles image preprocessing, augmentation, and data loading.
Supports both single-class and multi-label classification.
"""

import os
import json
from typing import Tuple, Optional, Union, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import cv2


class OpenCVTransform:
    """
    OpenCV-based preprocessing to match C++ deployment exactly.
    Input: PIL Image (RGB) or numpy array (RGB)
    Output: Tensor [C, H, W] normalized
    """
    def __init__(self, image_size: int, mean: list, std: list):
        self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img):
        # 1. Convert PIL to numpy (RGB)
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # 2. Resize using OpenCV (Bilinear default)
        # Note: cv2.resize expects (width, height) - reverse of numpy shape
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        # 3. Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        # 4. To Tensor [H, W, C] -> [C, H, W]
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)


class OpenCVResize:
    """
    OpenCV Resize implementation compatible with PIL pipeline.
    Used for training to align downsampling algorithm with C++ deployment.
    """
    def __init__(self, size):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)

    def __call__(self, img):
        # Convert PIL to numpy
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        # Resize
        # cv2.resize uses (width, height)
        img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Convert back to PIL for subsequent torchvision transforms
        return Image.fromarray(img)


class QRCodeDataset(Dataset):
    """
    Custom dataset for QR code images.
    Expects directory structure:
        data_dir/
            class_0/
                image1.jpg
                image2.jpg
            class_1/
                image1.jpg
                image2.jpg
    """
    
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Root directory containing class subdirectories
            transform: Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all images and their labels from the directory structure."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Get class directories
        class_dirs = sorted([d for d in os.listdir(self.data_dir) 
                           if os.path.isdir(os.path.join(self.data_dir, d))])
        
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {self.data_dir}")
        
        self.classes = class_dirs
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        
        # Load all images
        for class_name in class_dirs:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.samples.append((img_path, class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        for class_name, class_idx in self.class_to_idx.items():
            count = sum(1 for _, idx in self.samples if idx == class_idx)
            print(f"  - {class_name}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class QRCodeMultiLabelDataset(Dataset):
    """
    Multi-label dataset for QR code images.
    Supports three binary labels: is_copied, is_blurry, is_low_light
    
    Expected annotation format (JSON file):
    {
        "image1.jpg": {
            "is_copied": 1,
            "is_blurry": 0,
            "is_low_light": 0
        },
        "image2.jpg": {
            "is_copied": 1,
            "is_blurry": 1,
            "is_low_light": 0
        },
        ...
    }
    
    Alternative directory structure:
        data_dir/
            images/
                image1.jpg
                image2.jpg
            annotations.json
    """
    
    def __init__(
        self, 
        data_dir: str, 
        annotation_file: str = None,
        transform=None,
        label_names: List[str] = None
    ):
        """
        Args:
            data_dir: Root directory containing images
            annotation_file: Path to JSON annotation file. If None, looks for 'annotations.json' in data_dir
            transform: Optional transform to be applied on images
            label_names: List of label names (default: ["is_copied", "is_blurry", "is_low_light"])
        """
        self.data_dir = data_dir
        self.transform = transform
        self.label_names = label_names or ["is_copied", "is_low_light", "is_blurry"]
        self.num_labels = len(self.label_names)
        self.samples = []
        
        # Determine annotation file path
        if annotation_file is None:
            annotation_file = os.path.join(data_dir, 'annotations.json')
        
        self.annotation_file = annotation_file
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all images and their multi-label annotations from JSON file."""
        if not os.path.exists(self.annotation_file):
            raise ValueError(f"Annotation file not found: {self.annotation_file}")
        
        # Load annotations
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Process each image
        for img_name, labels_dict in annotations.items():
            # Support both absolute and relative paths
            if os.path.isabs(img_name):
                img_path = img_name
            else:
                # Try multiple possible locations
                possible_paths = [
                    os.path.join(self.data_dir, img_name),
                    os.path.join(self.data_dir, 'images', img_name),
                    img_name  # If it's already a valid path
                ]
                
                img_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        img_path = path
                        break
                
                if img_path is None:
                    print(f"Warning: Image not found: {img_name}, skipping...")
                    continue
            
            # Extract labels in the correct order
            label_vector = []
            for label_name in self.label_names:
                if label_name in labels_dict:
                    label_vector.append(float(labels_dict[label_name]))
                else:
                    # Default to 0 if label not specified
                    label_vector.append(0.0)
            
            self.samples.append((img_path, label_vector))
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {self.annotation_file}")
        
        # Print statistics
        print(f"Loaded {len(self.samples)} images with multi-label annotations")
        print(f"Labels: {self.label_names}")
        
        # Calculate label statistics
        label_counts = [0] * self.num_labels
        for _, labels in self.samples:
            for i, label_val in enumerate(labels):
                if label_val > 0.5:  # Consider as positive
                    label_counts[i] += 1
        
        for i, label_name in enumerate(self.label_names):
            percentage = (label_counts[i] / len(self.samples)) * 100
            print(f"  - {label_name}: {label_counts[i]} images ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels_tensor


def get_transforms(
    image_size: int = 224, 
    augment: bool = True,
    backend: str = 'opencv'  # Options: 'pil', 'opencv'
):
    """
    Get image transforms for training and validation.
    
    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation (images only)
        backend: 'pil' (default PyTorch) or 'opencv' (match C++ result)
        
    Returns:
        transform function/object
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # For pure inference without augmentations, use the dedicated OpenCVTransform
    if backend == 'opencv' and not augment:
        return OpenCVTransform(image_size, mean, std)

    # Prepare logic for Resize step
    if backend == 'opencv':
        # Use OpenCV for resizing (to align with deployment), assume square
        resize_transform = OpenCVResize(image_size)
    else:
        # Standard PIL resizing
        resize_transform = transforms.Resize((image_size, image_size))

    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            resize_transform,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Validation/test transforms without augmentation
        # Even if backend='opencv' is passed here (and for some reason didn't hit the first if block),
        # we'd construct a PIL-compatible sequence. But usually it hits the first block.
        # This block is mainly for backend='pil'
        transform = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    backend: str = 'opencv',
    multi_label: bool = False,
    label_names: List[str] = None
) -> Tuple[DataLoader, DataLoader, Union[list, List[str]]]:
    """
    Create training and validation data loaders.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for data loaders
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        backend: 'pil' or 'opencv' for preprocessing
        multi_label: If True, use multi-label dataset (requires annotations.json)
        label_names: List of label names for multi-label classification
        
    Returns:
        Tuple of (train_loader, val_loader, classes/label_names)
    """
    # Get transforms (Training usually stays on PIL for augmentation support)
    train_transform = get_transforms(image_size=image_size, augment=True, backend=backend)
    
    # Validation can use PIL or OpenCV, stick to PIL for standard training metrics
    val_transform = get_transforms(image_size=image_size, augment=False, backend=backend)
    
    # Create datasets based on classification type
    if multi_label:
        # Multi-label classification
        if label_names is None:
            label_names = ["is_copied", "is_low_light", "is_blurry"]
        
        train_dataset = QRCodeMultiLabelDataset(
            train_dir, 
            transform=train_transform,
            label_names=label_names
        )
        val_dataset = QRCodeMultiLabelDataset(
            val_dir, 
            transform=val_transform,
            label_names=label_names
        )
        
        # Verify label names match
        if train_dataset.label_names != val_dataset.label_names:
            print("Warning: Train and validation datasets have different label names!")
        
        metadata = train_dataset.label_names
    else:
        # Single-class classification
        train_dataset = QRCodeDataset(train_dir, transform=train_transform)
        val_dataset = QRCodeDataset(val_dir, transform=val_transform)
        
        # Verify classes match
        if train_dataset.classes != val_dataset.classes:
            print("Warning: Train and validation datasets have different classes!")
        
        metadata = train_dataset.classes
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, metadata


def get_inference_transform(image_size: int = 224, backend: str = 'opencv'):
    """
    Get transform for inference on single images.
    
    Args:
        image_size: Target image size
        backend: 'pil' (default) or 'opencv'
        
    Returns:
        transform function/object
    """
    return get_transforms(image_size=image_size, augment=False, backend=backend)
