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
try:
    from .qrcode_utils import QRCodeRegistrator
except ImportError:
    # Handle case where package is run from different root
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.qrcode_utils import QRCodeRegistrator



class OpenCVTransform:
    """
    OpenCV-based preprocessing to match C++ deployment exactly.
    Input: PIL Image (RGB) or numpy array (RGB or RGBA)
    Output: Tensor [C, H, W] normalized
    
    Supports both square and rectangular images.
    Args:
        image_size: Target image size as int (square) or tuple (height, width) for rectangular
        mean: Mean values for normalization
        std: Standard deviation values for normalization
    """
    def __init__(self, image_size: Union[int, Tuple[int, int]], mean: list, std: list):
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
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


class RGBAColorJitter:
    """
    ColorJitter that supports RGBA images.
    Applies color jittering to RGB channels while keeping Alpha channel unchanged.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
        # Create the standard ColorJitter for RGB channels
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, img):
        # Check if image is RGBA
        if img.mode == 'RGBA':
            # Split into RGB and Alpha channels
            rgb_img = img.convert('RGB')
            alpha_channel = img.split()[3]
            
            # Apply color jitter to RGB channels
            jittered_rgb = self.color_jitter(rgb_img)
            
            # Merge RGB and Alpha back together
            jittered_rgba = Image.merge('RGBA', jittered_rgb.split() + (alpha_channel,))
            return jittered_rgba
        else:
            # For non-RGBA images, apply standard color jitter
            return self.color_jitter(img)



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
    
    def __init__(self, data_dir: str, transform=None, use_gray: bool = False):
        """
        Args:
            data_dir: Root directory containing class subdirectories
            transform: Optional transform to be applied on images
            use_gray: Whether to convert images to grayscale
            use_contrastive: Whether to use contrastive learning
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        self.use_gray = use_gray
        self.use_contrastive = use_contrastive
        
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
    Supports four binary labels: is_copied, is_blurry, is_low_light, is_screen
    
    Expected annotation format (JSON file):
    {
        "image1.jpg": {
            "is_copied": 1,
            "is_blurry": 0,
            "is_low_light": 0,
            "is_screen": 0
        },
        "image2.jpg": {
            "is_copied": 1,
            "is_blurry": 1,
            "is_low_light": 0,
            "is_screen": 1
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
        label_names: List[str] = None,
        register_dir: str = None,
        wqmodules_dir: str = 'wqmodules',
        use_contrastive: bool = False,
        use_gray: bool = False
    ):
        """
        Args:
            data_dir: Root directory containing images
            annotation_file: Path to JSON annotation file. If None, looks for 'annotations.json' in data_dir
            transform: Optional transform to be applied on images
            label_names: List of label names (default: ["is_copied", "is_low_light", "is_blurry", "is_screen"])
            use_contrastive: Whether to use contrastive learning
            use_gray: Whether to convert images to grayscale
        """
        self.data_dir = data_dir
        self.transform = transform
        self.label_names = label_names or ["is_copied", "is_low_light", "is_blurry", "is_screen"]
        self.num_labels = len(self.label_names)
        self.samples = []
        self.use_gray = use_gray
        self.use_contrastive = use_contrastive
        
        # Determine annotation file path
        if annotation_file is None:
            annotation_file = os.path.join(data_dir, 'annotations.json')
        
        self.annotation_file = annotation_file
        
        # 4-channel registration setup
        if register_dir is None:
            # Look for 'register' in common locations
            possible_reg_dirs = [
                os.path.join(data_dir, 'register'),
                os.path.join(os.path.dirname(data_dir), 'register'),
                os.path.join(os.path.dirname(os.path.dirname(data_dir)), 'register')
            ]
            for p in possible_reg_dirs:
                if os.path.exists(p):
                    self.register_dir = p
                    break
            else:
                self.register_dir = os.path.join(os.path.dirname(data_dir), 'register')
        else:
            self.register_dir = register_dir

        self.wqmodules_dir = wqmodules_dir
        self.registrator = None  # Lazy init to handle multiprocessing
        
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
        image = Image.open(img_path)

        if self.use_contrastive:
            # Use 4-channel image for contrastive learning
            if image.mode == 'RGBA':
                # Split RGBA channels
                r, g, b, a = image.split()
                # Convert RGB to grayscale and replicate
                if self.use_gray:
                    gray = Image.merge('RGB', [r, g, b]).convert('L')
                    r = gray
                    g = gray
                    b = gray
                # Merge back to RGBA
                image = Image.merge('RGBA', [r, g, b, a])
            else:
                # Convert to RGBA first
                rgb_img = image.convert('RGB')
                if self.use_gray:
                    gray = rgb_img.convert('L')
                    r, g, b = gray, gray, gray
                else:
                    r, g, b = rgb_img.split()
                a = Image.new('L', image.size, 255)
                image = Image.merge('RGBA', [r, g, b, a])
        else:
            # Extract RGB 3 channels
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            else:
                image = image.convert('RGB')

            # Convert to grayscale and replicate to 3 channels if needed
            if self.use_gray:
                gray = image.convert('L')
                image = Image.merge('RGB', [gray, gray, gray])

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return image, labels_tensor


def get_transforms(
    image_size: Union[int, Tuple[int, int]] = 384,
    augment: bool = False,
    backend: str = 'pil',
    use_contrastive: bool = False,
    use_gray: bool = False
):
    """
    Get image transforms for training and validation.
    
    Args:
        image_size: Target image size as int (square) or tuple (height, width) for rectangular
        augment: Whether to apply data augmentation (images only)
        backend: 'pil' (default PyTorch) or 'opencv' (match C++ result)
        use_contrastive: Whether using contrastive learning (4 channels)
        use_gray: Whether using grayscale images
        
    Returns:
        transform function/object
    """

    # 默认是4通道 非gray图
    if use_contrastive:
        if use_gray:
            mean = [0.449, 0.449, 0.449, 0.449] 
            std = [0.226, 0.226, 0.226, 0.226] 
        else:
            mean = [0.485, 0.456, 0.406, 0.449]
            std = [0.229, 0.224, 0.225, 0.226] 
    else:
        if use_gray:
            mean = [0.449, 0.449, 0.449] 
            std = [0.226, 0.226, 0.226]
        else:
            mean = [0.485, 0.456, 0.406] 
            std = [0.229, 0.224, 0.225]

    # For pure inference without augmentations, use the dedicated OpenCVTransform
    if backend == 'opencv' and not augment:
        return OpenCVTransform(image_size, mean, std)

    # Prepare logic for Resize step
    if backend == 'opencv':
        resize_transform = OpenCVResize(image_size)
    else:
        if isinstance(image_size, int):
            resize_transform = transforms.Resize((image_size, image_size))
        else:
            resize_transform = transforms.Resize((image_size[0], image_size[1]))

    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            resize_transform,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            RGBAColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
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
    image_size: Union[int, Tuple[int, int]] = 384,
    num_workers: int = 0,
    backend: str = 'opencv',
    multi_label: bool = False,
    label_names: List[str] = None,
    use_contrastive: bool = False,
    use_gray: bool = False
) -> Tuple[DataLoader, DataLoader, Union[list, List[str]]]:
    """
    Create training and validation data loaders.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for data loaders
        image_size: Target image size as int (square) or tuple (height, width) for rectangular
        num_workers: Number of worker processes for data loading
        backend: 'pil' or 'opencv' for preprocessing
        multi_label: If True, use multi-label dataset (requires annotations.json)
        label_names: List of label names for multi-label classification
        use_contrastive: Whether using contrastive learning
        use_gray: Whether using grayscale images
        
    Returns:
        Tuple of (train_loader, val_loader, classes/label_names)
    """
    # Get transforms (Training usually stays on PIL for augmentation support)
    train_transform = get_transforms(image_size=image_size, augment=True, backend=backend, use_contrastive=use_contrastive, use_gray=use_gray)
    
    # Validation can use PIL or OpenCV, stick to PIL for standard training metrics
    val_transform = get_transforms(image_size=image_size, augment=False, backend=backend, use_contrastive=use_contrastive, use_gray=use_gray)
    
    # Create datasets based on classification type
    if multi_label:
        # Multi-label classification
        if label_names is None:
            label_names = ["is_copied", "is_low_light", "is_blurry"]
        
        train_dataset = QRCodeMultiLabelDataset(
            train_dir, 
            transform=train_transform,
            label_names=label_names,
            wqmodules_dir='wqmodules',
            use_contrastive=use_contrastive,
            use_gray=use_gray
        )
        val_dataset = QRCodeMultiLabelDataset(
            val_dir, 
            transform=val_transform,
            label_names=label_names,
            wqmodules_dir='wqmodules',
            use_contrastive=use_contrastive,
            use_gray=use_gray
        )
        
        # Verify label names match
        if train_dataset.label_names != val_dataset.label_names:
            print("Warning: Train and validation datasets have different label names!")
        
        metadata = train_dataset.label_names
    else:
        # Single-class classification
        train_dataset = QRCodeDataset(train_dir, transform=train_transform, use_contrastive=use_contrastive, use_gray=use_gray)
        val_dataset = QRCodeDataset(val_dir, transform=val_transform, use_contrastive=use_contrastive, use_gray=use_gray)
        
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


def get_inference_transform(image_size: Union[int, Tuple[int, int]] = 384, backend: str = 'opencv', use_contrastive: bool = False, use_gray: bool = False):
    """
    Get transform for inference on single images.
    
    Args:
        image_size: Target image size as int (square) or tuple (height, width) for rectangular
        backend: 'pil' (default) or 'opencv'
        use_contrastive: Whether using contrastive learning (4 channels)
        use_gray: Whether using grayscale images
        
    Returns:
        transform function/object
    """
    return get_transforms(image_size=image_size, augment=False, backend=backend, use_contrastive=use_contrastive, use_gray=use_gray)
