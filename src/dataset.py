"""
Dataset and data loading utilities for QR code classification.
Handles image preprocessing, augmentation, and data loading.
"""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image


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


def get_transforms(image_size: int = 224, augment: bool = True):
    """
    Get image transforms for training and validation.
    
    Args:
        image_size: Target image size (will be resized to image_size x image_size)
        augment: Whether to apply data augmentation (for training)
        
    Returns:
        torchvision.transforms.Compose object
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Create training and validation data loaders.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for data loaders
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    # Get transforms
    train_transform = get_transforms(image_size=image_size, augment=True)
    val_transform = get_transforms(image_size=image_size, augment=False)
    
    # Create datasets
    train_dataset = QRCodeDataset(train_dir, transform=train_transform)
    val_dataset = QRCodeDataset(val_dir, transform=val_transform)
    
    # Verify classes match
    if train_dataset.classes != val_dataset.classes:
        print("Warning: Train and validation datasets have different classes!")
    
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
    
    return train_loader, val_loader, train_dataset.classes


def get_inference_transform(image_size: int = 224):
    """
    Get transform for inference on single images.
    
    Args:
        image_size: Target image size
        
    Returns:
        torchvision.transforms.Compose object
    """
    return get_transforms(image_size=image_size, augment=False)
