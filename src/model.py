"""
Model definition and transfer learning setup for QR code multi-label classification.
Uses pretrained models from torchvision with custom classifier heads.
Supports multi-label classification for: is_copied, is_blurry, is_low_light.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class QRCodeClassifier(nn.Module):
    """
    Transfer learning model for QR code multi-label classification.
    Uses a pretrained backbone with a custom classifier head.
    Outputs 3 independent binary predictions: is_copied, is_blurry, is_low_light.
    """
    
    def __init__(
        self,
        num_labels: int = 3,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_labels: Number of output labels (default: 3 for is_copied, is_blurry, is_low_light)
            model_name: Name of the backbone model ('resnet50', 'resnet18', 'efficientnet_b0', etc.)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights (feature extraction mode)
        """
        super(QRCodeClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
            
        elif model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Backbone frozen - only training classifier head")
        
        # Custom classifier head for multi-label classification
        # Output layer has num_labels neurons (one per label)
        # No sigmoid here - will use BCEWithLogitsLoss which includes sigmoid
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
        
        print(f"Model created: {model_name}")
        print(f"Number of labels: {num_labels} (multi-label classification)")
        print(f"Feature dimension: {num_features}")
    
    def forward(self, x):
        """Forward pass through the network."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end. If None, unfreezes all.
        """
        if num_layers is None:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("All backbone layers unfrozen")
        else:
            # Unfreeze last n layers
            layers = list(self.backbone.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Last {num_layers} backbone layers unfrozen")
    
    def get_num_trainable_params(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self):
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(
    num_labels: int = 3,
    model_name: str = 'resnet50',
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = 'cpu'
) -> QRCodeClassifier:
    """
    Create and initialize a QR code multi-label classifier model.
    
    Args:
        num_labels: Number of output labels (default: 3 for is_copied, is_blurry, is_low_light)
        model_name: Name of the backbone model
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone weights
        device: Device to place the model on
        
    Returns:
        QRCodeClassifier model
    """
    model = QRCodeClassifier(
        num_labels=num_labels,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    model = model.to(device)
    
    print(f"Total parameters: {model.get_num_total_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    return model


def load_model_for_inference(checkpoint_path: str, num_labels: int = 3, model_name: str = 'resnet50', device: str = 'cpu'):
    """
    Load a trained model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_labels: Number of labels (default: 3 for multi-label classification)
        model_name: Name of the model architecture
        device: Device to load model on
        
    Returns:
        Loaded model in evaluation mode
    """
    model = create_model(
        num_labels=num_labels,
        model_name=model_name,
        pretrained=False,
        freeze_backbone=False,
        device=device
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'mean_accuracy' in checkpoint:
        print(f"Checkpoint mean accuracy: {checkpoint['mean_accuracy']:.4f}")
    
    return model
