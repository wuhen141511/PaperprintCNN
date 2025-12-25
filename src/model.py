"""
Model definition and transfer learning setup for QR code multi-label classification.
Uses pretrained models from torchvision with custom classifier heads.
Supports multi-label classification for: is_copied, is_blurry, is_low_light.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

def _patch_first_layer(layer, in_channels: int):
    """
    Patch the first convolution layer to accept arbitrary input channels.
    Copies weights from the first 3 channels to new channels if increasing.
    """
    if layer.in_channels == in_channels:
        return
        
    print(f"Patching first layer: {layer.in_channels} -> {in_channels} channels")
    
    # Create new layer with same parameters but new in_channels
    new_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        bias=layer.bias is not None,
        padding_mode=layer.padding_mode,
        dilation=layer.dilation,
        groups=layer.groups # Note: groups might need adjustment if it was input-dependent, but usually 1 for first layer
    )
    
    # Initialize weights
    with torch.no_grad():
        if layer.bias is not None:
            new_layer.bias = layer.bias
            
        # Copy existing weights
        # layer.weight shape: (out, in, k, k)
        # If new in_channels > old, we copy old to first few, and init rest
        src_channels = min(layer.in_channels, in_channels)
        new_layer.weight[:, :src_channels] = layer.weight[:, :src_channels]
        
        # Initialize new channels (e.g. with mean of RGB weights or similar)
        # For simplicity, we can copy the mean of RGB weights to the 4th channel
        # or just random init (standard). Pretrained weights usually expect normalized inputs.
        # If we just randomly init, it will learn. 
        # But commonly we reuse one of the channels or average.
        if in_channels > layer.in_channels:
            # Average of original channels
            avg_weight = torch.mean(layer.weight, dim=1, keepdim=True)
            # Assign to extra channels
            for i in range(layer.in_channels, in_channels):
                new_layer.weight[:, i:i+1] = avg_weight
                
    # Replace weights in place? No, we need to return new layer or modify object passing in.
    # The caller passed 'self.backbone.conv1'. This is an object reference.
    # We cannot replace the object reference in the parent by just assigning to 'layer'.
    # We need to modify the parent.
    # The caller code was: 
    # _patch_first_layer(self.backbone.conv1, in_channels)
    # This won't work because we can't replace the layer in the module this way.
    
    # Correction: The caller logic needs to assign the result.
    # But checking my previous edit:
    # _patch_first_layer(self.backbone.conv1, in_channels)
    # This implies I need to modify the layer IN PLACE or Return it and assign it.
    
    # Modification in place (modifying internals of the conv2d object):
    layer.in_channels = in_channels
    layer.weight = nn.Parameter(new_layer.weight)
    if layer.bias is not None:
        layer.bias = nn.Parameter(new_layer.bias)
    # verify strictly
    assert layer.weight.shape[1] == in_channels



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
        freeze_backbone: bool = False,
        in_channels: int = 4
    ):
        """
        Args:
            num_labels: Number of output labels (default: 3 for is_copied, is_blurry, is_low_light)
            model_name: Name of the backbone model ('resnet50', 'resnet18', 'efficientnet_b0', etc.)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights (feature extraction mode)
            in_channels: Number of input channels (default: 4 for RGB + Registered)
        """
        super(QRCodeClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.in_channels = in_channels
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
            
            # Patch first layer if needed
            if in_channels != 3:
                _patch_first_layer(self.backbone.conv1, in_channels)
                
        elif model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
            if in_channels != 3:
                _patch_first_layer(self.backbone.conv1, in_channels)
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
            if in_channels != 3:
                _patch_first_layer(self.backbone.features[0][0], in_channels)
            
        elif model_name == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            
            if in_channels != 3:
                _patch_first_layer(self.backbone.features[0][0], in_channels)
            
        else:
            # Try loading from timm
            try:
                import timm
                # timm supports in_chans argument
                self.backbone = timm.create_model(
                    model_name, 
                    pretrained=pretrained, 
                    num_classes=0,
                    in_chans=in_channels
                )
                num_features = self.backbone.num_features
            except ImportError:
                raise ImportError("Please install 'timm' library to use this model: pip install timm")
            except Exception as e:
                raise ValueError(f"Unsupported model: {model_name}. Error: {str(e)}")
        
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
    device: str = 'cpu',
    in_channels: int = 4
) -> QRCodeClassifier:
    """
    Create and initialize a QR code multi-label classifier model.
    
    Args:
        num_labels: Number of output labels
        model_name: Name of the backbone model
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone weights
        device: Device to place the model on
        in_channels: Number of input channels
        
    Returns:
        QRCodeClassifier model
    """
    model = QRCodeClassifier(
        num_labels=num_labels,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        in_channels=in_channels
    )
    
    model = model.to(device)
    
    print(f"Total parameters: {model.get_num_total_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    return model


def load_model_for_inference(
    checkpoint_path: str, 
    num_labels: int = 3, 
    model_name: str = 'resnet50', 
    device: str = 'cpu',
    in_channels: int = 4
):
    """
    Load a trained model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_labels: Number of labels
        model_name: Name of the model architecture
        device: Device to load model on
        in_channels: Number of input channels
        
    Returns:
        Loaded model in evaluation mode
    """
    model = create_model(
        num_labels=num_labels,
        model_name=model_name,
        pretrained=False,
        freeze_backbone=False,
        device=device,
        in_channels=in_channels
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'mean_accuracy' in checkpoint:
        print(f"Checkpoint mean accuracy: {checkpoint['mean_accuracy']:.4f}")
    
    return model
