"""
Utility functions for QR code classification system.
Includes device configuration, random seed setting, and helper functions.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import os
import json
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the best available device (CUDA GPU if available, else CPU).
    
    Returns:
        torch.device: The device to use for training/inference
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # If CUDA is unavailable, use mps if available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath, model_type=None, model_name=None, num_labels=None):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
        model_type: Type of model ('regular' or 'contrastive')
        model_name: Name of the backbone model (e.g., 'resnet50', 'convnextv2_tiny')
        num_labels: Number of labels (for multi-label) or classes (for single-class)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'model_type': model_type,
        'model_name': model_name,
        'num_labels': num_labels,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, optimizer, epoch, loss, accuracy)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    print(f"Checkpoint loaded from {filepath}")
    return model, optimizer, epoch, loss, accuracy


def calculate_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels
        
    Returns:
        Dictionary containing accuracy and other metrics
    """
    pred_classes = torch.argmax(predictions, dim=1)
    correct = (pred_classes == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    
    # Calculate per-class metrics
    num_classes = predictions.size(1)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    for i in range(total):
        label = labels[i].item()
        class_total[label] += 1
        if pred_classes[i] == label:
            class_correct[label] += 1
    
    class_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[f'class_{i}_accuracy'] = class_correct[i] / class_total[i]
    
    return {
        'accuracy': accuracy,
        **class_accuracy
    }


def calculate_multilabel_metrics(predictions: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate multi-label classification metrics.
    
    Args:
        predictions: Model predictions (logits), shape: (batch_size, num_labels)
        labels: Ground truth labels (0 or 1), shape: (batch_size, num_labels)
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Dictionary containing per-label accuracy, mean accuracy, and other metrics
    """
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(predictions)
    
    # Convert probabilities to binary predictions
    pred_labels = (probs >= threshold).float()
    
    # Calculate per-label accuracy
    num_labels = predictions.size(1)
    label_accuracies = []
    
    for i in range(num_labels):
        correct = (pred_labels[:, i] == labels[:, i]).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        label_accuracies.append(accuracy)
    
    # Calculate mean accuracy across all labels
    mean_accuracy = sum(label_accuracies) / num_labels
    
    # Calculate exact match ratio (all labels must be correct)
    exact_match = (pred_labels == labels).all(dim=1).sum().item()
    exact_match_ratio = exact_match / labels.size(0)
    
    # Build result dictionary
    result = {
        'mean_accuracy': mean_accuracy,
        'exact_match_ratio': exact_match_ratio
    }
    
    # Add per-label accuracies
    for i, acc in enumerate(label_accuracies):
        result[f'label_{i}_accuracy'] = acc
    
    return result


def split_train_val_data(
    train_dir: str = '/data/train',
    val_dir: str = '/data/val',
    val_ratio: float = 0.2,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Split training data into train and validation sets by moving files.
    Maintains the same class structure in both directories.
    
    Args:
        train_dir: Path to training directory containing class subdirectories
        val_dir: Path to validation directory (will be created if doesn't exist)
        val_ratio: Ratio of data to move to validation set (0.0 to 1.0)
        seed: Random seed for reproducibility
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing statistics about the split
        
    Example:
        >>> stats = split_train_val_data(
        ...     train_dir='/data/train',
        ...     val_dir='/data/val',
        ...     val_ratio=0.2
        ... )
        >>> print(f"Moved {stats['total_moved']} images to validation set")
    """
    import shutil
    from pathlib import Path
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Convert to Path objects
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    
    # Validate inputs
    if not train_path.exists():
        raise ValueError(f"Training directory does not exist: {train_dir}")
    
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {val_ratio}")
    
    # Create validation directory if it doesn't exist
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        'total_moved': 0,
        'total_remaining': 0,
        'classes': {},
        'val_ratio_actual': 0.0
    }
    
    # Get all class directories
    class_dirs = [d for d in train_path.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {train_dir}")
    
    if verbose:
        print(f"Found {len(class_dirs)} classes in {train_dir}")
        print(f"Target validation ratio: {val_ratio:.1%}")
        print("-" * 60)
    
    # Process each class
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        
        # Get all image files in this class
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        image_files = [
            f for f in class_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if len(image_files) == 0:
            if verbose:
                print(f"⚠️  Class '{class_name}': No images found, skipping")
            continue
        
        # Calculate number of files to move
        num_total = len(image_files)
        num_val = max(1, int(num_total * val_ratio))  # At least 1 image for validation
        num_train = num_total - num_val
        
        # Randomly shuffle and select files for validation
        random.shuffle(image_files)
        val_files = image_files[:num_val]
        
        # Create validation class directory
        val_class_dir = val_path / class_name
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Move files to validation directory
        moved_count = 0
        for img_file in val_files:
            dest_file = val_class_dir / img_file.name
            
            # Handle duplicate filenames
            if dest_file.exists():
                base_name = img_file.stem
                extension = img_file.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = val_class_dir / f"{base_name}_{counter}{extension}"
                    counter += 1
            
            # Move the file
            shutil.move(str(img_file), str(dest_file))
            moved_count += 1
        
        # Update statistics
        stats['total_moved'] += moved_count
        stats['total_remaining'] += num_train
        stats['classes'][class_name] = {
            'total': num_total,
            'moved_to_val': moved_count,
            'remaining_in_train': num_train,
            'actual_val_ratio': moved_count / num_total
        }
        
        if verbose:
            print(f"✓ Class '{class_name}': {moved_count}/{num_total} images moved "
                  f"({moved_count/num_total:.1%}) | {num_train} remaining")
    
    # Calculate overall statistics
    total_images = stats['total_moved'] + stats['total_remaining']
    stats['val_ratio_actual'] = stats['total_moved'] / total_images if total_images > 0 else 0.0
    
    if verbose:
        print("-" * 60)
        print(f"✅ Split completed successfully!")
        print(f"   Total images processed: {total_images}")
        print(f"   Moved to validation: {stats['total_moved']} ({stats['val_ratio_actual']:.1%})")
        print(f"   Remaining in training: {stats['total_remaining']}")
        print(f"   Validation directory: {val_dir}")
    
    return stats


def export_to_onnx(
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    input_size: Tuple[int, int, int] = (4, 384, 384),
    batch_size: int = 1,
    opset_version: int = 11,
    dynamic_axes: bool = True,
    verbose: bool = True,
    version: Optional[str] = None
) -> str:
    """
    Export PyTorch model (.pth) to ONNX format for C++ and OpenCV deployment.
    Supports both QRCodeClassifier and CrossAttentionQRCodeClassifier.
    
    Args:
        model_path: Path to the PyTorch checkpoint (.pth file).
                   If None, uses 'checkpoints/best_model.pth'
        output_path: Path to save the ONNX model. 
                    If None, saves in the same directory as model_path with .onnx extension
        input_size: Input tensor size as (channels, height, width). 
                   Default: (4, 350, 350) for CrossAttentionQRCodeClassifier
                   Use (3, 350, 350) for QRCodeClassifier
        batch_size: Batch size for the dummy input. Default: 1
        opset_version: ONNX opset version. Default: 11 (compatible with most OpenCV versions)
        dynamic_axes: Whether to use dynamic batch size. Default: True
        verbose: Whether to print detailed information. Default: True
        version: Version number to add to ONNX model metadata. Default: None
        
    Returns:
        Path to the exported ONNX model
        
    Example:
        >>> # Export default model (CrossAttentionQRCodeClassifier with 4 channels)
        >>> onnx_path = export_to_onnx()
        
        >>> # Export specific model
        >>> onnx_path = export_to_onnx(
        ...     model_path='checkpoints/epoch_10.pth',
        ...     output_path='models/qrcode_classifier.onnx'
        ... )
        
        >>> # Export QRCodeClassifier (3 channels)
        >>> onnx_path = export_to_onnx(
        ...     model_path='checkpoints/regular_model.pth',
        ...     input_size=(3, 350, 350)
        ... )
    """
    # Import required modules
    try:
        import torch.onnx
    except ImportError:
        raise ImportError("PyTorch is required for ONNX export")
    
    # Set default model path if not provided
    if model_path is None:
        model_path = 'checkpoints/best_model.pth'
    
    model_path = Path(model_path)
    
    # Validate model path
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Set output path
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}.onnx"
    else:
        output_path = Path(output_path)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 70)
        print("PyTorch to ONNX Model Export")
        print("=" * 70)
        print(f"Input model:     {model_path}")
        print(f"Output ONNX:     {output_path}")
        print(f"Input size:      {batch_size} x {input_size[0]} x {input_size[1]} x {input_size[2]}")
        print(f"Opset version:   {opset_version}")
        print(f"Dynamic axes:    {dynamic_axes}")
        print("-" * 70)
    
    # Load the model
    device = torch.device('cpu')  # Export on CPU for compatibility
    
    if verbose:
        print("Loading PyTorch model...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Import model class
    from src.model import QRCodeClassifier, CrossAttentionQRCodeClassifier, create_model, create_contrastive_model
    
    # Detect model type from checkpoint or state_dict
    model_type = checkpoint.get('model_type', None)
    
    if model_type is None:
        # Try to infer from state dict structure
        state_dict = checkpoint['model_state_dict']
        
        # Check for CrossAttentionQRCodeClassifier indicators
        has_rgb_backbone = any('rgb_backbone' in key for key in state_dict.keys())
        has_ref_backbone = any('ref_backbone' in key for key in state_dict.keys())
        has_cross_attention = any('cross_attention' in key for key in state_dict.keys())
        
        if has_rgb_backbone or has_ref_backbone or has_cross_attention:
            model_type = 'contrastive'
        else:
            model_type = 'regular'
    
    if verbose:
        print(f"Detected model type: {model_type}")
    
    # Get model configuration
    if 'num_labels' in checkpoint:
        num_labels = checkpoint['num_labels']
    else:
        # Try to infer from state dict
        try:
            if model_type == 'contrastive':
                num_labels = checkpoint['model_state_dict']['classifier.6.weight'].shape[0]
            else:
                num_labels = checkpoint['model_state_dict']['classifier.4.weight'].shape[0]
        except:
            num_labels = 4  # Default to 4 for multi-label mode
    
    # Get backbone model name
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    else:
        # 直接错报
        raise ValueError("Error: 'model_type' and 'num_labels' are required if 'model_name' not found in checkpoint")
    
    # Create model based on type
    if model_type == 'contrastive':
        # Create CrossAttentionQRCodeClassifier
        model = create_contrastive_model(
            num_labels=num_labels,
            model_name=model_name,
            pretrained=False,
            freeze_backbone=False,
            device=device
        )
        
        # For CrossAttentionQRCodeClassifier, we need 4 channels
        if input_size[0] != 4:
            if verbose:
                print(f"⚠ Warning: CrossAttentionQRCodeClassifier requires 4-channel input")
                print(f"  Adjusting input_size from {input_size} to (4, {input_size[1]}, {input_size[2]})")
            input_size = (4, input_size[1], input_size[2])
    else:
        # Create QRCodeClassifier
        model = create_model(
            num_labels=num_labels,
            model_name=model_name,
            pretrained=False,
            freeze_backbone=False,
            device=device,
            use_contrastive=False,
        )
        
        # For QRCodeClassifier, we need 3 channels
        if input_size[0] != 3:
            if verbose:
                print(f"⚠ Warning: QRCodeClassifier requires 3-channel input")
                print(f"  Adjusting input_size from {input_size} to (3, {input_size[1]}, {input_size[2]})")
            input_size = (3, input_size[1], input_size[2])
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    if verbose:
        print(f"✓ Model loaded successfully")
        print(f"  - Architecture: {model_name}")
        print(f"  - Type:         {model_type}")
        print(f"  - Classes:      {num_labels}")
        if 'epoch' in checkpoint:
            print(f"  - Epoch:        {checkpoint['epoch']}")
        if 'accuracy' in checkpoint:
            print(f"  - Accuracy:     {checkpoint['accuracy']:.4f}")
        print("-" * 70)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_size, device=device)
    
    if verbose:
        print("Exporting to ONNX format...")
    
    # Define dynamic axes for variable batch size
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        dynamic_axes_dict = None
    
    # Export to ONNX
    try:
        # Use legacy exporter for better compatibility
        # The new dynamo-based exporter has compatibility issues with some onnxscript versions
        torch.onnx.export(
            model,                          # Model to export
            dummy_input,                    # Dummy input
            str(output_path),              # Output path
            export_params=True,            # Store trained parameters
            opset_version=opset_version,   # ONNX version
            do_constant_folding=True,      # Optimize constant folding
            input_names=['input'],         # Input names
            output_names=['output'],       # Output names
            dynamic_axes=dynamic_axes_dict,# Dynamic axes
            dynamo=False                   # Use legacy TorchScript-based exporter
        )
        
        if verbose:
            print(f"✓ ONNX export successful!")
            print("-" * 70)
            
        # Verify the exported model
        try:
            import onnx
            from onnx import helper
            
            onnx_model = onnx.load(str(output_path))
            
            # Add training info metadata
            train_info = {}
            if 'epoch' in checkpoint:
                train_info['epoch'] = checkpoint['epoch']
            if 'accuracy' in checkpoint:
                train_info['accuracy'] = round(float(checkpoint['accuracy']), 4)
            if 'loss' in checkpoint:
                train_info['loss'] = round(float(checkpoint['loss']), 4)

            if train_info:
                meta = onnx_model.metadata_props.add()
                meta.key = "custom_info"
                meta.value = json.dumps(train_info)
                onnx.save(onnx_model, str(output_path))
                if verbose:
                    print(f"✓ Added training info to ONNX metadata: {train_info}")

            # Add version metadata if provided
            if version is not None:
                meta = onnx_model.metadata_props.add()
                meta.key = "custom_version"
                meta.value = version
                onnx.save(onnx_model, str(output_path))
                if verbose:
                    print(f"✓ Added version {version} to ONNX model metadata")
            
            onnx.checker.check_model(onnx_model)
            
            if verbose:
                print("✓ ONNX model verification passed")
                
                # Get model size
                model_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  - Model size: {model_size_mb:.2f} MB")
                
                # Print input/output info
                print("\nModel I/O Information:")
                print(f"  Input:  {onnx_model.graph.input[0].name}")
                print(f"          Shape: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
                print(f"  Output: {onnx_model.graph.output[0].name}")
                print(f"          Shape: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")
                
        except ImportError:
            if verbose:
                print("⚠ ONNX package not found - skipping verification")
                print("  Install with: pip install onnx")
        except Exception as e:
            if verbose:
                print(f"⚠ ONNX verification warning: {e}")
        
        if verbose:
            print("=" * 70)
            print("Export completed successfully!")
            print(f"ONNX model saved to: {output_path.absolute()}")
            print("=" * 70)
            print("\nUsage in C++ with OpenCV:")
            print("  cv::dnn::Net net = cv::dnn::readNetFromONNX(\"model.onnx\");")
            print("  cv::Mat blob = cv::dnn::blobFromImage(image, 1.0/255.0, ")
            print(f"                 cv::Size({input_size[2]}, {input_size[1]}), ")
            print("                 cv::Scalar(0.485, 0.456, 0.406), true, false);")
            print("  net.setInput(blob);")
            print("  cv::Mat output = net.forward();")
            print("=" * 70)
        
        return str(output_path.absolute())
        
    except Exception as e:
        raise RuntimeError(f"Failed to export model to ONNX: {e}")


class QRCodeContrastiveLoss(nn.Module):
    """
    专门针对QR码分类的对比损失
    
    核心思想：
    - 原生图：RGB特征 ≈ 参考图特征（高相似度）
    - 非原生图：RGB特征 ≠ 参考图特征（低相似度）
    """
    def __init__(self, margin=1.0, temperature=0.5):
        """
        Args:
            margin: 边界参数，控制正负样本对的距离
            temperature: 温度参数，控制相似度的缩放
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, rgb_feat, ref_feat, labels):
        """
        计算对比损失
        
        Args:
            rgb_feat: (B, D) RGB特征向量
            ref_feat: (B, D) 参考图特征向量
            labels: (B, 3) 多标签 [is_copied, is_low_light, is_blurry]
        
        Returns:
            loss: 对比损失值
        """
        # 归一化特征向量
        rgb_feat = F.normalize(rgb_feat, dim=1)
        ref_feat = F.normalize(ref_feat, dim=1)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(rgb_feat, ref_feat, dim=1)  # (B,)
        
        # 判断是否为原生图（所有标签都为0）
        is_native = (labels[:, 0] == 0) & (labels[:, 1] == 0) & (labels[:, 2] == 0)
        
        # 原生图：相似度应该高（接近1）
        # 非原生图：相似度应该低（接近0）
        target_similarity = is_native.float()
        
        # 使用温度缩放相似度
        similarity_scaled = similarity / self.temperature
        
        # 计算二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(
            similarity_scaled,
            target_similarity
        )
        
        return loss

