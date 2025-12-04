"""
Utility functions for QR code classification system.
Includes device configuration, random seed setting, and helper functions.
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os


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
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
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


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def visualize_predictions(images, predictions, labels, class_names, num_images=8):
    """
    Visualize model predictions on a batch of images.
    
    Args:
        images: Batch of images (tensor)
        predictions: Model predictions (logits)
        labels: Ground truth labels
        class_names: List of class names
        num_images: Number of images to display
    """
    num_images = min(num_images, len(images))
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    pred_classes = torch.argmax(predictions, dim=1)
    probs = torch.softmax(predictions, dim=1)
    
    for i in range(num_images):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        pred_class = pred_classes[i].item()
        true_class = labels[i].item()
        confidence = probs[i][pred_class].item()
        
        color = 'green' if pred_class == true_class else 'red'
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(
            f'Pred: {class_names[pred_class]} ({confidence:.2%})\n'
            f'True: {class_names[true_class]}',
            color=color,
            fontsize=10
        )
    
    plt.tight_layout()
    plt.show()
