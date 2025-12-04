"""
Standalone training script with default configuration.
Quick start script for training a QR code classifier.
"""

from src.train import train_model

if __name__ == '__main__':
    print("\n" + "="*60)
    print("QR Code Classification - Quick Training")
    print("="*60)
    print("This script will train a ResNet-50 model on your QR code dataset.")
    print("Make sure your data is organized as:")
    print("  data/train/original/  - Original QR code images")
    print("  data/train/copied/    - Copied QR code images")
    print("  data/val/original/    - Validation original images")
    print("  data/val/copied/      - Validation copied images")
    print("="*60 + "\n")
    
    # Train with default settings
    history = train_model(
        train_dir='data/train',
        val_dir='data/val',
        model_name='resnet50',
        num_classes=2,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=20,
        image_size=224,
        freeze_backbone=True,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("Best model saved to: checkpoints/best_model.pth")
    print("To view training curves, run: tensorboard --logdir=logs")
    print("="*60)
