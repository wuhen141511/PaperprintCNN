"""
Standalone training script with default configuration.
Quick start script for training a QR code classifier.
"""

from src.train import train_model

if __name__ == '__main__':
    print("\n" + "="*60)
    print("QR Code Classification - Quick Training")
    print("="*60)
    print("This script will train a ResNet-50 model on your QR code dataset (Multi-Label).")
    print("Make sure your data is organized as:")
    print("  data/train/images/    - Training images")
    print("  data/train/annotations.json - Training annotations")
    print("  data/val/images/      - Validation images")
    print("  data/val/annotations.json   - Validation annotations")
    print("="*60 + "\n")
    
    # Train with default settings (Multi-Label)
    # Set load_checkpoint_path to resume training from a checkpoint
    history = train_model(
        train_dir='data/train',
        val_dir='data/val',
        model_name='convnextv2_tiny',
        num_labels=3,  # Multi-label count
        batch_size=32,
        learning_rate=0.001,
        num_epochs=40,
        image_size=384,
        freeze_backbone=True,
        checkpoint_dir='checkpoints',
        log_dir='logs',
        multi_label=True,
        label_names=["is_copied", "is_low_light", "is_blurry"],
        load_checkpoint_path=None,  # Set to checkpoint path to resume training，default is None
        pretrained_path=r'checkpoints/convnextv2_tiny_22k_384_ema.pt'  # Load local weights
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("Best model saved to: checkpoints/best_model.pth")
    print("To view training curves, run: tensorboard --logdir=logs")
    print("="*60)
