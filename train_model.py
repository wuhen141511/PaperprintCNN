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
        model_name='convnextv2_pico',
        num_labels=4,  # Multi-label count
        batch_size=16,  # 减小batch size适应384x384输入
        learning_rate=1e-4,  # 使用较小学习率
        num_epochs=40,  # 增加epoch数
        image_size=384,  # 使用384x384输入尺寸
        freeze_backbone=True,
        checkpoint_dir='checkpoints',
        log_dir='logs',
        multi_label=True,
        label_names=["is_copied", "is_low_light", "is_blurry", "is_screen"],
        load_checkpoint_path= None,  # Set to checkpoint path to resume training，default is None
        pretrained_path=None,  # Load local weights
        use_contrastive=False,  # 启用对比学习
        contrastive_weight=0.3,  # 对比学习权重
        classification_weight=1.0,  # 分类损失权重
        use_gray=False
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("Best model saved to: checkpoints/best_model.pth")
    print("To view training curves, run: tensorboard --logdir=logs")
    print("="*60)
