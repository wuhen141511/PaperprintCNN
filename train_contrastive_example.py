"""
示例脚本：使用Cross-Attention + QRCodeContrastiveLoss进行训练

这个脚本展示了如何使用对比学习架构来训练QR码分类模型。
"""

from src.train import train_model

def train_with_contrastive():
    """
    使用对比学习架构训练模型
    """
    print("=" * 70)
    print("QR Code Classification with Contrastive Learning")
    print("=" * 70)
    print("\n使用Cross-Attention + QRCodeContrastiveLoss方案")
    print("-" * 70)
    
    # 训练参数
    train_dir = 'data/train'
    val_dir = 'data/val'
    model_name = 'resnet50'
    num_labels = 3
    batch_size = 16  # 对比学习可能需要更多显存，减小batch size
    learning_rate = 1e-4  # 对比学习建议使用较小的学习率
    num_epochs = 50  # 对比学习可能需要更多epoch
    image_size = 224
    freeze_backbone = True  # 初始冻结backbone
    checkpoint_dir = 'checkpoints_contrastive'
    log_dir = 'logs_contrastive'
    multi_label = True
    label_names = ["is_copied", "is_low_light", "is_blurry"]
    
    # 对比学习参数
    use_contrastive = True  # 启用对比学习
    contrastive_weight = 0.3  # 对比损失权重
    classification_weight = 1.0  # 分类损失权重
    
    print(f"\n训练配置:")
    print(f"  - 模型: {model_name}")
    print(f"  - 标签: {label_names}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - 对比学习: {use_contrastive}")
    print(f"  - 对比损失权重: {contrastive_weight}")
    print(f"  - 分类损失权重: {classification_weight}")
    print("-" * 70)
    
    # 开始训练
    history = train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        model_name=model_name,
        num_labels=num_labels,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        image_size=image_size,
        freeze_backbone=freeze_backbone,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        multi_label=multi_label,
        label_names=label_names,
        use_contrastive=use_contrastive,
        contrastive_weight=contrastive_weight,
        classification_weight=classification_weight
    )
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"\n最佳模型保存在: {checkpoint_dir}/best_model.pth")
    print(f"训练历史保存在: {checkpoint_dir}/training_history.json")
    print(f"TensorBoard日志保存在: {log_dir}")
    print("\n使用TensorBoard查看训练曲线:")
    print(f"  tensorboard --logdir={log_dir}")
    print("=" * 70)


def train_without_contrastive():
    """
    使用标准架构训练模型（不使用对比学习）
    """
    print("=" * 70)
    print("QR Code Classification without Contrastive Learning")
    print("=" * 70)
    print("\n使用标准BCEWithLogitsLoss")
    print("-" * 70)
    
    # 训练参数
    train_dir = 'data/train'
    val_dir = 'data/val'
    model_name = 'resnet50'
    num_labels = 3
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    image_size = 224
    freeze_backbone = True
    checkpoint_dir = 'checkpoints_standard'
    log_dir = 'logs_standard'
    multi_label = True
    label_names = ["is_copied", "is_low_light", "is_blurry"]
    
    # 不使用对比学习
    use_contrastive = False
    
    print(f"\n训练配置:")
    print(f"  - 模型: {model_name}")
    print(f"  - 标签: {label_names}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - 对比学习: {use_contrastive}")
    print("-" * 70)
    
    # 开始训练
    history = train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        model_name=model_name,
        num_labels=num_labels,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        image_size=image_size,
        freeze_backbone=freeze_backbone,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        multi_label=multi_label,
        label_names=label_names,
        use_contrastive=use_contrastive
    )
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"\n最佳模型保存在: {checkpoint_dir}/best_model.pth")
    print(f"训练历史保存在: {checkpoint_dir}/training_history.json")
    print(f"TensorBoard日志保存在: {log_dir}")
    print("\n使用TensorBoard查看训练曲线:")
    print(f"  tensorboard --logdir={log_dir}")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'contrastive':
            train_with_contrastive()
        elif mode == 'standard':
            train_without_contrastive()
        else:
            print(f"未知模式: {mode}")
            print("使用方法:")
            print("  python train_contrastive_example.py contrastive  # 使用对比学习")
            print("  python train_contrastive_example.py standard    # 使用标准训练")
    else:
        print("请选择训练模式:")
        print("  python train_contrastive_example.py contrastive  # 使用对比学习")
        print("  python train_contrastive_example.py standard    # 使用标准训练")
        print("\n推荐: 先使用standard训练建立基线，再使用contrastive提升性能")
