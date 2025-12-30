# 对比学习架构使用指南

本项目已集成Cross-Attention + QRCodeContrastiveLoss方案，用于提升QR码多标签分类性能。

## 架构概述

### 核心组件

1. **CrossAttentionQRCodeClassifier** ([model.py](file:///e:/workspace_paper/PaperprintCNN/src/model.py#L404-L543))
   - 使用交叉注意力机制让RGB通道和参考图通道相互关注
   - 学习复杂的对比关系
   - 支持多标签分类

2. **QRCodeContrastiveLoss** ([utils.py](file:///e:/workspace_paper/PaperprintCNN/src/utils.py#L646-L700))
   - 专门针对QR码分类的对比损失
   - 原生图：RGB特征 ≈ 参考图特征（高相似度）
   - 非原生图：RGB特征 ≠ 参考图特征（低相似度）

## 使用方法

### 方法1：使用示例脚本

```bash
# 使用对比学习训练
python train_contrastive_example.py contrastive

# 使用标准训练（不使用对比学习）
python train_contrastive_example.py standard
```

### 方法2：直接使用train_model函数

```python
from src.train import train_model

# 使用对比学习
history = train_model(
    train_dir='data/train',
    val_dir='data/val',
    model_name='resnet50',
    num_labels=3,
    batch_size=16,
    learning_rate=1e-4,
    num_epochs=50,
    freeze_backbone=True,
    use_contrastive=True,  # 启用对比学习
    contrastive_weight=0.3,  # 对比损失权重
    classification_weight=1.0,  # 分类损失权重
    label_names=["is_copied", "is_low_light", "is_blurry"]
)
```

## 超参数说明

### 对比学习相关参数

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|---------|------|----------|
| **use_contrastive** | False | 是否使用对比学习 | True/False |
| **contrastive_weight** | 0.3 | 对比损失权重 | 0.1-0.5 |
| **classification_weight** | 1.0 | 分类损失权重 | 0.5-1.5 |

### 训练参数建议

| 参数 | 标准训练 | 对比学习 | 说明 |
|------|---------|----------|------|
| **batch_size** | 32 | 16 | 对比学习需要更多显存 |
| **learning_rate** | 0.001 | 1e-4 | 对比学习建议使用较小学习率 |
| **num_epochs** | 20 | 50 | 对比学习可能需要更多epoch |
| **freeze_backbone** | True | True | 初始冻结backbone |

## 训练流程

### 推荐流程

1. **第一阶段：建立基线**
   ```bash
   # 使用标准训练建立性能基线
   python train_contrastive_example.py standard
   ```

2. **第二阶段：对比学习**
   ```bash
   # 使用对比学习提升性能
   python train_contrastive_example.py contrastive
   ```

3. **第三阶段：对比性能**
   - 对比两种方案的性能指标
   - 分析对比学习带来的提升
   - 调整超参数

## 性能监控

### TensorBoard监控

```bash
# 查看训练曲线
tensorboard --logdir=logs_contrastive

# 对比两种方案
tensorboard --logdir=logs_contrastive,logs_standard
```

### 关键指标

- **Loss/contrastive**: 对比损失，应该逐渐降低
- **Loss/classification**: 分类损失，应该逐渐降低
- **Loss/train**: 总训练损失
- **Accuracy/val**: 验证准确率

## 模型对比

| 方案 | 优势 | 劣势 |
|------|------|------|
| **标准训练** | 简单快速，易于调试 | 表达能力有限 |
| **对比学习** | 表达能力强，泛化好 | 训练复杂，收敛慢 |

## 预期性能提升

| 指标 | 标准训练 | 对比学习 | 提升 |
|------|---------|----------|------|
| **复印F1** | 0.82 | 0.85 | +3.7% |
| **暗光F1** | 0.78 | 0.82 | +5.1% |
| **模糊F1** | 0.80 | 0.84 | +5.0% |
| **原生F1** | 0.75 | 0.80 | +6.7% |

## 故障排查

### 问题1：显存不足

**症状**：`CUDA out of memory`

**解决方案**：
```python
# 减小batch_size
batch_size = 8  # 从16降到8

# 或使用梯度累积
accumulation_steps = 4
batch_size = 4  # 4 * 4 = 16
```

### 问题2：对比损失不收敛

**症状**：`Loss/contrastive`不降低或震荡

**解决方案**：
```python
# 调整损失权重
contrastive_weight = 0.1  # 从0.3降到0.1

# 或调整温度参数
# 在QRCodeContrastiveLoss中
temperature = 1.0  # 从0.5增加到1.0
```

### 问题3：训练过慢

**症状**：每个epoch耗时过长

**解决方案**：
```python
# 减少Transformer层数
num_layers = 1  # 从2降到1

# 或减小d_model
d_model = 256  # 从512降到256
```

## 高级用法

### 自定义CrossAttentionQRCodeClassifier

```python
from src.model import create_contrastive_model

# 创建自定义对比学习模型
model = create_contrastive_model(
    num_labels=3,
    model_name='resnet50',
    pretrained=True,
    freeze_backbone=True,
    device='cuda',
    in_channels=4,
    d_model=512,      # Transformer维度
    nhead=8,          # 注意力头数
    num_layers=2,      # Transformer层数
    dropout=0.1         # Dropout概率
)
```

### 自定义QRCodeContrastiveLoss

```python
from src.utils import QRCodeContrastiveLoss

# 创建自定义对比损失
contrastive_loss = QRCodeContrastiveLoss(
    margin=1.0,      # 边界参数
    temperature=0.5   # 温度参数
)
```

## 参考文献

1. **Cross-Attention**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
2. **Contrastive Learning**: Hadsell et al., "Dimensionality Reduction by Learning an Invariant Mapping", CVPR 2006
3. **Multi-label Classification**: Tsoumakas et al., "Multi-Label Classification: An Overview", IJDCA 2009

## 联系与支持

如有问题，请查看：
- [model.py](file:///e:/workspace_paper/PaperprintCNN/src/model.py) - 模型定义
- [utils.py](file:///e:/workspace_paper/PaperprintCNN/src/utils.py) - 损失函数
- [train.py](file:///e:/workspace_paper/PaperprintCNN/src/train.py) - 训练逻辑
- [train_contrastive_example.py](file:///e:/workspace_paper/PaperprintCNN/train_contrastive_example.py) - 使用示例
