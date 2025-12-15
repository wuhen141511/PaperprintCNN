# 项目改造总结 - 多标签分类升级

## 概述

您的二维码分类项目已成功升级为**多标签分类系统**。现在可以同时识别二维码图像的三种属性：

1. **is_copied** - 是否为复印件
2. **is_blurry** - 是否模糊  
3. **is_low_light** - 是否暗光

---

## 主要变更

### 1. 模型架构 (`src/model.py`)

**变更内容**:
- ✅ 参数名从 `num_classes` 改为 `num_labels`
- ✅ 输出层改为 3 个神经元（每个标签一个）
- ✅ 移除 Softmax，使用 Sigmoid 激活（通过 BCEWithLogitsLoss 隐式应用）

**关键代码**:
```python
# 旧版本（二分类）
num_classes: int = 2
nn.Linear(512, num_classes)

# 新版本（多标签）
num_labels: int = 3
nn.Linear(512, num_labels)  # 输出 3 个独立的二值预测
```

### 2. 数据集 (`src/dataset.py`)

**新增内容**:
- ✅ 新增 `QRCodeMultiLabelDataset` 类
- ✅ 支持 JSON 格式的标注文件
- ✅ 返回多标签向量（shape: [3]）而非单一类别索引

**数据格式**:
```json
{
    "image.jpg": {
        "is_copied": 1,
        "is_blurry": 0,
        "is_low_light": 1
    }
}
```

### 3. 训练逻辑 (`src/train.py`)

**变更内容**:
- ✅ 损失函数: `CrossEntropyLoss` → `BCEWithLogitsLoss`
- ✅ 评估指标: 单一准确率 → 每个标签的准确率 + 平均准确率
- ✅ 新增 `multi_label` 参数控制分类模式

**关键变更**:
```python
# 损失函数
if multi_label:
    criterion = nn.BCEWithLogitsLoss()  # 多标签
else:
    criterion = nn.CrossEntropyLoss()   # 单分类

# 评估指标
if multi_label:
    metrics = calculate_multilabel_metrics(predictions, labels)
    accuracy = metrics['mean_accuracy']
else:
    metrics = calculate_metrics(predictions, labels)
    accuracy = metrics['accuracy']
```

### 4. 推理 (`src/inference.py`)

**变更内容**:
- ✅ `QRCodePredictor` 支持多标签模式
- ✅ 输出格式包含每个标签的预测值和概率
- ✅ 可调节的二值化阈值（默认 0.5）

**输出格式**:
```python
{
    'predictions': {
        'is_copied': {'value': 1, 'probability': 0.92},
        'is_blurry': {'value': 0, 'probability': 0.16},
        'is_low_light': {'value': 1, 'probability': 0.79}
    }
}
```

### 5. 工具函数 (`src/utils.py`)

**新增内容**:
- ✅ `calculate_multilabel_metrics()` - 计算多标签指标
  - Per-label accuracy（每个标签的准确率）
  - Mean accuracy（平均准确率）
  - Exact match ratio（完全匹配率）

### 6. 配置文件 (`config.yaml`)

**变更内容**:
```yaml
model:
  num_labels: 3  # 从 num_classes: 2 改为 num_labels: 3
  label_names: ["is_copied", "is_blurry", "is_low_light"]
```

---

## 新增文件

### 1. `MULTILABEL_GUIDE.md`
完整的多标签分类使用指南，包括：
- 数据准备详细说明
- 训练流程
- 推理示例
- 配置参数说明

### 2. `create_annotations.py`
标注文件管理工具，支持：
- 创建标注模板
- 验证标注文件
- 合并多个标注文件

**使用示例**:
```bash
# 创建模板
python create_annotations.py create data/train/images

# 验证标注
python create_annotations.py validate data/train/annotations.json

# 合并标注
python create_annotations.py merge file1.json file2.json -o merged.json
```

### 3. `example_multilabel.py`
完整的工作流程示例，展示：
- 如何创建标注文件
- 如何训练模型
- 如何进行推理
- 批量预测示例

---

## 使用方法

### 快速开始

1. **准备数据**:
```bash
# 创建目录结构
mkdir -p data/train/images data/val/images

# 将图片放入 images 目录

# 创建标注模板
python create_annotations.py create data/train/images
python create_annotations.py create data/val/images

# 编辑 annotations.json，设置正确的标签值
```

2. **训练模型**:
```python
from src.train import train_model

train_model(
    train_dir='data/train',
    val_dir='data/val',
    model_name='resnet50',
    num_labels=3,
    multi_label=True,
    num_epochs=20
)
```

3. **推理**:
```python
from src.inference import QRCodePredictor

predictor = QRCodePredictor(
    checkpoint_path='checkpoints/best_model.pth',
    multi_label=True
)

result = predictor.predict_image('test.jpg')
print(result['predictions'])
```

---

## 兼容性说明

### 向后兼容

项目仍然支持单分类模式（二分类），通过设置 `multi_label=False` 即可：

```python
# 单分类模式（原有功能）
train_model(
    train_dir='data/train',
    val_dir='data/val',
    num_labels=2,
    multi_label=False
)
```

### 数据格式

- **多标签**: 使用 JSON 标注文件
- **单分类**: 使用目录结构（保持原有方式）

---

## 性能指标

### 多标签评估指标

1. **Per-Label Accuracy**: 每个标签的独立准确率
   ```
   is_copied accuracy: 95.2%
   is_blurry accuracy: 92.8%
   is_low_light accuracy: 89.5%
   ```

2. **Mean Accuracy**: 平均准确率
   ```
   Mean Accuracy = (95.2 + 92.8 + 89.5) / 3 = 92.5%
   ```

3. **Exact Match Ratio**: 所有标签都正确的比例
   ```
   Exact Match Ratio = 85.3%
   ```

---

## 常见问题

### Q1: 如何从二分类模型迁移到多标签？

**A**: 需要重新训练模型，因为：
- 输出层结构不同（2 个神经元 vs 3 个神经元）
- 损失函数不同（CrossEntropy vs BCEWithLogits）
- 数据格式不同（类别索引 vs 多标签向量）

### Q2: 可以只识别部分标签吗？

**A**: 可以，通过修改 `label_names` 参数：
```python
label_names = ["is_copied", "is_blurry"]  # 只识别两个标签
```

### Q3: 如何处理标签不平衡？

**A**: 
- 收集更多少数类样本
- 使用数据增强
- 调整损失函数权重（需修改代码）
- 调整预测阈值

### Q4: 如何调整预测阈值？

**A**:
```python
predictor = QRCodePredictor(
    checkpoint_path='checkpoints/best_model.pth',
    multi_label=True,
    threshold=0.6  # 提高阈值，减少假阳性
)
```

---

## 技术细节

### 损失函数对比

| 特性 | CrossEntropyLoss | BCEWithLogitsLoss |
|------|------------------|-------------------|
| 用途 | 单分类 | 多标签分类 |
| 输出 | 类别概率分布 | 每个标签的独立概率 |
| 激活函数 | Softmax | Sigmoid |
| 标签互斥性 | 互斥 | 非互斥 |

### 模型输出

**单分类**:
```python
outputs = [2.3, -1.5]  # logits for 2 classes
probs = softmax(outputs) = [0.98, 0.02]  # sum = 1.0
```

**多标签**:
```python
outputs = [2.3, -1.5, 0.8]  # logits for 3 labels
probs = sigmoid(outputs) = [0.91, 0.18, 0.69]  # independent
```

---

## 下一步建议

1. ✅ 准备标注数据（最重要）
2. ✅ 使用小数据集验证流程
3. ✅ 训练完整模型
4. ✅ 评估模型性能
5. ✅ 调整超参数优化
6. ✅ 导出 ONNX 用于部署

---

## 文档资源

- 📖 `MULTILABEL_GUIDE.md` - 详细使用指南
- 📖 `README.md` - 项目概述
- 📖 `DATA_GUIDE.md` - 数据准备指南
- 📖 `INTEGRATION_GUIDE.md` - 集成部署指南

---

## 联系支持

如有问题，请：
1. 查看 `MULTILABEL_GUIDE.md` 详细文档
2. 运行 `python example_multilabel.py` 查看示例
3. 创建 GitHub Issue

---

**祝您使用愉快！** 🎉
