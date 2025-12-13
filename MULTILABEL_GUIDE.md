# 多标签分类使用指南 (Multi-Label Classification Guide)

本项目已升级为**多标签分类系统**,可以同时识别二维码图像的三种属性:

1. **is_copied**: 是否为复印件(介质属性)
2. **is_blurry**: 是否模糊(成像质量)
3. **is_low_light**: 是否暗光(成像质量)

**注意**: 这三个属性是非互斥的,一张图片可以同时具有多个属性(例如:既是复印件,又是暗光拍摄)。

---

## 📋 目录

1. [数据准备](#数据准备)
2. [训练模型](#训练模型)
3. [模型推理](#模型推理)
4. [配置说明](#配置说明)

---

## 数据准备

### 数据格式

多标签分类使用 **JSON 注释文件** 来标注每张图片的属性。

### 目录结构

```
data/
├── train/
│   ├── images/              # 训练图片
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── annotations.json     # 训练标注文件
└── val/
    ├── images/              # 验证图片
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    └── annotations.json     # 验证标注文件
```

### 注释文件格式 (annotations.json)

```json
{
    "img001.jpg": {
        "is_copied": 1,
        "is_blurry": 0,
        "is_low_light": 0
    },
    "img002.jpg": {
        "is_copied": 1,
        "is_blurry": 1,
        "is_low_light": 0
    },
    "img003.jpg": {
        "is_copied": 0,
        "is_blurry": 0,
        "is_low_light": 1
    },
    "img004.jpg": {
        "is_copied": 1,
        "is_blurry": 0,
        "is_low_light": 1
    }
}
```

**说明**:
- 键: 图片文件名(相对于 `images/` 目录)
- 值: 包含三个标签的字典,每个标签的值为 `0` (否) 或 `1` (是)
- 一张图片可以有多个标签为 `1`

### 创建注释文件示例

```python
import json

# 示例数据
annotations = {
    "qr_original_001.jpg": {
        "is_copied": 0,  # 原件
        "is_blurry": 0,  # 清晰
        "is_low_light": 0  # 正常光照
    },
    "qr_copied_blur_001.jpg": {
        "is_copied": 1,  # 复印件
        "is_blurry": 1,  # 模糊
        "is_low_light": 0  # 正常光照
    },
    "qr_copied_dark_001.jpg": {
        "is_copied": 1,  # 复印件
        "is_blurry": 0,  # 清晰
        "is_low_light": 1  # 暗光
    }
}

# 保存为 JSON 文件
with open('data/train/annotations.json', 'w', encoding='utf-8') as f:
    json.dump(annotations, f, indent=4, ensure_ascii=False)
```

---

## 训练模型

### 方法 1: 使用配置文件 (推荐)

1. **编辑 `config.yaml`**:

```yaml
# Data settings
data:
  train_dir: "data/train"
  val_dir: "data/val"
  image_size: 224
  batch_size: 32
  num_workers: 0

# Model settings
model:
  name: "resnet50"
  num_labels: 3  # 三个标签
  pretrained: true
  freeze_backbone: true
  label_names: ["is_copied", "is_blurry", "is_low_light"]

# Training settings
training:
  num_epochs: 20
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "reduce_on_plateau"
```

2. **运行训练**:

```bash
.venv\Scripts\python.exe train_model.py
```

### 方法 2: 使用 Python 代码

```python
from src.train import train_model

history = train_model(
    train_dir='data/train',
    val_dir='data/val',
    model_name='resnet50',
    num_labels=3,
    batch_size=32,
    learning_rate=0.001,
    num_epochs=20,
    freeze_backbone=True,
    multi_label=True,
    label_names=["is_copied", "is_blurry", "is_low_light"]
)
```

### 训练输出

训练完成后,会生成以下文件:

```
checkpoints/
├── best_model.pth           # 最佳模型
├── checkpoint_epoch_1.pth   # 每个 epoch 的检查点
├── checkpoint_epoch_2.pth
├── ...
├── training_history.json    # 训练历史
└── class_names.json         # 标签名称
```

### 训练指标

多标签分类使用以下指标:

- **Mean Accuracy**: 所有标签的平均准确率
- **Per-Label Accuracy**: 每个标签的独立准确率
- **Exact Match Ratio**: 所有标签都预测正确的样本比例

---

## 模型推理

### 方法 1: 使用 Python API

```python
from src.inference import QRCodePredictor

# 初始化预测器
predictor = QRCodePredictor(
    checkpoint_path='checkpoints/best_model.pth',
    model_name='resnet50',
    multi_label=True,
    label_names=["is_copied", "is_blurry", "is_low_light"],
    threshold=0.5  # 二值化阈值
)

# 预测单张图片
result = predictor.predict_image('test_image.jpg')

# 输出结果
print("预测结果:")
for label_name, pred in result['predictions'].items():
    print(f"  {label_name}: {pred['value']} (概率: {pred['probability']:.2%})")
```

**输出示例**:
```
预测结果:
  is_copied: 1 (概率: 92.34%)
  is_blurry: 0 (概率: 15.67%)
  is_low_light: 1 (概率: 78.91%)
```

### 方法 2: 批量预测

```python
# 预测多张图片
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.predict_batch(image_paths)

for result in results:
    print(f"\n图片: {result['image_path']}")
    for label_name, pred in result['predictions'].items():
        print(f"  {label_name}: {pred['value']}")
```

### 结果格式

```python
{
    'image_path': 'test_image.jpg',
    'predictions': {
        'is_copied': {
            'value': 1,           # 0 或 1
            'probability': 0.9234  # 0.0 到 1.0
        },
        'is_blurry': {
            'value': 0,
            'probability': 0.1567
        },
        'is_low_light': {
            'value': 1,
            'probability': 0.7891
        }
    },
    'probabilities': {
        'is_copied': 0.9234,
        'is_blurry': 0.1567,
        'is_low_light': 0.7891
    }
}
```

---

## 配置说明

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_labels` | 标签数量 | 3 |
| `multi_label` | 是否多标签分类 | True |
| `label_names` | 标签名称列表 | `["is_copied", "is_blurry", "is_low_light"]` |
| `threshold` | 二值化阈值 | 0.5 |
| `learning_rate` | 学习率 | 0.001 |
| `batch_size` | 批次大小 | 32 |
| `num_epochs` | 训练轮数 | 20 |

### 模型选择

| 模型 | 参数量 | 速度 | 精度 | 推荐场景 |
|------|--------|------|------|----------|
| ResNet-50 | 25M | 中等 | 高 | 默认选择 |
| ResNet-18 | 11M | 快 | 中等 | 快速训练 |
| EfficientNet-B0 | 5M | 快 | 高 | 高效训练 |
| MobileNet-V3 | 2M | 很快 | 中等 | 移动端部署 |

---

## 常见问题

### 1. 如何调整预测阈值?

```python
predictor = QRCodePredictor(
    checkpoint_path='checkpoints/best_model.pth',
    multi_label=True,
    threshold=0.6  # 提高阈值,减少假阳性
)
```

### 2. 如何处理类别不平衡?

如果某个标签的正样本很少,可以:
- 收集更多该标签的正样本
- 使用数据增强
- 调整损失函数权重(需要修改代码)

### 3. 如何评估模型性能?

查看训练历史:
```python
import json

with open('checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

print(f"最佳验证准确率: {max(history['val_acc']):.4f}")
```

### 4. 如何导出 ONNX 模型?

```python
from src.utils import export_to_onnx

onnx_path = export_to_onnx(
    model_path='checkpoints/best_model.pth',
    output_path='models/qrcode_multilabel.onnx',
    input_size=(3, 224, 224)
)
```

---

## 技术细节

### 损失函数

多标签分类使用 **BCEWithLogitsLoss** (Binary Cross-Entropy with Logits):
- 将每个标签视为独立的二分类问题
- 自动包含 Sigmoid 激活函数
- 数值稳定性更好

### 评估指标

1. **Per-Label Accuracy**: 每个标签的准确率
   ```
   Accuracy_i = (TP_i + TN_i) / N
   ```

2. **Mean Accuracy**: 所有标签准确率的平均值
   ```
   Mean_Accuracy = (Accuracy_1 + Accuracy_2 + Accuracy_3) / 3
   ```

3. **Exact Match Ratio**: 所有标签都预测正确的比例
   ```
   EMR = (完全正确的样本数) / (总样本数)
   ```

---

## 示例代码

完整的训练和推理示例:

```python
# 训练
from src.train import train_model

train_model(
    train_dir='data/train',
    val_dir='data/val',
    model_name='resnet50',
    num_labels=3,
    multi_label=True,
    label_names=["is_copied", "is_blurry", "is_low_light"],
    num_epochs=20
)

# 推理
from src.inference import QRCodePredictor

predictor = QRCodePredictor(
    checkpoint_path='checkpoints/best_model.pth',
    multi_label=True
)

result = predictor.predict_image('test.jpg')
print(result['predictions'])
```

---

## 下一步

- 📊 使用 TensorBoard 监控训练: `tensorboard --logdir=logs`
- 🔧 调整超参数以提高性能
- 📦 导出模型用于部署
- 🧪 在测试集上评估模型

如有问题,请查看项目 README 或创建 Issue。
