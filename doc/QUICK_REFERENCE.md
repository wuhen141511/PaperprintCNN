# 多标签分类快速参考

## 一、数据准备

### 创建标注模板
```bash
python create_annotations.py create data/train/images
python create_annotations.py create data/val/images
```

### 标注文件格式
```json
{
    "image.jpg": {
        "is_copied": 1,
        "is_blurry": 0,
        "is_low_light": 1
    }
}
```

### 验证标注
```bash
python create_annotations.py validate data/train/annotations.json
```

### 分割数据集
```bash
# 自动检测模式，分割为训练集和验证集（20%）
python split_dataset.py --ratio 0.2

# 手动指定多标签模式
python split_dataset.py --mode multi-label --ratio 0.2

# 自定义比例（30% 验证集）
python split_dataset.py --ratio 0.3
```

---

## 二、训练模型

### 基础训练
```python
from src.train import train_model

train_model(
    train_dir='data/train',
    val_dir='data/val',
    num_labels=3,
    multi_label=True,
    num_epochs=20
)
```

### 自定义参数
```python
train_model(
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

---

## 三、模型推理

### 单张图片预测
```python
from src.inference import QRCodePredictor

predictor = QRCodePredictor(
    checkpoint_path='checkpoints/best_model.pth',
    multi_label=True,
    threshold=0.5
)

result = predictor.predict_image('test.jpg')

# 查看结果
for label, pred in result['predictions'].items():
    print(f"{label}: {pred['value']} ({pred['probability']:.1%})")
```

### 批量预测
```python
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.predict_batch(image_paths)
```

---

## 四、结果格式

### 预测结果
```python
{
    'image_path': 'test.jpg',
    'predictions': {
        'is_copied': {
            'value': 1,           # 0 或 1
            'probability': 0.92   # 0.0 到 1.0
        },
        'is_blurry': {
            'value': 0,
            'probability': 0.16
        },
        'is_low_light': {
            'value': 1,
            'probability': 0.79
        }
    }
}
```

---

## 五、常用命令

### 监控训练
```bash
tensorboard --logdir=logs
```

### 查看训练历史
```python
import json
with open('checkpoints/training_history.json') as f:
    history = json.load(f)
print(f"Best accuracy: {max(history['val_acc']):.4f}")
```

### 导出 ONNX
```python
from src.utils import export_to_onnx

export_to_onnx(
    model_path='checkpoints/best_model.pth',
    output_path='models/model.onnx'
)
```

---

## 六、配置文件 (config.yaml)

```yaml
data:
  train_dir: "data/train"
  val_dir: "data/val"
  image_size: 224
  batch_size: 32

model:
  name: "resnet50"
  num_labels: 3
  label_names: ["is_copied", "is_blurry", "is_low_light"]

training:
  num_epochs: 20
  learning_rate: 0.001
```

---

## 七、目录结构

```
PaperprintCNN/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── annotations.json
│   └── val/
│       ├── images/
│       └── annotations.json
├── checkpoints/
│   ├── best_model.pth
│   └── training_history.json
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── inference.py
├── create_annotations.py
├── example_multilabel.py
└── config.yaml
```

---

## 八、故障排除

### 问题：找不到标注文件
```
ValueError: Annotation file not found
```
**解决**: 确保 `annotations.json` 在正确位置

### 问题：标签值错误
```
Error: label must be 0 or 1
```
**解决**: 检查 JSON 文件，确保所有值为 0 或 1

### 问题：维度不匹配
```
RuntimeError: size mismatch
```
**解决**: 检查 `num_labels` 是否与标注文件中的标签数量一致

---

## 九、性能优化

### 调整批次大小
```python
batch_size=64  # 增大批次（需要更多显存）
batch_size=16  # 减小批次（显存不足时）
```

### 调整学习率
```python
learning_rate=0.0001  # 降低学习率（模型不收敛时）
learning_rate=0.01    # 提高学习率（收敛太慢时）
```

### 调整预测阈值
```python
threshold=0.3  # 降低阈值（增加召回率）
threshold=0.7  # 提高阈值（增加精确率）
```

---

## 十、示例脚本

### 完整训练脚本
```python
from src.train import train_model

# 训练
history = train_model(
    train_dir='data/train',
    val_dir='data/val',
    model_name='resnet50',
    num_labels=3,
    multi_label=True,
    num_epochs=20
)

print(f"Training complete!")
print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
```

### 完整推理脚本
```python
from src.inference import QRCodePredictor
import os

# 初始化
predictor = QRCodePredictor(
    checkpoint_path='checkpoints/best_model.pth',
    multi_label=True
)

# 批量预测
image_dir = 'test_images'
results = []

for img_file in os.listdir(image_dir):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join(image_dir, img_file)
        result = predictor.predict_image(img_path)
        results.append(result)

# 保存结果
import csv
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'is_copied', 'is_blurry', 'is_low_light'])
    for r in results:
        writer.writerow([
            os.path.basename(r['image_path']),
            r['predictions']['is_copied']['value'],
            r['predictions']['is_blurry']['value'],
            r['predictions']['is_low_light']['value']
        ])

print(f"Processed {len(results)} images")
```

---

**更多详细信息，请查看 `MULTILABEL_GUIDE.md`**
