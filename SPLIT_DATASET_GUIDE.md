# 数据集分割工具使用指南

## 功能说明

这个工具支持两种模式的数据集分割：

1. **多标签模式** (Multi-Label): 适用于带有 JSON 标注文件的多标签分类
2. **单分类模式** (Single-Class): 适用于基于目录结构的单分类任务

工具会自动检测数据集类型，或者可以手动指定模式。

---

## 支持的数据结构

### 多标签模式 (推荐用于当前项目)

```
data/train/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── annotations.json
```

分割后：

```
data/train/
├── images/              # 剩余训练图片
│   ├── img001.jpg
│   └── ...
└── annotations.json     # 训练集标注

data/val/
├── images/              # 验证图片
│   ├── img002.jpg
│   └── ...
└── annotations.json     # 验证集标注
```

### 单分类模式 (兼容旧版)

```
data/train/
├── class1/
│   ├── image1.jpg
│   └── ...
└── class2/
    ├── image1.jpg
    └── ...
```

---

## 使用方法

### 方法1: 自动检测模式 (推荐)

```bash
# 基本用法 - 自动检测数据类型，使用默认参数 (20% 验证集)
python split_dataset.py

# 自定义验证集比例
python split_dataset.py --ratio 0.3

# 指定目录
python split_dataset.py --train_dir data/train --val_dir data/val --ratio 0.2
```

### 方法2: 手动指定模式

```bash
# 多标签模式
python split_dataset.py --mode multi-label --ratio 0.2

# 单分类模式
python split_dataset.py --mode single-class --ratio 0.2
```

### 方法3: 在Python代码中使用

#### 多标签模式

```python
from split_dataset import split_multilabel_data

stats = split_multilabel_data(
    train_dir='data/train',
    val_dir='data/val',
    val_ratio=0.2,
    seed=42,
    verbose=True
)

print(f"训练集: {stats['train_images']} 张")
print(f"验证集: {stats['val_images']} 张")
print(f"实际比例: {stats['val_ratio_actual']:.1%}")
```

#### 单分类模式

```python
from split_dataset import split_singleclass_data

stats = split_singleclass_data(
    train_dir='data/train',
    val_dir='data/val',
    val_ratio=0.2,
    seed=42,
    verbose=True
)

for class_name, class_stats in stats['classes'].items():
    print(f"{class_name}: {class_stats['moved_to_val']}/{class_stats['total']}")
```

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_dir` | str | `'data/train'` | 训练数据目录路径 |
| `--val_dir` | str | `'data/val'` | 验证数据目录路径 |
| `--ratio` | float | `0.2` | 验证集比例 (0.0 到 1.0) |
| `--seed` | int | `42` | 随机种子 |
| `--mode` | str | `'auto'` | 模式: `auto`, `multi-label`, `single-class` |
| `--quiet` | flag | `False` | 静默模式 |

---

## 功能特性

### 多标签模式特性 (v2.1 升级)

✅ **分层采样 (Stratified Sampling)**: 采用标签幂集策略 (Label Powerset)，将样本按"标签组合"分组后进行比例采样，确保验证集分布与训练集高度一致。

✅ **自动分割标注文件**: 同步分割 `annotations.json`

✅ **保护稀有样本**: 即使是极少见的标签组合，也会尽可能均分到验证集，避免随机划分造成的样本缺失。

✅ **标签分布统计**: 显示验证集中各标签的分布情况

✅ **文件移动**: 图片从 `train/images/` 移动到 `val/images/`

### 单分类模式特性

✅ **保持类别结构**: 自动创建相同的类别子目录

✅ **随机采样**: 使用随机种子确保可重现

✅ **支持多种图片格式**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.webp`

✅ **文件名冲突处理**: 自动处理重复文件名

### 通用特性

✅ **自动模式检测**: 智能识别数据集类型

✅ **详细统计信息**: 提供完整的分割统计

✅ **安全确认**: 操作前要求用户确认

---

## 输出示例

### 多标签模式输出

```
============================================================
Dataset Splitting Tool
============================================================
Mode: multi-label
Training directory: E:\workspace_paper\PaperprintCNN\data\train
Validation directory: E:\workspace_paper\PaperprintCNN\data\val
Validation ratio: 20.0%
Random seed: 42
============================================================

⚠️  This will MOVE files from train to val directory. Continue? (y/n): y

Found 500 images in annotations
Target validation ratio: 20.0%
Will move 100 images to validation (20.0%)
------------------------------------------------------------
✓ Moved 100 images to validation
✓ Training set: 400 images
✓ Validation set: 100 images
✓ Saved annotations to both directories

Label distribution in validation set:
  - is_copied: 45/100 (45.0%)
  - is_blurry: 23/100 (23.0%)
  - is_low_light: 38/100 (38.0%)

============================================================
Split Summary
============================================================
Total images: 500
Training set: 400
Validation set: 100
Actual validation ratio: 20.0%
============================================================
✅ Dataset split completed successfully!
============================================================
```

### 单分类模式输出

```
============================================================
Dataset Splitting Tool
============================================================
Mode: single-class
Training directory: E:\workspace_paper\PaperprintCNN\data\train
Validation directory: E:\workspace_paper\PaperprintCNN\data\val
Validation ratio: 20.0%
Random seed: 42
============================================================

⚠️  This will MOVE files from train to val directory. Continue? (y/n): y

Found 2 classes in data/train
Target validation ratio: 20.0%
------------------------------------------------------------
✓ Class 'original': 20/100 images moved (20.0%) | 80 remaining
✓ Class 'copied': 20/100 images moved (20.0%) | 80 remaining
------------------------------------------------------------
✅ Split completed successfully!
   Total images processed: 200
   Moved to validation: 40 (20.0%)
   Remaining in training: 160
```

---

## 完整工作流程示例

### 多标签分类项目

```bash
# 1. 准备数据
mkdir -p data/train/images
# 将所有图片放入 data/train/images/

# 2. 创建标注文件
python create_annotations.py create data/train/images

# 3. 编辑 data/train/annotations.json，设置标签值

# 4. 验证标注
python create_annotations.py validate data/train/annotations.json

# 5. 分割数据集
python split_dataset.py --ratio 0.2

# 6. 开始训练
python train_model.py
```

---

## 注意事项

### ⚠️ 重要提醒

1. **文件移动**: 此操作会**移动**文件，不是复制。请确保在操作前备份数据。

2. **标注同步**: 多标签模式会自动更新两个目录的 `annotations.json` 文件。

3. **不可逆操作**: 分割后如需恢复，需要手动将文件移回并合并标注文件。

4. **测试建议**: 建议先用小数据集测试，确认符合预期后再处理完整数据集。

### 💡 最佳实践

1. **备份数据**: 在分割前备份原始数据和标注文件

2. **验证标注**: 分割前使用 `create_annotations.py validate` 验证标注文件

3. **检查结果**: 分割后检查两个目录的图片数量和标注文件

4. **固定种子**: 使用相同的随机种子可以获得相同的分割结果

---

## 常见问题

### Q1: 如何恢复原始数据结构?

**多标签模式**:
```bash
# 手动将 val/images/ 下的文件移回 train/images/
# 然后合并两个 annotations.json 文件
python create_annotations.py merge data/train/annotations.json data/val/annotations.json -o data/train/annotations.json
```

**单分类模式**:
```bash
# 手动将 val/ 下的文件移回 train/ 对应的类别目录
```

### Q2: 可以多次运行这个脚本吗?

可以，但每次运行都会从当前的 `train_dir` 中移动文件。如果需要重新分割：

1. 先将验证集的文件移回训练集
2. 合并标注文件（多标签模式）
3. 重新运行分割脚本

### Q3: 如何确保训练集和验证集没有重复?

工具使用 `shutil.move()` 移动文件，确保文件只存在于一个位置。使用相同的随机种子可以获得相同的分割结果。

### Q4: 自动检测模式的逻辑是什么?

检测顺序：
1. 检查是否存在 `images/` 目录和 `annotations.json` → 多标签模式
2. 检查是否存在包含图片的子目录 → 单分类模式
3. 默认使用多标签模式

### Q5: 分割后标签分布不均衡怎么办?

由于是随机分割，可能出现某些标签在验证集中分布不均。解决方法：

1. 使用不同的随机种子重新分割
2. 增加数据量
3. 手动调整验证集（高级用户）

### Q6: 支持嵌套的目录结构吗?

- **多标签模式**: 所有图片必须在 `images/` 目录下（可以是一级）
- **单分类模式**: 只支持一层类别目录结构

---

## 高级用法

### 自定义分割比例

```bash
# 10% 验证集（更多训练数据）
python split_dataset.py --ratio 0.1

# 30% 验证集（更多验证数据）
python split_dataset.py --ratio 0.3
```

### 使用不同的随机种子

```bash
# 获得不同的分割结果
python split_dataset.py --seed 123
python split_dataset.py --seed 456
```

### 静默模式（用于脚本）

```bash
# 减少输出，适合在脚本中使用
python split_dataset.py --ratio 0.2 --quiet
```

---

## 相关工具

- `create_annotations.py` - 创建和管理标注文件
- `train_model.py` - 训练模型
- `example_multilabel.py` - 完整示例

---

## 更新日志

### v2.0 (当前版本)
- ✅ 新增多标签模式支持
- ✅ 自动检测数据集类型
- ✅ JSON 标注文件自动分割
- ✅ 标签分布统计
- ✅ 向后兼容单分类模式

### v1.0
- 基础的单分类目录分割功能

---

**需要帮助?** 查看 `MULTILABEL_GUIDE.md` 或运行 `python example_multilabel.py`
