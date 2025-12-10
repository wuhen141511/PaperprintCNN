# 数据集分割工具使用指南

## 功能说明

这个工具用于将训练目录 `/data/train/` 下的图片按指定比例剪切转移到验证目录 `/data/val/`，同时保持类别结构不变。

## 目录结构要求

训练数据应该按照以下结构组织:

```
/data/train/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

分割后的结构:

```
/data/train/          # 保留的训练数据
├── class1/
│   ├── image1.jpg
│   └── ...
└── class2/
    └── ...

/data/val/            # 移动到验证集的数据
├── class1/
│   ├── image3.jpg
│   └── ...
└── class2/
    └── ...
```

## 使用方法

### 方法1: 使用命令行脚本 (推荐)

```bash
# 基本用法 - 使用默认参数 (20% 验证集)
python split_dataset.py

# 自定义参数
python split_dataset.py --train_dir data/train --val_dir data/val --ratio 0.2

# 指定不同的验证集比例 (例如 30%)
python split_dataset.py --ratio 0.3

# 使用自定义随机种子
python split_dataset.py --ratio 0.2 --seed 123

# 静默模式 (减少输出)
python split_dataset.py --ratio 0.2 --quiet
```

### 方法2: 在Python代码中使用

```python
from src.utils import split_train_val_data

# 基本用法
stats = split_train_val_data(
    train_dir='data/train',
    val_dir='data/val',
    val_ratio=0.2,
    seed=42,
    verbose=True
)

# 查看统计信息
print(f"总共移动了 {stats['total_moved']} 张图片")
print(f"训练集剩余 {stats['total_remaining']} 张图片")
print(f"实际验证集比例: {stats['val_ratio_actual']:.1%}")

# 查看每个类别的详细信息
for class_name, class_stats in stats['classes'].items():
    print(f"{class_name}: {class_stats['moved_to_val']}/{class_stats['total']} 移动到验证集")
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_dir` | str | `'data/train'` | 训练数据目录路径 |
| `val_dir` | str | `'data/val'` | 验证数据目录路径 (不存在会自动创建) |
| `ratio` | float | `0.2` | 验证集比例 (0.0 到 1.0 之间) |
| `seed` | int | `42` | 随机种子，用于可重现性 |
| `verbose` | bool | `True` | 是否显示详细进度信息 |

## 功能特性

✅ **保持类别结构**: 自动在验证目录中创建相同的类别子目录

✅ **随机采样**: 使用随机种子确保可重现的数据分割

✅ **支持多种图片格式**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.webp`

✅ **文件名冲突处理**: 自动处理重复文件名

✅ **详细统计信息**: 提供每个类别和总体的分割统计

✅ **安全确认**: 命令行模式会要求用户确认操作

✅ **最小验证集保证**: 每个类别至少移动1张图片到验证集

## 输出示例

```
============================================================
Dataset Splitting Tool
============================================================
Training directory: E:\workspace_p\ImageNet\data\train
Validation directory: E:\workspace_p\ImageNet\data\val
Validation ratio: 20.0%
Random seed: 42
============================================================

⚠️  This will MOVE files from train to val directory. Continue? (y/n): y

Found 2 classes in E:\workspace_p\ImageNet\data\train
Target validation ratio: 20.0%
------------------------------------------------------------
✓ Class 'original': 20/100 images moved (20.0%) | 80 remaining
✓ Class 'photocopy': 20/100 images moved (20.0%) | 80 remaining
------------------------------------------------------------
✅ Split completed successfully!
   Total images processed: 200
   Moved to validation: 40 (20.0%)
   Remaining in training: 160
   Validation directory: E:\workspace_p\ImageNet\data\val
```

## 注意事项

⚠️ **重要**: 此操作会**移动**文件，不是复制。请确保在操作前备份数据或确认不需要保留原始完整训练集。

⚠️ 如果验证目录已存在同名文件，工具会自动重命名以避免覆盖。

⚠️ 建议在首次使用时先用小数据集测试，确认符合预期后再处理完整数据集。

## 常见问题

### Q: 如何恢复原始数据结构?

A: 可以手动将 `/data/val/` 下的文件移回 `/data/train/` 对应的类别目录。

### Q: 可以多次运行这个脚本吗?

A: 可以，但每次运行都会从当前的 `train_dir` 中移动文件。如果需要重新分割，建议先将验证集的文件移回训练集。

### Q: 支持嵌套的类别目录吗?

A: 目前只支持一层类别目录结构 (如 `/data/train/class_name/images`)。

### Q: 如何确保训练集和验证集没有重复?

A: 工具使用 `shutil.move()` 移动文件，确保文件只存在于一个位置。使用相同的随机种子可以获得相同的分割结果。
