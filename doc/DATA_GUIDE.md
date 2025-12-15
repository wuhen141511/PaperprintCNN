# 数据组织指南

本文档详细说明如何准备和组织训练数据。

## 目录结构要求

您的数据必须按照以下结构组织：

```
data/
├── train/                    # 训练数据集
│   ├── original/            # 类别1：原始印刷的二维码
│   │   ├── qr_001.jpg
│   │   ├── qr_002.jpg
│   │   ├── qr_003.png
│   │   └── ...
│   └── copied/              # 类别2：复印的二维码
│       ├── qr_001.jpg
│       ├── qr_002.jpg
│       └── ...
└── val/                      # 验证数据集
    ├── original/            # 验证集：原始印刷的二维码
    │   ├── qr_val_001.jpg
    │   └── ...
    └── copied/              # 验证集:复印的二维码
        ├── qr_val_001.jpg
        └── ...
```

## 重要说明

### 1. 类别名称

- 类别文件夹的名称可以自定义（不一定是 `original` 和 `copied`）
- 但训练集和验证集中的类别名称必须一致
- 系统会自动按字母顺序排序类别

### 2. 支持的图片格式

- JPG / JPEG
- PNG
- BMP
- GIF

### 3. 数据量建议

| 数据集大小 | 每类图片数量 | 预期准确率 | 训练时间（GPU） |
|-----------|-------------|-----------|----------------|
| 小型 | 50-100 | 80-90% | 5-10分钟 |
| 中型 | 100-500 | 85-95% | 10-20分钟 |
| 大型 | 500-1000 | 90-98% | 20-40分钟 |
| 超大型 | 1000+ | 95%+ | 40分钟+ |

### 4. 训练/验证集划分

推荐比例：
- **训练集**：80%
- **验证集**：20%

最小要求：
- 训练集：每类至少 50 张
- 验证集：每类至少 10 张

### 5. 数据质量要求

#### 图片分辨率
- **最低**：224 x 224 像素
- **推荐**：512 x 512 像素或更高
- 系统会自动调整大小，但原始图片质量越高越好

#### 图片内容
- 确保二维码在图片中清晰可见
- 可以包含背景，不需要裁剪到只有二维码
- 避免过度模糊或损坏的图片

#### 数据多样性
为了获得更好的泛化能力，建议包含：
- 不同光照条件下的图片
- 不同角度拍摄的图片
- 不同背景的图片
- 不同尺寸的二维码

### 6. 数据平衡

- 尽量保持两个类别的图片数量接近
- 如果数据不平衡（如 original:copied = 3:1），考虑：
  - 收集更多少数类的数据
  - 使用数据增强
  - 调整类别权重（需要修改代码）

## 数据收集建议

### 原始印刷二维码
- 从打印机直接打印的二维码
- 印刷品上的二维码
- 屏幕显示的二维码（如果适用）

### 复印二维码
- 通过复印机复印的二维码
- 拍照后打印的二维码
- 扫描后打印的二维码

### 拍摄技巧
1. 使用稳定的光源
2. 保持相机稳定，避免模糊
3. 确保二维码占据图片的合理比例
4. 拍摄多个角度和距离

## 创建数据目录

### Windows PowerShell

```powershell
# 创建目录结构
New-Item -ItemType Directory -Path "data\train\original" -Force
New-Item -ItemType Directory -Path "data\train\copied" -Force
New-Item -ItemType Directory -Path "data\val\original" -Force
New-Item -ItemType Directory -Path "data\val\copied" -Force
```

### Linux / macOS

```bash
# 创建目录结构
mkdir -p data/train/original
mkdir -p data/train/copied
mkdir -p data/val/original
mkdir -p data/val/copied
```

## 数据验证

创建数据后，可以使用以下 Python 脚本验证数据结构：

```python
import os

def validate_data_structure(data_dir):
    """验证数据目录结构"""
    required_dirs = [
        'train/original',
        'train/copied',
        'val/original',
        'val/copied'
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"❌ 缺少目录: {full_path}")
        else:
            num_images = len([f for f in os.listdir(full_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
            print(f"✓ {dir_path}: {num_images} 张图片")

# 运行验证
validate_data_structure('data')
```

## 常见问题

### Q: 我的数据很少，只有每类 30 张图片，能训练吗？

A: 可以尝试，但效果可能不理想。建议：
- 使用数据增强（系统已自动启用）
- 使用更小的模型（如 ResNet-18）
- 减少训练轮数，避免过拟合
- 尽量收集更多数据

### Q: 训练集和验证集的图片可以重复吗？

A: **不可以**！验证集必须是模型从未见过的数据，用于评估模型的泛化能力。

### Q: 我可以有超过 2 个类别吗？

A: 可以！只需：
1. 在 `train/` 和 `val/` 下创建更多类别文件夹
2. 训练时设置 `--num-classes` 参数为实际类别数

### Q: 图片文件名有要求吗？

A: 没有特殊要求，只要是有效的文件名即可。系统只关心文件所在的文件夹（类别）。

## 数据增强

训练时，系统会自动对训练数据应用以下增强：

- ✅ 随机水平翻转
- ✅ 随机旋转（±15度）
- ✅ 颜色抖动（亮度、对比度、饱和度）
- ✅ 随机仿射变换（平移、缩放）

验证数据**不会**应用增强，只会调整大小和归一化。

## 下一步

数据准备好后，参考 [README.md](README.md) 开始训练模型！
