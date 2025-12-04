# QR Code Classification System

一个基于深度学习的二维码图片分类系统，用于区分**原始印刷的二维码**和**复印的二维码**。

## 项目简介

本项目使用迁移学习（Transfer Learning）技术，基于预训练的 ResNet-50 模型，训练一个自定义的二维码分类器。系统能够自动识别一张二维码图片是原始印刷的还是通过图片二次复印的。

### 主要特性

- ✅ 基于 PyTorch 和预训练模型（ResNet-50/18, EfficientNet, MobileNet）
- ✅ 完整的训练和推理流程
- ✅ 数据增强防止过拟合
- ✅ TensorBoard 可视化训练过程
- ✅ 自动保存最佳模型
- ✅ 支持 GPU 加速训练
- ✅ 简单易用的命令行接口

## 环境要求

- Python 3.14+
- PyTorch 2.0+
- CUDA（可选，用于 GPU 加速）

## 安装

本项目使用 `uv` 进行包管理，但也可以使用标准的 pip：

```bash
# 使用虚拟环境中的 pip 安装依赖
.venv\Scripts\python.exe -m pip install torch torchvision pillow numpy matplotlib tqdm tensorboard pyyaml
```

或者如果您有 `uv`：

```bash
uv sync
```

## 数据准备

### 目录结构

将您的数据组织成以下结构：

```
data/
├── train/
│   ├── original/      # 原始印刷的二维码图片
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── copied/        # 复印的二维码图片
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── val/
    ├── original/      # 验证集：原始印刷的二维码
    │   └── ...
    └── copied/        # 验证集：复印的二维码
        └── ...
```

### 数据建议

- **训练集**：每个类别至少 100 张图片（越多越好）
- **验证集**：每个类别至少 20-30 张图片
- **图片格式**：支持 JPG, PNG, BMP, GIF
- **图片质量**：建议分辨率至少 224x224 像素
- **数据平衡**：两个类别的图片数量尽量接近

## 使用方法

### 方法 1: 快速开始（推荐）

使用预配置的训练脚本：

```bash
# 确保数据已按上述结构组织在 data/ 目录下
.venv\Scripts\python.exe train_model.py
```

### 方法 2: 使用主脚本

```bash
# 训练模型
.venv\Scripts\python.exe main.py train --train-dir data/train --val-dir data/val --epochs 20

# 预测单张图片
.venv\Scripts\python.exe main.py predict --image test.jpg --checkpoint checkpoints/best_model.pth
```

### 方法 3: 使用独立脚本

```bash
# 预测
.venv\Scripts\python.exe predict.py test_image.jpg
```

### 高级配置

编辑 `config.yaml` 文件来自定义训练参数：

```yaml
data:
  train_dir: "data/train"
  val_dir: "data/val"
  batch_size: 32

model:
  name: "resnet50"  # 可选: resnet18, efficientnet_b0, mobilenet_v3_small
  
training:
  num_epochs: 20
  learning_rate: 0.001
```

然后运行：

```bash
.venv\Scripts\python.exe main.py train --config config.yaml
```

## 训练监控

使用 TensorBoard 查看训练过程：

```bash
.venv\Scripts\python.exe -m tensorboard --logdir=logs
```

然后在浏览器中打开 `http://localhost:6006`

## 模型选择

系统支持多种预训练模型：

| 模型 | 参数量 | 速度 | 精度 | 推荐场景 |
|------|--------|------|------|----------|
| ResNet-50 | 25M | 中等 | 高 | 默认选择，平衡性能 |
| ResNet-18 | 11M | 快 | 中等 | 快速训练 |
| EfficientNet-B0 | 5M | 快 | 高 | 高效训练 |
| MobileNet-V3 | 2M | 很快 | 中等 | 移动端部署 |

## 项目结构

```
ImageNet/
├── src/
│   ├── dataset.py      # 数据加载和预处理
│   ├── model.py        # 模型定义
│   ├── train.py        # 训练逻辑
│   ├── inference.py    # 推理逻辑
│   └── utils.py        # 工具函数
├── data/               # 数据目录（需自行创建）
├── checkpoints/        # 模型检查点（训练时自动创建）
├── logs/               # TensorBoard 日志（训练时自动创建）
├── main.py             # 主入口脚本
├── train_model.py      # 快速训练脚本
├── predict.py          # 快速预测脚本
├── config.yaml         # 配置文件
├── pyproject.toml      # 项目依赖
└── README.md           # 本文件
```

## 训练输出

训练完成后，您将获得：

- `checkpoints/best_model.pth` - 最佳模型（验证集上表现最好）
- `checkpoints/checkpoint_epoch_N.pth` - 每个 epoch 的检查点
- `checkpoints/training_history.json` - 训练历史记录
- `checkpoints/class_names.json` - 类别名称映射
- `logs/` - TensorBoard 日志文件

## 预期性能

根据数据集大小，预期性能如下：

- **100+ 张图片/类**：准确率 85-95%
- **500+ 张图片/类**：准确率 90-98%
- **1000+ 张图片/类**：准确率 95%+

实际性能取决于：
- 图片质量和多样性
- 原始和复印图片的差异程度
- 训练参数设置

## 服务器部署

### 快速集成到现有服务器

如果您已经有一个 Python 服务器环境，只需三步即可集成模型：

#### 1️⃣ 准备文件

**方式 A: 使用打包工具（推荐）**

运行打包脚本生成部署包：
```bash
python create_deployment_package.py
```

这会生成一个包含所有必需文件的 ZIP 包（约 95MB），可以直接发送给服务器同事。

**方式 B: 手动复制**

将以下文件复制到服务器：
```
checkpoints/best_model.pth
checkpoints/class_names.json
predictor_singleton.py
src/
```


#### 2️⃣ 安装依赖

```bash
pip install torch torchvision pillow numpy
```

#### 3️⃣ 集成代码

```python
from predictor_singleton import predictor

# 服务器启动时初始化（只需一次）
predictor.initialize(checkpoint_path='checkpoints/best_model.pth')

# 在接口中使用
def your_api_endpoint(uploaded_file):
    image_bytes = uploaded_file.read()
    result = predictor.predict_from_bytes(image_bytes)
    
    return {
        'label': result['predicted_label'],      # 'copied' 或 'original'
        'confidence': result['confidence'],       # 0.0 - 1.0
    }
```

### 完整示例

项目提供了完整的 API 示例：

- **Flask 示例**: `examples/flask_api.py`
- **FastAPI 示例**: `examples/fastapi_api.py`
- **客户端示例**: `examples/api_client.py`

### 性能参考

| 环境 | 单张预测耗时 | 吞吐量 |
|-----|------------|--------|
| CPU | ~140ms | 7 张/秒 |
| GPU | ~20ms | 50 张/秒 |

### 详细文档

- 📖 [快速开始指南](QUICKSTART.md) - 最简单的集成步骤
- 📖 [完整集成指南](INTEGRATION_GUIDE.md) - 详细的部署文档
- 📦 [打包工具说明](PACKAGING_GUIDE.md) - 部署包打包指南
- 🧪 [API 测试](test_api.py) - 测试 API 接口

## 常见问题

### 1. 训练速度慢？

- 如果有 NVIDIA GPU，确保安装了 CUDA 版本的 PyTorch
- 减小 batch_size（如果内存不足）
- 使用更小的模型（如 ResNet-18）

### 2. 准确率不高？

- 增加训练数据量
- 增加训练轮数（epochs）
- 尝试不同的学习率
- 检查数据质量和标注是否正确

### 3. 内存不足？

- 减小 batch_size
- 使用更小的模型
- 减小 image_size（如 128 或 160）

## 技术细节

### 迁移学习策略

1. **特征提取模式**（默认）：
   - 冻结预训练模型的主干网络
   - 只训练最后的分类层
   - 适合小数据集，训练快

2. **微调模式**（可选）：
   - 解冻部分或全部主干网络
   - 用较小的学习率训练整个网络
   - 适合大数据集，精度更高

### 数据增强

训练时自动应用以下增强：
- 随机水平翻转
- 随机旋转（±15°）
- 颜色抖动（亮度、对比度、饱和度）
- 随机仿射变换

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请创建 Issue。
