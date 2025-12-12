# ONNX Export Feature - Summary

## 已完成的功能 (Completed Features)

### 1. 新增工具函数 (New Utility Function)
在 `src/utils.py` 中添加了 `export_to_onnx()` 函数，用于将 PyTorch 模型转换为 ONNX 格式。

**主要特性：**
- ✅ 自动加载 PyTorch 模型检查点 (.pth)
- ✅ 导出为 ONNX 格式，兼容 C++ 和 OpenCV
- ✅ 支持动态批次大小 (dynamic batch size)
- ✅ 自动验证导出的 ONNX 模型
- ✅ 默认使用 `checkpoints/best_model.pth`
- ✅ 自动保存到模型同目录下，扩展名为 `.onnx`
- ✅ 提供详细的导出信息和 C++ 使用示例

**函数签名：**
```python
def export_to_onnx(
    model_path: Optional[str] = None,           # 默认: checkpoints/best_model.pth
    output_path: Optional[str] = None,          # 默认: 同目录，.onnx 扩展名
    input_size: Tuple[int, int, int] = (3, 350, 350),
    batch_size: int = 1,
    opset_version: int = 11,                    # OpenCV 兼容版本
    dynamic_axes: bool = True,
    verbose: bool = True
) -> str
```

### 2. 快捷调用脚本 (Convenient Script)
创建了 `export_to_onnx.py` 脚本，提供命令行接口。

**基本用法：**
```bash
# 导出默认模型
python export_to_onnx.py

# 导出指定模型
python export_to_onnx.py --model checkpoints/epoch_10.pth

# 自定义输出路径
python export_to_onnx.py --output models/qrcode.onnx

# 自定义输入尺寸
python export_to_onnx.py --input-size 224

# 指定 ONNX opset 版本
python export_to_onnx.py --opset 13
```

**命令行参数：**
- `--model, -m`: 指定输入模型路径
- `--output, -o`: 指定输出 ONNX 路径
- `--input-size, -s`: 输入图像尺寸 (默认: 350)
- `--batch-size, -b`: 批次大小 (默认: 1)
- `--opset`: ONNX opset 版本 (默认: 11)
- `--no-dynamic`: 禁用动态批次大小
- `--quiet, -q`: 静默模式

### 3. 使用文档 (Documentation)
创建了 `ONNX_EXPORT_GUIDE.md` 完整使用指南，包含：
- Python 使用示例
- C++ + OpenCV 使用示例
- 编译说明
- 常见问题解决

## 使用示例 (Usage Examples)

### Python 代码调用
```python
from src.utils import export_to_onnx

# 方式1: 使用默认设置
onnx_path = export_to_onnx()

# 方式2: 自定义设置
onnx_path = export_to_onnx(
    model_path='checkpoints/best_model.pth',
    output_path='models/qrcode_classifier.onnx',
    input_size=(3, 224, 224)
)
```

### 命令行调用
```bash
# 最简单的方式
python export_to_onnx.py

# 导出特定模型
python export_to_onnx.py --model checkpoints/epoch_15.pth
```

### C++ + OpenCV 使用
```cpp
#include <opencv2/dnn.hpp>

// 加载 ONNX 模型
cv::dnn::Net net = cv::dnn::readNetFromONNX("checkpoints/best_model.onnx");

// 预处理图像
cv::Mat blob = cv::dnn::blobFromImage(
    image, 1.0/255.0, 
    cv::Size(350, 350),
    cv::Scalar(0.485, 0.456, 0.406),
    true, false
);

// 推理
net.setInput(blob);
cv::Mat output = net.forward();
```

## 文件清单 (File List)

1. **src/utils.py** - 添加了 `export_to_onnx()` 函数
2. **export_to_onnx.py** - 命令行脚本 (新建)
3. **ONNX_EXPORT_GUIDE.md** - 使用指南 (新建)
4. **ONNX_SUMMARY.md** - 本文件 (新建)

## 技术细节 (Technical Details)

### ONNX 导出配置
- **Opset Version**: 11 (兼容大多数 OpenCV 版本)
- **Dynamic Axes**: 支持动态批次大小
- **Input Names**: "input"
- **Output Names**: "output"
- **Constant Folding**: 已启用 (优化性能)

### 模型信息自动推断
函数会自动从检查点中提取：
- 模型架构 (model_name)
- 类别数量 (num_classes)
- 训练轮次 (epoch)
- 准确率 (accuracy)

### 兼容性
- ✅ PyTorch 模型 (.pth)
- ✅ OpenCV DNN 模块
- ✅ C++ 部署
- ✅ 支持所有项目中的模型架构:
  - ResNet50
  - ResNet18
  - EfficientNet-B0
  - MobileNet-V3-Small

## 下一步 (Next Steps)

1. 训练模型:
   ```bash
   python main.py
   ```

2. 导出 ONNX:
   ```bash
   python export_to_onnx.py
   ```

3. 在 C++ 中使用导出的模型 (参考 ONNX_EXPORT_GUIDE.md)

## 注意事项 (Important Notes)

1. **归一化**: 模型期望 ImageNet 归一化的输入
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

2. **输入尺寸**: 必须与训练时一致 (默认: 350x350)

3. **颜色通道**: 模型期望 RGB，OpenCV 加载的是 BGR，需要转换

4. **输出格式**: 模型输出 logits，需要 softmax 转换为概率
