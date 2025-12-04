# 快速开始 - 服务器集成

本文档提供最简单的步骤，帮助您快速将模型集成到现有服务器。

## 🚀 三步快速集成

### 步骤 1: 准备文件

将以下文件复制到您的服务器：

```bash
# 必需文件
checkpoints/best_model.pth          # 模型文件 (~102MB)
checkpoints/class_names.json        # 类别配置
predictor_singleton.py              # 预测器
src/                                # 源代码目录
├── inference.py
├── model.py
├── dataset.py
└── utils.py
```

### 步骤 2: 安装依赖

```bash
pip install torch torchvision pillow numpy
```

### 步骤 3: 集成到您的代码

#### 方式 A: 最简单的集成（推荐）

```python
from predictor_singleton import predictor

# 1. 服务器启动时初始化（只需一次）
predictor.initialize(checkpoint_path='checkpoints/best_model.pth')

# 2. 在接口中使用
def your_api_endpoint(uploaded_file):
    # 读取上传的文件字节
    image_bytes = uploaded_file.read()
    
    # 预测
    result = predictor.predict_from_bytes(image_bytes)
    
    # 返回结果
    return {
        'label': result['predicted_label'],      # 'copied' 或 'original'
        'confidence': result['confidence'],       # 0.0 - 1.0
        'probabilities': result['class_probabilities']
    }
```

#### 方式 B: 从文件路径预测

```python
# 如果图片已保存到服务器
result = predictor.predict('/path/to/image.png')
```

#### 方式 C: 批量预测

```python
# 批量处理多张图片
image_bytes_list = [file1.read(), file2.read(), file3.read()]
results = predictor.predict_batch_from_bytes(image_bytes_list)
```

---

## 📝 完整示例

### Flask 集成示例

```python
from flask import Flask, request, jsonify
from predictor_singleton import predictor

app = Flask(__name__)

# 启动时初始化
@app.before_first_request
def init():
    predictor.initialize(checkpoint_path='checkpoints/best_model.pth')

# API 接口
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image_bytes = file.read()
    result = predictor.predict_from_bytes(image_bytes)
    
    return jsonify({
        'success': True,
        'label': result['predicted_label'],
        'confidence': result['confidence']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**运行:**
```bash
python your_app.py
```

**测试:**
```bash
curl -X POST -F "file=@test.png" http://localhost:5000/predict
```

---

### FastAPI 集成示例

```python
from fastapi import FastAPI, File, UploadFile
from predictor_singleton import predictor

app = FastAPI()

@app.on_event("startup")
async def startup():
    predictor.initialize(checkpoint_path='checkpoints/best_model.pth')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predictor.predict_from_bytes(image_bytes)
    
    return {
        'success': True,
        'label': result['predicted_label'],
        'confidence': result['confidence']
    }
```

**运行:**
```bash
uvicorn your_app:app --host 0.0.0.0 --port 8000
```

**访问文档:**
```
http://localhost:8000/docs
```

---

### Django 集成示例

```python
# views.py
from django.http import JsonResponse
from predictor_singleton import predictor

# 在 apps.py 中初始化
# class YourAppConfig(AppConfig):
#     def ready(self):
#         from predictor_singleton import predictor
#         predictor.initialize(checkpoint_path='checkpoints/best_model.pth')

def predict_view(request):
    file = request.FILES['file']
    image_bytes = file.read()
    result = predictor.predict_from_bytes(image_bytes)
    
    return JsonResponse({
        'success': True,
        'label': result['predicted_label'],
        'confidence': result['confidence']
    })
```

---

## 🎯 返回值说明

```python
result = {
    'predicted_class': 0,                    # 类别索引 (0 或 1)
    'predicted_label': 'copied',             # 类别名称 ('copied' 或 'original')
    'confidence': 0.6288,                    # 置信度 (0.0 - 1.0)
    'class_probabilities': {                 # 所有类别的概率
        'copied': 0.6288,
        'original': 0.3712
    }
}
```

---

## ⚡ 性能参考

| 环境 | 单张预测耗时 | 吞吐量 |
|-----|------------|--------|
| CPU | ~140ms | 7 张/秒 |
| GPU | ~20ms | 50 张/秒 |

**首次加载**: 模型加载约 0.5 秒（仅启动时一次）

---

## 🔧 常见问题

### Q: 如何检查模型是否已加载？

```python
if predictor.is_initialized():
    print("模型已就绪")
else:
    print("模型未初始化")
```

### Q: 如何获取模型信息？

```python
info = predictor.get_model_info()
print(info)
# {
#     'initialized': True,
#     'model_name': 'resnet50',
#     'num_classes': 2,
#     'class_names': ['copied', 'original'],
#     'device': 'cpu',
#     'image_size': 224
# }
```

### Q: 支持哪些图片格式？

支持: PNG, JPG, JPEG, BMP, GIF

### Q: 如何使用 GPU？

模型会自动检测并使用 GPU（如果可用）。无需额外配置。

### Q: 预测失败怎么办？

```python
try:
    result = predictor.predict_from_bytes(image_bytes)
except Exception as e:
    print(f"预测失败: {e}")
    # 处理错误
```

---

## 📚 更多资源

- **完整集成指南**: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **Flask 完整示例**: [examples/flask_api.py](examples/flask_api.py)
- **FastAPI 完整示例**: [examples/fastapi_api.py](examples/fastapi_api.py)
- **客户端调用示例**: [examples/api_client.py](examples/api_client.py)

---

## 💡 最佳实践

1. ✅ **在服务器启动时初始化模型一次**，不要每次请求都加载
2. ✅ **使用 `predict_from_bytes`** 处理上传的文件
3. ✅ **添加文件大小和格式验证**，防止恶意请求
4. ✅ **使用批量预测** 提高多图片处理效率
5. ✅ **添加异常处理**，确保服务稳定性

---

## 🎉 就这么简单！

现在您已经可以在服务器上使用 QR 码分类模型了。如有问题，请参考完整文档。
