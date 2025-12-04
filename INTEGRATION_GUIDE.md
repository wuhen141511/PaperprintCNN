# 模型集成指南

本指南说明如何将训练好的 QR 码分类模型集成到现有的服务器环境中。

## 📋 目录

1. [环境准备](#环境准备)
2. [模型文件准备](#模型文件准备)
3. [核心代码集成](#核心代码集成)
4. [API 接口示例](#api-接口示例)
5. [性能优化建议](#性能优化建议)

---

## 1. 环境准备

### 1.1 Python 依赖

在服务器上安装以下依赖：

```bash
pip install torch torchvision pillow numpy
```

或使用 requirements.txt：

```txt
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
numpy>=1.24.0
```

### 1.2 检查 GPU 支持（可选）

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

---

## 2. 模型文件准备

### 2.1 需要上传到服务器的文件

```
your_server/
├── models/
│   ├── best_model.pth          # 模型权重文件 (~102MB)
│   └── class_names.json        # 类别名称
└── src/
    ├── inference.py            # 推理代码
    ├── model.py                # 模型定义
    ├── dataset.py              # 数据预处理
    └── utils.py                # 工具函数
```

### 2.2 class_names.json 内容

```json
["copied", "original"]
```

---

## 3. 核心代码集成

### 3.1 创建预测器单例（推荐）

在服务器启动时加载模型一次，避免重复加载：

```python
# predictor_singleton.py
from src.inference import QRCodePredictor

class ModelPredictor:
    _instance = None
    _predictor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, checkpoint_path='models/best_model.pth'):
        """初始化模型（服务器启动时调用一次）"""
        if self._predictor is None:
            print("Loading QR code classification model...")
            self._predictor = QRCodePredictor(
                checkpoint_path=checkpoint_path,
                model_name='resnet50'
            )
            print("Model loaded successfully!")
    
    def predict(self, image_path):
        """预测单张图片"""
        if self._predictor is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        return self._predictor.predict_image(image_path, return_probs=True)
    
    def predict_from_bytes(self, image_bytes):
        """从字节流预测（适用于文件上传）"""
        import io
        from PIL import Image
        import torch
        
        if self._predictor is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # 从字节流加载图片
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 预处理
        image_tensor = self._predictor.transform(image).unsqueeze(0).to(self._predictor.device)
        
        # 推理
        with torch.no_grad():
            outputs = self._predictor.model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': self._predictor.class_names[predicted_class],
            'confidence': confidence_score,
            'class_probabilities': {
                self._predictor.class_names[i]: probs[0][i].item()
                for i in range(len(self._predictor.class_names))
            }
        }

# 全局实例
predictor = ModelPredictor()
```

### 3.2 服务器启动时初始化

```python
# 在服务器启动脚本中
from predictor_singleton import predictor

# 服务器启动时调用
predictor.initialize(checkpoint_path='/path/to/models/best_model.pth')
```

---

## 4. API 接口示例

### 4.1 Flask 示例

```python
from flask import Flask, request, jsonify
from predictor_singleton import predictor
import os

app = Flask(__name__)

# 服务器启动时初始化模型
predictor.initialize(checkpoint_path='models/best_model.pth')

@app.route('/api/qrcode/predict', methods=['POST'])
def predict_qrcode():
    """
    预测 QR 码图片
    
    请求方式: POST
    Content-Type: multipart/form-data
    参数: file (图片文件)
    
    返回示例:
    {
        "success": true,
        "data": {
            "predicted_label": "copied",
            "confidence": 0.6288,
            "class_probabilities": {
                "copied": 0.6288,
                "original": 0.3712
            }
        }
    }
    """
    try:
        # 检查文件
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        # 验证文件类型
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if ext not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # 读取文件字节
        image_bytes = file.read()
        
        # 预测
        result = predictor.predict_from_bytes(image_bytes)
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/qrcode/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor._predictor is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 4.2 FastAPI 示例

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from predictor_singleton import predictor
from typing import Dict

app = FastAPI(title="QR Code Classification API")

# 服务器启动时初始化模型
@app.on_event("startup")
async def startup_event():
    predictor.initialize(checkpoint_path='models/best_model.pth')

@app.post("/api/qrcode/predict")
async def predict_qrcode(file: UploadFile = File(...)) -> Dict:
    """
    预测 QR 码图片
    
    - **file**: 上传的图片文件
    """
    # 验证文件类型
    allowed_types = {'image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/gif'}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        # 读取文件
        image_bytes = await file.read()
        
        # 预测
        result = predictor.predict_from_bytes(image_bytes)
        
        return {
            'success': True,
            'data': result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/qrcode/health")
async def health_check():
    """健康检查"""
    return {
        'status': 'healthy',
        'model_loaded': predictor._predictor is not None
    }

# 运行: uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4.3 Django 示例

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from predictor_singleton import predictor

# 在 Django 启动时初始化（apps.py）
# from django.apps import AppConfig
# class YourAppConfig(AppConfig):
#     def ready(self):
#         from predictor_singleton import predictor
#         predictor.initialize(checkpoint_path='models/best_model.pth')

@csrf_exempt
@require_http_methods(["POST"])
def predict_qrcode(request):
    """预测 QR 码图片"""
    try:
        # 获取上传的文件
        if 'file' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No file provided'}, status=400)
        
        file = request.FILES['file']
        
        # 读取文件字节
        image_bytes = file.read()
        
        # 预测
        result = predictor.predict_from_bytes(image_bytes)
        
        return JsonResponse({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def health_check(request):
    """健康检查"""
    return JsonResponse({
        'status': 'healthy',
        'model_loaded': predictor._predictor is not None
    })
```

---

## 5. 性能优化建议

### 5.1 使用 GPU 加速

```python
# 在初始化时指定设备
predictor.initialize(checkpoint_path='models/best_model.pth')

# 检查是否使用 GPU
import torch
print(f"Using device: {predictor._predictor.device}")
```

### 5.2 批量预测优化

如果需要批量处理多张图片：

```python
def predict_batch(self, image_bytes_list):
    """批量预测多张图片"""
    import torch
    from PIL import Image
    import io
    
    images = []
    for image_bytes in image_bytes_list:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = self._predictor.transform(image)
        images.append(image_tensor)
    
    # 批量处理
    batch_tensor = torch.stack(images).to(self._predictor.device)
    
    with torch.no_grad():
        outputs = self._predictor.model(batch_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, 1)
    
    results = []
    for i in range(len(images)):
        results.append({
            'predicted_class': predictions[i].item(),
            'predicted_label': self._predictor.class_names[predictions[i].item()],
            'confidence': confidences[i].item(),
            'class_probabilities': {
                self._predictor.class_names[j]: probs[i][j].item()
                for j in range(len(self._predictor.class_names))
            }
        })
    
    return results
```

### 5.3 添加缓存（可选）

对于重复的图片请求，可以添加缓存：

```python
from functools import lru_cache
import hashlib

def get_image_hash(image_bytes):
    """计算图片哈希值"""
    return hashlib.md5(image_bytes).hexdigest()

# 使用 Redis 或内存缓存
cache = {}

def predict_with_cache(image_bytes):
    img_hash = get_image_hash(image_bytes)
    
    if img_hash in cache:
        return cache[img_hash]
    
    result = predictor.predict_from_bytes(image_bytes)
    cache[img_hash] = result
    
    return result
```

### 5.4 异步处理（高并发场景）

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def predict_async(image_bytes):
    """异步预测"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        predictor.predict_from_bytes,
        image_bytes
    )
    return result
```

---

## 6. 测试接口

### 6.1 使用 cURL 测试

```bash
# 测试预测接口
curl -X POST \
  http://your-server:5000/api/qrcode/predict \
  -F "file=@test_image.png"

# 测试健康检查
curl http://your-server:5000/api/qrcode/health
```

### 6.2 使用 Python requests 测试

```python
import requests

# 预测
url = "http://your-server:5000/api/qrcode/predict"
files = {'file': open('test_image.png', 'rb')}
response = requests.post(url, files=files)
print(response.json())

# 健康检查
response = requests.get("http://your-server:5000/api/qrcode/health")
print(response.json())
```

### 6.3 使用 JavaScript 测试

```javascript
// 文件上传预测
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://your-server:5000/api/qrcode/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## 7. 常见问题

### Q1: 模型加载很慢怎么办？
**A:** 模型只在服务器启动时加载一次（约 0.5 秒），之后每次预测只需 100-150ms。

### Q2: 如何减少内存占用？
**A:** 
- 使用 `torch.no_grad()` 避免计算梯度
- 及时清理临时文件
- 考虑使用模型量化（INT8）

### Q3: 支持哪些图片格式？
**A:** PNG, JPG, JPEG, BMP, GIF

### Q4: 单次预测耗时多少？
**A:** 
- CPU: ~140ms
- GPU: ~15-30ms

### Q5: 如何处理大批量图片？
**A:** 使用批量预测接口，或使用消息队列（如 Celery）异步处理。

---

## 8. 性能参考

基于 benchmark 测试结果：

| 指标 | CPU | GPU (预估) |
|-----|-----|-----------|
| 模型加载 | 534ms | 300ms |
| 单张预测 | 143ms | 15-30ms |
| 吞吐量 | 7 张/秒 | 35-70 张/秒 |

---

## 9. 完整示例项目结构

```
your_server/
├── models/
│   ├── best_model.pth
│   └── class_names.json
├── src/
│   ├── __init__.py
│   ├── inference.py
│   ├── model.py
│   ├── dataset.py
│   └── utils.py
├── predictor_singleton.py
├── app.py                    # Flask/FastAPI 主程序
└── requirements.txt
```

---

## 10. 联系与支持

如有问题，请参考：
- 项目 README.md
- benchmark.py（性能测试）
- test_api.py（接口测试）
