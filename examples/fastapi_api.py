"""
FastAPI 示例
展示如何在 FastAPI 服务器中集成 QR 码分类模型
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor_singleton import predictor

app = FastAPI(
    title="QR Code Classification API",
    description="API for classifying QR code images as original or copied",
    version="1.0.0"
)

# 配置
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = {'image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/gif'}


@app.on_event("startup")
async def startup_event():
    """服务器启动时初始化模型"""
    checkpoint_path = os.environ.get('MODEL_PATH', 'checkpoints/best_model.pth')
    predictor.initialize(checkpoint_path=checkpoint_path)


@app.get("/")
async def root():
    """API 首页"""
    return {
        "service": "QR Code Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "/api/qrcode/predict (POST)",
            "batch_predict": "/api/qrcode/predict/batch (POST)",
            "health": "/api/qrcode/health (GET)",
            "model_info": "/api/qrcode/model/info (GET)"
        }
    }


@app.post("/api/qrcode/predict", response_model=Dict)
async def predict_qrcode(file: UploadFile = File(...)):
    """
    预测单张 QR 码图片
    
    - **file**: 上传的图片文件 (PNG, JPG, JPEG, BMP, GIF)
    
    返回:
    ```json
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
    ```
    """
    # 验证文件类型
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {', '.join(ALLOWED_TYPES)}"
        )
    
    try:
        # 读取文件
        image_bytes = await file.read()
        
        # 检查文件大小
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # 预测
        result = predictor.predict_from_bytes(image_bytes)
        
        response_data = {"success": True}
        
        if 'predictions' in result:
             # Multi-label format
             response_data['data'] = result
        else:
             # Single-label format (legacy compatibility)
             response_data['data'] = {
                "predicted_label": result['predicted_label'],
                "confidence": result['confidence'],
                "class_probabilities": result['class_probabilities']
            }
            
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qrcode/predict/batch", response_model=Dict)
async def batch_predict_qrcode(files: List[UploadFile] = File(...)):
    """
    批量预测多张 QR 码图片
    
    - **files**: 多个图片文件
    """
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        # 读取所有文件
        image_bytes_list = []
        filenames = []
        
        for file in files:
            # 验证文件类型
            if file.content_type not in ALLOWED_TYPES:
                continue
            
            image_bytes = await file.read()
            
            # 检查文件大小
            if len(image_bytes) > MAX_FILE_SIZE:
                continue
            
            image_bytes_list.append(image_bytes)
            filenames.append(file.filename)
        
        if len(image_bytes_list) == 0:
            raise HTTPException(status_code=400, detail="No valid files provided")
        
        # 批量预测
        results = predictor.predict_batch_from_bytes(image_bytes_list)
        
        # 添加文件名
        for i, result in enumerate(results):
            result['filename'] = filenames[i]
        
        return {
            "success": True,
            "count": len(results),
            "data": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qrcode/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": predictor.is_initialized()
    }


@app.get("/api/qrcode/model/info")
async def model_info():
    """获取模型信息"""
    return predictor.get_model_info()


if __name__ == "__main__":
    import uvicorn
    
    # 开发环境运行
    # 生产环境建议使用: uvicorn fastapi_api:app --host 0.0.0.0 --port 8000 --workers 4
    uvicorn.run(app, host="0.0.0.0", port=8000)
