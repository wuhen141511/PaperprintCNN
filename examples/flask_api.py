"""
Flask API 示例
展示如何在 Flask 服务器中集成 QR 码分类模型
"""

from flask import Flask, request, jsonify
from predictor_singleton import predictor
import os

app = Flask(__name__)

# 配置
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.before_first_request
def initialize_model():
    """服务器启动时初始化模型"""
    checkpoint_path = os.environ.get('MODEL_PATH', 'checkpoints/best_model.pth')
    predictor.initialize(checkpoint_path=checkpoint_path)


@app.route('/')
def index():
    """API 首页"""
    return jsonify({
        'service': 'QR Code Classification API',
        'version': '1.0.0',
        'endpoints': {
            'predict': '/api/qrcode/predict (POST)',
            'batch_predict': '/api/qrcode/predict/batch (POST)',
            'health': '/api/qrcode/health (GET)',
            'model_info': '/api/qrcode/model/info (GET)'
        }
    })


@app.route('/api/qrcode/predict', methods=['POST'])
def predict_qrcode():
    """
    预测单张 QR 码图片
    
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
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        # 验证文件类型
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # 检查文件大小
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB'
            }), 400
        
        # 读取文件字节
        image_bytes = file.read()
        
        # 预测
        result = predictor.predict_from_bytes(image_bytes)
        
        return jsonify({
            'success': True,
            'data': {
                'predicted_label': result['predicted_label'],
                'confidence': result['confidence'],
                'class_probabilities': result['class_probabilities']
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/qrcode/predict/batch', methods=['POST'])
def batch_predict_qrcode():
    """
    批量预测多张 QR 码图片
    
    请求方式: POST
    Content-Type: multipart/form-data
    参数: files[] (多个图片文件)
    """
    try:
        # 检查文件
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        
        if len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        # 读取所有文件
        image_bytes_list = []
        filenames = []
        
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                continue
            
            image_bytes = file.read()
            image_bytes_list.append(image_bytes)
            filenames.append(file.filename)
        
        if len(image_bytes_list) == 0:
            return jsonify({
                'success': False,
                'error': 'No valid files provided'
            }), 400
        
        # 批量预测
        results = predictor.predict_batch_from_bytes(image_bytes_list)
        
        # 添加文件名
        for i, result in enumerate(results):
            result['filename'] = filenames[i]
        
        return jsonify({
            'success': True,
            'count': len(results),
            'data': results
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
        'model_loaded': predictor.is_initialized()
    })


@app.route('/api/qrcode/model/info', methods=['GET'])
def model_info():
    """获取模型信息"""
    info = predictor.get_model_info()
    return jsonify(info)


@app.errorhandler(404)
def not_found(error):
    """404 错误处理"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 错误处理"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # 开发环境运行
    # 生产环境建议使用 gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 flask_api:app
    app.run(host='0.0.0.0', port=5000, debug=False)
