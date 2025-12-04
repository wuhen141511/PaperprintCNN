"""
部署包打包脚本
将服务器集成所需的所有文件打包成 ZIP 文件
"""

import os
import zipfile
import shutil
import json
from datetime import datetime


def create_deployment_package():
    """创建部署包"""
    
    print("=" * 80)
    print("QR Code Classification - 部署包打包工具")
    print("=" * 80)
    
    # 配置
    package_name = f"qrcode_model_deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    # 必需文件列表
    required_files = {
        # 核心文件
        'predictor_singleton.py': '核心预测器',
        
        # 模型文件
        'checkpoints/best_model.pth': '模型权重',
        'checkpoints/class_names.json': '类别配置',
        
        # 源代码
        'src/__init__.py': '源码包初始化',
        'src/inference.py': '推理模块',
        'src/model.py': '模型定义',
        'src/dataset.py': '数据处理',
        'src/utils.py': '工具函数',
        
        # 文档
        'QUICKSTART.md': '快速开始指南',
        'INTEGRATION_GUIDE.md': '完整集成指南',
        'README.md': '项目说明',
        
        # 示例代码
        'examples/flask_api.py': 'Flask API 示例',
        'examples/fastapi_api.py': 'FastAPI 示例',
        'examples/api_client.py': '客户端示例',
        
        # 测试工具
        'test_api.py': 'API 测试脚本',
    }
    
    # 可选文件（如果存在则包含）
    optional_files = {
        'requirements.txt': '依赖列表',
        'config.yaml': '配置文件示例',
    }
    
    print(f"\n📦 准备打包...")
    print(f"   输出文件: {package_name}\n")
    
    # 检查必需文件
    missing_files = []
    existing_files = []
    
    print("检查必需文件:")
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            size_str = format_size(file_size)
            print(f"   ✓ {file_path:<40} ({size_str}) - {description}")
            existing_files.append(file_path)
        else:
            print(f"   ✗ {file_path:<40} - {description} [缺失]")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ 错误: 缺少 {len(missing_files)} 个必需文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    # 检查可选文件
    print("\n检查可选文件:")
    for file_path, description in optional_files.items():
        if os.path.exists(file_path):
            print(f"   ✓ {file_path:<40} - {description}")
            existing_files.append(file_path)
        else:
            print(f"   - {file_path:<40} - {description} [跳过]")
    
    # 创建 ZIP 文件
    print(f"\n📦 开始打包...")
    
    total_size = 0
    file_count = 0
    
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in existing_files:
            # 添加文件到 ZIP
            arcname = file_path  # 保持原始路径结构
            zipf.write(file_path, arcname)
            
            file_size = os.path.getsize(file_path)
            total_size += file_size
            file_count += 1
            
            print(f"   ✓ 添加: {file_path}")
        
        # 创建 README_DEPLOYMENT.txt
        deployment_readme = create_deployment_readme()
        zipf.writestr('README_DEPLOYMENT.txt', deployment_readme)
        print(f"   ✓ 添加: README_DEPLOYMENT.txt (部署说明)")
        file_count += 1
        
        # 创建 requirements.txt（如果不存在）
        if 'requirements.txt' not in existing_files:
            requirements = create_requirements()
            zipf.writestr('requirements.txt', requirements)
            print(f"   ✓ 添加: requirements.txt (自动生成)")
            file_count += 1
    
    # 完成
    package_size = os.path.getsize(package_name)
    
    print("\n" + "=" * 80)
    print("✅ 打包完成!")
    print("=" * 80)
    print(f"📦 包名称: {package_name}")
    print(f"📊 文件数量: {file_count} 个")
    print(f"💾 原始大小: {format_size(total_size)}")
    print(f"🗜️  压缩后: {format_size(package_size)}")
    print(f"📉 压缩率: {(1 - package_size/total_size)*100:.1f}%")
    print("=" * 80)
    
    print(f"\n📝 包含内容:")
    print(f"   ✓ 核心预测器 (predictor_singleton.py)")
    print(f"   ✓ 模型文件 (best_model.pth + class_names.json)")
    print(f"   ✓ 源代码 (src/)")
    print(f"   ✓ 文档 (QUICKSTART.md, INTEGRATION_GUIDE.md, README.md)")
    print(f"   ✓ API 示例 (Flask, FastAPI)")
    print(f"   ✓ 客户端示例 (Python, cURL, JavaScript)")
    print(f"   ✓ 测试工具 (test_api.py)")
    print(f"   ✓ 部署说明 (README_DEPLOYMENT.txt)")
    print(f"   ✓ 依赖列表 (requirements.txt)")
    
    print(f"\n🚀 使用方法:")
    print(f"   1. 将 {package_name} 发送给服务器对接同事")
    print(f"   2. 解压到服务器目录")
    print(f"   3. 阅读 README_DEPLOYMENT.txt 开始集成")
    
    print(f"\n💡 提示:")
    print(f"   - 包文件位置: {os.path.abspath(package_name)}")
    print(f"   - 解压后即可使用，无需额外配置")
    
    return True


def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def create_deployment_readme():
    """创建部署说明文件"""
    return """
================================================================================
QR Code Classification Model - 部署包
================================================================================

📦 包内容
--------
本部署包包含将 QR 码分类模型集成到服务器所需的所有文件。

📁 文件结构
----------
qrcode_model_deployment/
├── predictor_singleton.py          # 核心预测器（必需）
├── checkpoints/
│   ├── best_model.pth              # 模型权重文件 (~102MB)
│   └── class_names.json            # 类别配置
├── src/                            # 源代码（必需）
│   ├── __init__.py
│   ├── inference.py
│   ├── model.py
│   ├── dataset.py
│   └── utils.py
├── examples/                       # API 示例代码
│   ├── flask_api.py                # Flask API 完整示例
│   ├── fastapi_api.py              # FastAPI 完整示例
│   └── api_client.py               # 客户端调用示例
├── QUICKSTART.md                   # 快速开始指南（推荐先读）
├── INTEGRATION_GUIDE.md            # 完整集成文档
├── README.md                       # 项目说明
├── test_api.py                     # API 测试脚本
├── requirements.txt                # Python 依赖
└── README_DEPLOYMENT.txt           # 本文件

🚀 快速开始（3步集成）
--------------------

步骤 1: 安装依赖
--------------
pip install -r requirements.txt

或者手动安装：
pip install torch torchvision pillow numpy

步骤 2: 测试预测器
----------------
python predictor_singleton.py path/to/test_image.png

如果看到预测结果，说明环境配置正确。

步骤 3: 集成到服务器
------------------
在您的服务器代码中：

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

📖 详细文档
----------
- QUICKSTART.md        - 最简单的集成步骤（推荐先读）
- INTEGRATION_GUIDE.md - 完整的集成指南和示例
- README.md            - 项目完整说明

🔧 示例代码
----------
examples/ 目录包含完整的 API 示例：

1. Flask API 示例 (examples/flask_api.py)
   运行: python examples/flask_api.py
   访问: http://localhost:5000

2. FastAPI 示例 (examples/fastapi_api.py)
   运行: uvicorn examples.fastapi_api:app --host 0.0.0.0 --port 8000
   文档: http://localhost:8000/docs

3. 客户端示例 (examples/api_client.py)
   包含 Python、cURL、JavaScript 调用示例

🧪 测试
------
使用 test_api.py 测试 API 接口：

# 测试 Flask API
python test_api.py flask

# 测试 FastAPI
python test_api.py fastapi

⚡ 性能参考
----------
- CPU: 单张预测 ~140ms，吞吐量 7 张/秒
- GPU: 单张预测 ~20ms，吞吐量 50 张/秒
- 模型加载: ~500ms（仅启动时一次）

📋 API 响应格式
--------------
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

❓ 常见问题
----------
Q: 需要 GPU 吗？
A: 不需要，CPU 即可运行。有 GPU 会更快。

Q: 支持哪些图片格式？
A: PNG, JPG, JPEG, BMP, GIF

Q: 如何检查模型是否加载成功？
A: predictor.is_initialized() 返回 True 表示已加载

Q: 预测失败怎么办？
A: 检查图片格式是否正确，查看错误日志

📞 技术支持
----------
如有问题，请参考：
1. QUICKSTART.md - 快速开始指南
2. INTEGRATION_GUIDE.md - 完整集成文档
3. 项目 README.md

================================================================================
祝您集成顺利！
================================================================================
"""


def create_requirements():
    """创建 requirements.txt"""
    return """# QR Code Classification Model - 服务器部署依赖

# 核心依赖（必需）
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
numpy>=1.24.0

# API 框架（根据需要选择）
# Flask 示例需要:
# flask>=2.3.0

# FastAPI 示例需要:
# fastapi>=0.100.0
# uvicorn>=0.23.0
# python-multipart>=0.0.6

# 测试工具需要:
# requests>=2.31.0

# 说明:
# 1. 核心依赖是必需的
# 2. API 框架根据您使用的框架选择安装
# 3. 如果只是集成到现有服务器，只需安装核心依赖即可
"""


if __name__ == '__main__':
    import sys
    
    # 检查是否在项目根目录
    if not os.path.exists('predictor_singleton.py'):
        print("❌ 错误: 请在项目根目录下运行此脚本")
        print(f"   当前目录: {os.getcwd()}")
        print(f"   预期文件: predictor_singleton.py")
        sys.exit(1)
    
    # 执行打包
    success = create_deployment_package()
    
    if not success:
        sys.exit(1)
