# API 测试脚本
# 用于测试模型 API 接口的功能和性能

import requests
import time
import os
import sys


def test_flask_api(base_url='http://localhost:5000'):
    """测试 Flask API"""
    print("=" * 80)
    print("测试 Flask API")
    print("=" * 80)
    
    # 1. 健康检查
    print("\n1. 健康检查...")
    try:
        response = requests.get(f"{base_url}/api/qrcode/health")
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   错误: {e}")
        return False
    
    # 2. 模型信息
    print("\n2. 模型信息...")
    try:
        response = requests.get(f"{base_url}/api/qrcode/model/info")
        print(f"   状态码: {response.status_code}")
        info = response.json()
        if info.get('initialized'):
            print(f"   模型: {info.get('model_name')}")
            print(f"   类别: {info.get('class_names')}")
            print(f"   设备: {info.get('device')}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 3. 单张图片预测
    print("\n3. 单张图片预测...")
    test_image = 'data/test/010.png'
    
    if not os.path.exists(test_image):
        print(f"   测试图片不存在: {test_image}")
        return False
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            start_time = time.time()
            response = requests.post(f"{base_url}/api/qrcode/predict", files=files)
            elapsed = time.time() - start_time
        
        print(f"   状态码: {response.status_code}")
        print(f"   耗时: {elapsed*1000:.2f} ms")
        
        result = response.json()
        if result.get('success'):
            data = result['data']
            print(f"   预测: {data['predicted_label']}")
            print(f"   置信度: {data['confidence']:.2%}")
        else:
            print(f"   错误: {result.get('error')}")
    except Exception as e:
        print(f"   错误: {e}")
        return False
    
    # 4. 批量预测
    print("\n4. 批量预测...")
    test_dir = 'data/test'
    
    if os.path.exists(test_dir):
        image_files = [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ][:3]  # 只测试前3张
        
        if image_files:
            try:
                files = [('files', open(path, 'rb')) for path in image_files]
                start_time = time.time()
                response = requests.post(f"{base_url}/api/qrcode/predict/batch", files=files)
                elapsed = time.time() - start_time
                
                # 关闭文件
                for _, f in files:
                    f.close()
                
                print(f"   状态码: {response.status_code}")
                print(f"   耗时: {elapsed*1000:.2f} ms")
                
                result = response.json()
                if result.get('success'):
                    print(f"   预测数量: {result.get('count')}")
                    for i, data in enumerate(result['data'], 1):
                        print(f"     {i}. {data.get('filename')}: {data['predicted_label']} ({data['confidence']:.2%})")
                else:
                    print(f"   错误: {result.get('error')}")
            except Exception as e:
                print(f"   错误: {e}")
    
    # 5. 性能测试
    print("\n5. 性能测试 (10次请求)...")
    times = []
    
    for i in range(10):
        try:
            with open(test_image, 'rb') as f:
                files = {'file': f}
                start_time = time.time()
                response = requests.post(f"{base_url}/api/qrcode/predict", files=files)
                elapsed = time.time() - start_time
                times.append(elapsed)
        except Exception as e:
            print(f"   请求 {i+1} 失败: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"   平均耗时: {avg_time*1000:.2f} ms")
        print(f"   最快: {min_time*1000:.2f} ms")
        print(f"   最慢: {max_time*1000:.2f} ms")
        print(f"   吞吐量: {1/avg_time:.1f} 请求/秒")
    
    # 6. 错误处理测试
    print("\n6. 错误处理测试...")
    
    # 6.1 无文件
    try:
        response = requests.post(f"{base_url}/api/qrcode/predict")
        print(f"   无文件 - 状态码: {response.status_code}")
    except Exception as e:
        print(f"   无文件 - 错误: {e}")
    
    # 6.2 错误文件类型
    try:
        files = {'file': ('test.txt', b'not an image', 'text/plain')}
        response = requests.post(f"{base_url}/api/qrcode/predict", files=files)
        print(f"   错误类型 - 状态码: {response.status_code}")
        print(f"   错误类型 - 响应: {response.json()}")
    except Exception as e:
        print(f"   错误类型 - 错误: {e}")
    
    print("\n" + "=" * 80)
    print("✓ 测试完成")
    print("=" * 80)
    
    return True


def test_fastapi(base_url='http://localhost:8000'):
    """测试 FastAPI"""
    print("=" * 80)
    print("测试 FastAPI")
    print("=" * 80)
    print(f"\n访问 API 文档: {base_url}/docs")
    print(f"访问 ReDoc: {base_url}/redoc")
    
    # FastAPI 测试逻辑与 Flask 类似
    return test_flask_api(base_url)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("\n用法:")
        print("  python test_api.py flask [url]")
        print("  python test_api.py fastapi [url]")
        print("\n示例:")
        print("  python test_api.py flask")
        print("  python test_api.py flask http://192.168.1.100:5000")
        print("  python test_api.py fastapi")
        sys.exit(1)
    
    api_type = sys.argv[1].lower()
    
    if api_type == 'flask':
        base_url = sys.argv[2] if len(sys.argv) > 2 else 'http://localhost:5000'
        test_flask_api(base_url)
    elif api_type == 'fastapi':
        base_url = sys.argv[2] if len(sys.argv) > 2 else 'http://localhost:8000'
        test_fastapi(base_url)
    else:
        print(f"未知的 API 类型: {api_type}")
        print("支持: flask, fastapi")
        sys.exit(1)


if __name__ == '__main__':
    main()
