"""
API 客户端示例
展示如何调用 QR 码分类 API
"""

import requests
import json
import os


class QRCodeAPIClient:
    """QR 码分类 API 客户端"""
    
    def __init__(self, base_url='http://localhost:5000'):
        """
        初始化客户端
        
        Args:
            base_url: API 服务器地址
        """
        self.base_url = base_url.rstrip('/')
    
    def predict(self, image_path):
        """
        预测单张图片
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            预测结果字典
        """
        url = f"{self.base_url}/api/qrcode/predict"
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        return response.json()
    
    def predict_batch(self, image_paths):
        """
        批量预测多张图片
        
        Args:
            image_paths: 图片文件路径列表
            
        Returns:
            预测结果列表
        """
        url = f"{self.base_url}/api/qrcode/predict/batch"
        
        files = [('files', open(path, 'rb')) for path in image_paths]
        
        try:
            response = requests.post(url, files=files)
            return response.json()
        finally:
            # 关闭所有文件
            for _, f in files:
                f.close()
    
    def health_check(self):
        """健康检查"""
        url = f"{self.base_url}/api/qrcode/health"
        response = requests.get(url)
        return response.json()
    
    def model_info(self):
        """获取模型信息"""
        url = f"{self.base_url}/api/qrcode/model/info"
        response = requests.get(url)
        return response.json()


def print_result(result):
    """打印预测结果"""
    if result.get('success'):
        data = result['data']
        print(f"  预测结果: {data['predicted_label']}")
        print(f"  置信度: {data['confidence']:.2%}")
        print("  类别概率:")
        for class_name, prob in data['class_probabilities'].items():
            print(f"    - {class_name}: {prob:.2%}")
    else:
        print(f"  错误: {result.get('error')}")


def main():
    """示例代码"""
    # 创建客户端
    client = QRCodeAPIClient(base_url='http://localhost:5000')
    
    print("=" * 80)
    print("QR Code Classification API - 客户端示例")
    print("=" * 80)
    
    # 1. 健康检查
    print("\n1. 健康检查...")
    health = client.health_check()
    print(f"   状态: {health.get('status')}")
    print(f"   模型已加载: {health.get('model_loaded')}")
    
    # 2. 获取模型信息
    print("\n2. 模型信息...")
    info = client.model_info()
    if info.get('initialized'):
        print(f"   模型名称: {info.get('model_name')}")
        print(f"   类别数量: {info.get('num_classes')}")
        print(f"   类别名称: {info.get('class_names')}")
        print(f"   设备: {info.get('device')}")
    
    # 3. 单张图片预测
    print("\n3. 单张图片预测...")
    test_image = 'data/test/010.png'
    
    if os.path.exists(test_image):
        print(f"   图片: {test_image}")
        result = client.predict(test_image)
        print_result(result)
    else:
        print(f"   图片不存在: {test_image}")
    
    # 4. 批量预测
    print("\n4. 批量预测...")
    test_dir = 'data/test'
    
    if os.path.exists(test_dir):
        image_files = [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if image_files:
            print(f"   图片数量: {len(image_files)}")
            result = client.predict_batch(image_files[:3])  # 只测试前3张
            
            if result.get('success'):
                print(f"   成功预测: {result.get('count')} 张")
                for i, data in enumerate(result['data'], 1):
                    print(f"\n   图片 {i}: {data.get('filename')}")
                    print(f"     预测: {data['predicted_label']} ({data['confidence']:.2%})")
            else:
                print(f"   错误: {result.get('error')}")
        else:
            print(f"   目录中没有图片: {test_dir}")
    else:
        print(f"   目录不存在: {test_dir}")
    
    print("\n" + "=" * 80)


# ============================================================================
# cURL 命令示例
# ============================================================================

CURL_EXAMPLES = """
# 1. 健康检查
curl http://localhost:5000/api/qrcode/health

# 2. 获取模型信息
curl http://localhost:5000/api/qrcode/model/info

# 3. 预测单张图片
curl -X POST \\
  http://localhost:5000/api/qrcode/predict \\
  -F "file=@data/test/010.png"

# 4. 批量预测
curl -X POST \\
  http://localhost:5000/api/qrcode/predict/batch \\
  -F "files=@data/test/010.png" \\
  -F "files=@data/test/09.png" \\
  -F "files=@data/test/14.png"
"""


# ============================================================================
# JavaScript/Fetch 示例
# ============================================================================

JAVASCRIPT_EXAMPLE = """
// 单张图片预测
async function predictImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:5000/api/qrcode/predict', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    
    if (result.success) {
        console.log('预测结果:', result.data.predicted_label);
        console.log('置信度:', result.data.confidence);
    } else {
        console.error('错误:', result.error);
    }
}

// 使用示例
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        predictImage(file);
    }
});

// 批量预测
async function predictBatch(files) {
    const formData = new FormData();
    
    for (const file of files) {
        formData.append('files', file);
    }
    
    const response = await fetch('http://localhost:5000/api/qrcode/predict/batch', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    
    if (result.success) {
        console.log('预测数量:', result.count);
        result.data.forEach((item, index) => {
            console.log(`图片 ${index + 1}:`, item.predicted_label, item.confidence);
        });
    }
}
"""


# ============================================================================
# Python Requests 详细示例
# ============================================================================

def advanced_examples():
    """高级使用示例"""
    
    # 1. 带错误处理的预测
    def predict_with_error_handling(image_path):
        try:
            url = "http://localhost:5000/api/qrcode/predict"
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(url, files=files, timeout=10)
            
            response.raise_for_status()  # 检查 HTTP 错误
            
            result = response.json()
            
            if result.get('success'):
                return result['data']
            else:
                print(f"API 错误: {result.get('error')}")
                return None
        
        except requests.exceptions.Timeout:
            print("请求超时")
            return None
        except requests.exceptions.ConnectionError:
            print("连接失败，请检查服务器是否运行")
            return None
        except Exception as e:
            print(f"未知错误: {e}")
            return None
    
    # 2. 异步批量预测
    import concurrent.futures
    
    def predict_async(image_paths, max_workers=4):
        """异步批量预测多张图片"""
        client = QRCodeAPIClient()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(client.predict, path): path for path in image_paths}
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results.append((path, result))
                except Exception as e:
                    print(f"预测失败 {path}: {e}")
            
            return results
    
    # 3. 保存结果到文件
    def predict_and_save(image_path, output_file='result.json'):
        """预测并保存结果"""
        client = QRCodeAPIClient()
        result = client.predict(image_path)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {output_file}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--curl':
            print(CURL_EXAMPLES)
        elif sys.argv[1] == '--js':
            print(JAVASCRIPT_EXAMPLE)
        else:
            print("Usage: python api_client.py [--curl|--js]")
    else:
        main()
