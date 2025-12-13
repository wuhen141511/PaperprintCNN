"""
预测器单例模式
用于在服务器中集成模型，避免重复加载
"""

import io
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List

from src.inference import QRCodePredictor


class ModelPredictor:
    """
    模型预测器单例
    
    使用方法:
        # 1. 服务器启动时初始化
        predictor = ModelPredictor()
        predictor.initialize(checkpoint_path='models/best_model.pth')
        
        # 2. 在接口中使用
        result = predictor.predict_from_bytes(image_bytes)
    """
    
    _instance = None
    _predictor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, checkpoint_path='checkpoints/best_model.pth', model_name='convnextv2_tiny', multi_label=True):
        """
        初始化模型（服务器启动时调用一次）
        
        Args:
            checkpoint_path: 模型检查点路径
            model_name: 模型名称，默认 convnextv2_tiny
            multi_label: 是否开启多标签模式
        """
        if self._predictor is None:
            print("=" * 80)
            print("Loading QR Code Classification Model...")
            print("=" * 80)
            
            self._predictor = QRCodePredictor(
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                multi_label=multi_label
            )
            
            print("=" * 80)
            print("✓ Model loaded successfully and ready for predictions!")
            print("=" * 80)
            print()
    
    def is_initialized(self) -> bool:
        """检查模型是否已初始化"""
        return self._predictor is not None
    
    def predict(self, image_path: str) -> Dict:
        """
        预测单张图片（从文件路径）
        """
        if self._predictor is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        return self._predictor.predict_image(image_path, return_probs=True)
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict:
        """从字节流预测（适用于文件上传）"""
        if self._predictor is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # 从字节流加载图片
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 预处理
        image_tensor = self._predictor.transform(image).unsqueeze(0).to(self._predictor.device)
        
        # 推理
        with torch.no_grad():
            outputs = self._predictor.model(image_tensor)
            
            if self._predictor.multi_label:
                probs = torch.sigmoid(outputs)
                predictions = (probs >= self._predictor.threshold).float()
                
                result = {
                    'predictions': {},
                    'probabilities': {}
                }
                
                for i, label_name in enumerate(self._predictor.label_names):
                    result['predictions'][label_name] = {
                        'value': int(predictions[0][i].item()),
                        'probability': float(probs[0][i].item())
                    }
                    result['probabilities'][label_name] = float(probs[0][i].item())
                return result
            else:
                probs = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
            
                predicted_class = predicted.item()
                confidence_score = confidence.item()
                
                return {
                    'predicted_class': predicted_class,
                    'predicted_label': self._predictor.label_names[predicted_class],
                    'confidence': confidence_score,
                    'class_probabilities': {
                        self._predictor.label_names[i]: probs[0][i].item()
                        for i in range(len(self._predictor.label_names))
                    }
                }
    
    def predict_batch_from_bytes(self, image_bytes_list: List[bytes]) -> List[Dict]:
        """批量预测多张图片（从字节流）"""
        if self._predictor is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # 加载所有图片
        images = []
        for image_bytes in image_bytes_list:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self._predictor.transform(image)
            images.append(image_tensor)
        
        # 批量处理
        batch_tensor = torch.stack(images).to(self._predictor.device)
        
        with torch.no_grad():
            outputs = self._predictor.model(batch_tensor)
            
            if self._predictor.multi_label:
                probs = torch.sigmoid(outputs)
                predictions = (probs >= self._predictor.threshold).float()
                results = []
                for i in range(len(images)):
                    res = {'predictions': {}, 'probabilities': {}}
                    for j, label_name in enumerate(self._predictor.label_names):
                        res['predictions'][label_name] = {
                            'value': int(predictions[i][j].item()),
                            'probability': float(probs[i][j].item())
                        }
                        res['probabilities'][label_name] = float(probs[i][j].item())
                    results.append(res)
                return results
            else:
                probs = F.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probs, 1)
        
                results = []
                for i in range(len(images)):
                    results.append({
                        'predicted_class': predictions[i].item(),
                        'predicted_label': self._predictor.label_names[predictions[i].item()],
                        'confidence': confidences[i].item(),
                        'class_probabilities': {
                            self._predictor.label_names[j]: probs[i][j].item()
                            for j in range(len(self._predictor.label_names))
                        }
                    })
                return results
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self._predictor is None:
            return {
                'initialized': False,
                'error': 'Model not initialized'
            }
        
        return {
            'initialized': True,
            'model_name': 'convnextv2_tiny',
            'num_classes': len(self._predictor.label_names),
            'class_names': self._predictor.label_names,
            'device': str(self._predictor.device),
            'image_size': self._predictor.image_size,
            'multi_label': self._predictor.multi_label
        }


# 全局单例实例
predictor = ModelPredictor()


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predictor_singleton.py <image_path>")
        sys.exit(1)
    
    # 初始化
    predictor.initialize()
    
    # 测试文件路径预测
    image_path = sys.argv[1]
    result = predictor.predict(image_path)
    
    print("\n" + "=" * 60)
    print("Prediction Result:")
    print("=" * 60)
    print(f"Predicted: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nClass Probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        print(f"  {class_name}: {prob:.2%}")
    print("=" * 60)
    
    # 测试字节流预测
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    result2 = predictor.predict_from_bytes(image_bytes)
    print("\n✓ Bytes prediction test passed!")
    
    # 获取模型信息
    info = predictor.get_model_info()
    print("\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
