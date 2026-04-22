import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse


class ONNXPredictor:
    def __init__(self, onnx_path: str, input_size: Tuple[int, int] = (384, 384)):
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.session = self._load_model()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"模型加载成功: {onnx_path}")
        print(f"输入形状: {self.session.get_inputs()[0].shape}")
        print(f"输出形状: {self.session.get_outputs()[0].shape}")
    
    def _load_model(self):
        return ort.InferenceSession(self.onnx_path)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        
        if img.mode != 'RGBA':
            raise ValueError(f"图片必须是4通道(RGBA)格式，当前格式: {img.mode}")
        
        img_array = np.array(img, dtype=np.float32)
        
        mean = np.array([0.485, 0.456, 0.406, 0.456], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225, 0.224], dtype=np.float32)
        
        img_array = cv2.resize(img_array, (self.input_size[0], self.input_size[1]), interpolation=cv2.INTER_LINEAR)
        
        img_array = img_array / 255.0
        img_array = (img_array - mean) / std
        
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        img_array = img_array.astype(np.float32)
        
        return img_array
    
    def predict(self, image_path: str) -> Dict[str, float]:
        input_tensor = self.preprocess_image(image_path)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0]
        probabilities = 1 / (1 + np.exp(-logits))
        
        labels = ["is_copied", "is_low_light", "is_blurry", "is_screen"]
        result = {label: float(prob) for label, prob in zip(labels, probabilities[0])}
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, float]]:
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"预测失败 {image_path}: {e}")
                results.append(None)
        return results


def main():
    parser = argparse.ArgumentParser(description='使用ONNX模型预测4通道PNG图片')
    parser.add_argument('--model', '-m', type=str, default='checkpoints/best_model.onnx',
                        help='ONNX模型路径')
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='要预测的图片路径')
    parser.add_argument('--input-size', '-s', type=int, default=384,
                        help='输入图片尺寸 (默认: 384)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出结果到JSON文件')
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"错误: 图片不存在 {args.image}")
        return
    
    predictor = ONNXPredictor(args.model, (args.input_size, args.input_size))
    
    print(f"\n预测图片: {args.image}")
    print("=" * 60)
    
    result = predictor.predict(args.image)
    
    for label, prob in result.items():
        status = "✓" if prob > 0.5 else "✗"
        print(f"{status} {label:20s}: {prob:.4f} ({prob*100:.2f}%)")
    
    print("=" * 60)
    
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
