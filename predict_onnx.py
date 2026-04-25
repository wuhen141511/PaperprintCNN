import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse


class ONNXPredictor:
    def __init__(self, onnx_path: str, input_size: Tuple[int, int] = (384, 384),
                 use_contrastive: bool = False, use_gray: bool = False):
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.use_contrastive = use_contrastive
        self.use_gray = use_gray
        self.session = self._load_model()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        if use_contrastive:
            if use_gray:
                self.mean = np.array([0.449, 0.449, 0.449, 0.449], dtype=np.float32)
                self.std = np.array([0.226, 0.226, 0.226, 0.226], dtype=np.float32)
            else:
                self.mean = np.array([0.485, 0.456, 0.406, 0.449], dtype=np.float32)
                self.std = np.array([0.229, 0.224, 0.225, 0.226], dtype=np.float32)
        else:
            if use_gray:
                self.mean = np.array([0.449, 0.449, 0.449], dtype=np.float32)
                self.std = np.array([0.226, 0.226, 0.226], dtype=np.float32)
            else:
                self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print(f"模型加载成功: {onnx_path}")
        print(f"输入形状: {self.session.get_inputs()[0].shape}")
        print(f"输出形状: {self.session.get_outputs()[0].shape}")
    
    def _load_model(self):
        return ort.InferenceSession(self.onnx_path)
    
    def get_model_version(self) -> str:
        model_meta = self.session.get_modelmeta()
        custom_metadata = model_meta.custom_metadata_map
        version = custom_metadata.get("custom_version", str(model_meta.version))
        return version
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        
        if self.use_contrastive:
            if img.mode == 'RGBA':
                r, g, b, a = img.split()
                if self.use_gray:
                    gray = Image.merge('RGB', [r, g, b]).convert('L')
                    r = gray
                    g = gray
                    b = gray
                img = Image.merge('RGBA', [r, g, b, a])
            else:
                rgb_img = img.convert('RGB')
                if self.use_gray:
                    gray = rgb_img.convert('L')
                    r, g, b = gray, gray, gray
                else:
                    r, g, b = rgb_img.split()
                a = Image.new('L', img.size, 255)
                img = Image.merge('RGBA', [r, g, b, a])
        else:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            else:
                img = img.convert('RGB')
            if self.use_gray:
                gray = img.convert('L')
                img = Image.merge('RGB', [gray, gray, gray])
        
        img_array = np.array(img, dtype=np.float32)
        img_array = cv2.resize(img_array, (self.input_size[0], self.input_size[1]), interpolation=cv2.INTER_LINEAR)
        img_array = img_array / 255.0
        img_array = (img_array - self.mean) / self.std
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
    parser = argparse.ArgumentParser(description='使用ONNX模型预测图片')
    parser.add_argument('--model', '-m', type=str, default='checkpoints/best_model.onnx',
                        help='ONNX模型路径')
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='要预测的图片路径')
    parser.add_argument('--input-size', '-s', type=int, default=512,
                        help='输入图片尺寸 (默认: 384)')
    parser.add_argument('--use-contrastive', action='store_true',
                        help='使用对比学习模式 (4通道输入)')
    parser.add_argument('--use-gray', action='store_true',
                        help='使用灰度图模式')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出结果到JSON文件')
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"错误: 图片不存在 {args.image}")
        return
    
    predictor = ONNXPredictor(args.model, (args.input_size, args.input_size),
                              args.use_contrastive, args.use_gray)
    
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
