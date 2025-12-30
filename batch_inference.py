"""
批量预测脚本：支持标准模型和对比学习模型

展示如何使用训练好的模型进行批量预测。
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
import json

from src.model import load_model_for_inference
from src.dataset import get_inference_transform
from src.utils import get_device


def batch_inference(
    model,
    image_paths: List[str],
    device: torch.device,
    image_size: int = 384,
    use_contrastive: bool = False,
    copied_threshold: float = 0.8,
    batch_size: int = 16
) -> List[Dict]:
    """
    批量预测
    
    Args:
        model: 训练好的模型
        image_paths: 图片路径列表
        device: 设备
        image_size: 输入尺寸
        use_contrastive: 是否使用对比学习模型
        copied_threshold: 复印标签的预测阈值
        batch_size: 批量大小
    
    Returns:
        预测结果列表
    """
    # 获取推理变换
    transform = get_inference_transform(image_size=image_size)
    
    # 存储预测结果
    results = []
    
    # 分批处理
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch Inference"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            # 加载和预处理图片
            for img_path in batch_paths:
                try:
                    # 加载图片
                    image = Image.open(img_path)
                    
                    # 处理RGBA
                    if image.mode == 'RGBA':
                        image_4c = image
                    else:
                        # 处理1通道或3通道图片
                        image = image.convert('RGB')
                        
                        # 如果需要4通道，这里简化处理
                        # 实际应用中应该使用QRCodeRegistrator生成第4通道
                        # 这里创建一个全0的第4通道
                        image_np = np.array(image)
                        fourth_channel = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
                        combined_img = np.dstack((image_np, fourth_channel))
                        image_4c = Image.fromarray(combined_img, 'RGBA')
                    
                    # 应用变换
                    image_tensor = transform(image_4c)
                    batch_images.append(image_tensor)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
            
            # 转换为tensor
            batch_tensor = torch.stack(batch_images).to(device)
            
            # 预测
            if use_contrastive:
                # 对比学习模型返回 (output, rgb_feat, ref_feat)
                outputs, _, _ = model(batch_tensor)
            else:
                # 标准模型只返回output
                outputs = model(batch_tensor)
            
            # 转换为概率
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # 保存结果
            for j, img_path in enumerate(batch_paths):
                result = {
                    'image_path': img_path,
                    'image_name': os.path.basename(img_path),
                    'predictions': {
                        'is_copied': float(probs[j, 0]),
                        'is_low_light': float(probs[j, 1]),
                        'is_blurry': float(probs[j, 2])
                    },
                    'labels': {
                        'is_copied': int(probs[j, 0] >= copied_threshold),
                        'is_low_light': int(probs[j, 1] >= 0.5),
                        'is_blurry': int(probs[j, 2] >= 0.5)
                    }
                }
                results.append(result)
    
    return results


def save_predictions(results: List[Dict], output_path: str):
    """
    保存预测结果到JSON文件
    
    Args:
        results: 预测结果列表
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Predictions saved to {output_path}")


def print_prediction_summary(results: List[Dict]):
    """
    打印预测结果摘要
    
    Args:
        results: 预测结果列表
    """
    print("\n" + "=" * 70)
    print("Prediction Summary")
    print("=" * 70)
    
    # 统计每个标签的预测数量
    label_counts = {
        'is_copied': 0,
        'is_low_light': 0,
        'is_blurry': 0
    }
    
    for result in results:
        labels = result['labels']
        for label_name in label_counts:
            if labels[label_name] == 1:
                label_counts[label_name] += 1
    
    total = len(results)
    
    print(f"\nTotal predictions: {total}")
    print(f"\nLabel distribution:")
    for label_name, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"  {label_name}: {count} ({percentage:.2f}%)")
    
    print("\n" + "=" * 70)


def batch_inference_from_directory(
    model,
    input_dir: str,
    device: torch.device,
    image_size: int = 384,
    use_contrastive: bool = False,
    copied_threshold: float = 0.8,
    batch_size: int = 16,
    output_path: str = 'predictions.json'
):
    """
    从目录批量预测
    
    Args:
        model: 训练好的模型
        input_dir: 输入目录
        device: 设备
        image_size: 输入尺寸
        use_contrastive: 是否使用对比学习模型
        copied_threshold: 复印标签的预测阈值
        batch_size: 批量大小
        output_path: 输出文件路径
    """
    # 获取所有图片路径
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    
    if len(image_paths) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    
    # 批量预测
    results = batch_inference(
        model=model,
        image_paths=image_paths,
        device=device,
        image_size=image_size,
        use_contrastive=use_contrastive,
        copied_threshold=copied_threshold,
        batch_size=batch_size
    )
    
    # 保存结果
    save_predictions(results, output_path)
    
    # 打印摘要
    print_prediction_summary(results)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch inference for QR code classification')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to input directory containing images')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Path to output JSON file')
    parser.add_argument('--image_size', type=int, default=384,
                       help='Input image size (default: 384)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference (default: 16)')
    parser.add_argument('--copied_threshold', type=float, default=0.8,
                       help='Threshold for is_copied prediction (default: 0.8)')
    parser.add_argument('--use_contrastive', action='store_true',
                       help='Use contrastive learning model')
    parser.add_argument('--model_name', type=str, default='convnextv2_tiny',
                       help='Model name (default: convnextv2_tiny)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # 获取设备
    device = get_device() if args.device is None else torch.device(args.device)
    
    print("=" * 70)
    print("Batch Inference for QR Code Classification")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output file: {args.output}")
    print(f"  Image size: {args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Copied threshold: {args.copied_threshold}")
    print(f"  Use contrastive: {args.use_contrastive}")
    print(f"  Model name: {args.model_name}")
    print(f"  Device: {device}")
    print("-" * 70)
    
    # 加载模型
    print("\nLoading model...")
    model = load_model_for_inference(
        checkpoint_path=args.checkpoint,
        num_labels=3,
        model_name=args.model_name,
        device=device,
        in_channels=4,
        use_contrastive=args.use_contrastive
    )
    
    # 批量预测
    batch_inference_from_directory(
        model=model,
        input_dir=args.input_dir,
        device=device,
        image_size=args.image_size,
        use_contrastive=args.use_contrastive,
        copied_threshold=args.copied_threshold,
        batch_size=args.batch_size,
        output_path=args.output
    )
    
    print("\n" + "=" * 70)
    print("Batch inference completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
