"""
批量预测脚本 - 测试多张图片
"""

import os
import sys
import shutil
from src.inference import QRCodePredictor

def batch_predict(image_dir, checkpoint_path='checkpoints/best_model.pth', move_files=False):
    """
    批量预测目录中的所有图片
    
    Args:
        image_dir: 包含测试图片的目录
        checkpoint_path: 模型检查点路径
        move_files: 是否将测试文件按预测结果移动到对应的子目录中（默认 False）
    """
    # 创建预测器
    predictor = QRCodePredictor(
        checkpoint_path=checkpoint_path,
        model_name='resnet50'
    )
    
    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) 
        if f.lower().endswith(image_extensions)
    ]
    
    if len(image_files) == 0:
        print(f"错误：在 {image_dir} 中没有找到图片文件")
        return
    
    print(f"\n找到 {len(image_files)} 张图片，开始预测...\n")
    print("="*80)
    
    # 批量预测
    results = predictor.predict_batch(image_files)
    
    # 如果需要移动文件，创建输出目录
    if move_files:
        # 获取所有可能的类别
        all_classes = set(result['predicted_label'] for result in results)
        for class_name in all_classes:
            class_dir = os.path.join(image_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        print(f"\n已创建类别目录: {', '.join(all_classes)}\n")
    
    # 显示结果并移动文件（如果需要）
    for i, result in enumerate(results, 1):
        filename = os.path.basename(result['image_path'])
        predicted_label = result['predicted_label']
        confidence = result['confidence']
        
        print(f"{i}. {filename}")
        print(f"   预测: {predicted_label} (置信度: {confidence:.2%})")
        
        # 显示详细概率
        probs = result['class_probabilities']
        for class_name, prob in probs.items():
            print(f"      - {class_name}: {prob:.2%}")
        
        # 移动文件到对应的类别目录
        if move_files:
            # 构造新文件名：原文件名_置信度.扩展名
            name_without_ext, ext = os.path.splitext(filename)
            new_filename = f"{name_without_ext}_{confidence:.4f}{ext}"
            
            # 目标路径
            target_dir = os.path.join(image_dir, predicted_label)
            target_path = os.path.join(target_dir, new_filename)
            
            # 移动文件
            shutil.move(result['image_path'], target_path)
            print(f"   → 已移动到: {os.path.join(predicted_label, new_filename)}")
        
        print()
    
    print("="*80)
    
    # 统计
    predictions_count = {}
    for result in results:
        label = result['predicted_label']
        predictions_count[label] = predictions_count.get(label, 0) + 1
    
    print("\n预测统计:")
    for label, count in predictions_count.items():
        print(f"  {label}: {count} 张 ({count/len(results)*100:.1f}%)")
    
    print(f"\n总计: {len(results)} 张图片")
    
    if move_files:
        print(f"\n✓ 所有文件已按预测结果移动到对应的子目录中")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='批量预测QR码图片的类别',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_predict.py test_images
  python batch_predict.py test_images --checkpoint checkpoints/best_model.pth
  python batch_predict.py test_images --move-files
  python batch_predict.py test_images --checkpoint checkpoints/best_model.pth --move-files
        """
    )
    
    parser.add_argument(
        'image_dir',
        type=str,
        help='包含测试图片的目录'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='模型检查点路径（默认: checkpoints/best_model.pth）'
    )
    
    parser.add_argument(
        '--move-files',
        action='store_true',
        default=False,
        help='将测试文件按预测结果移动到对应的子目录中，并在文件名末尾附加置信度'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        print(f"错误：目录不存在: {args.image_dir}")
        sys.exit(1)
    
    batch_predict(args.image_dir, args.checkpoint, args.move_files)

