"""
批量预测脚本 - 测试多张图片
"""

import os
import sys
from src.inference import QRCodePredictor

def batch_predict(image_dir, checkpoint_path='checkpoints/best_model.pth'):
    """
    批量预测目录中的所有图片
    
    Args:
        image_dir: 包含测试图片的目录
        checkpoint_path: 模型检查点路径
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
    
    # 显示结果
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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n用法: python batch_predict.py <图片目录> [模型路径]")
        print("\n示例:")
        print("  python batch_predict.py test_images")
        print("  python batch_predict.py E:\\test_images checkpoints/best_model.pth")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/best_model.pth'
    
    if not os.path.exists(image_dir):
        print(f"错误：目录不存在: {image_dir}")
        sys.exit(1)
    
    batch_predict(image_dir, checkpoint_path)
