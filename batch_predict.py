"""
批量预测脚本 - 测试多张图片
"""

import os
import sys
import csv
import time
from src.inference import QRCodePredictor

def batch_predict(image_dir, checkpoint_path='checkpoints/checkpoint_epoch_27.pth', 
                  output_csv='predictions.csv', multi_label=True, use_contrastive=False):
    """
    批量预测目录中的所有图片
    
    Args:
        image_dir: 包含测试图片的目录
        checkpoint_path: 模型检查点路径
        output_csv: 结果CSV文件路径
        multi_label: 是否启用多标签模式
        use_contrastive: 是否使用对比学习模型
    """
    # 创建预测器
    predictor = QRCodePredictor(
        checkpoint_path=checkpoint_path,
        model_name='convnextv2_pico',
        backend='opencv',
        multi_label=multi_label,
        use_contrastive=use_contrastive
    )
    
    # 获取标签列表
    label_names = predictor.label_names
    
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
    
    # 开始计时
    start_time = time.time()
    
    # 批量预测
    results = predictor.predict_batch(image_files)
    
    # 结束计时
    total_time = time.time() - start_time
    # 准备CSV数据
    csv_rows = []
    csv_header = ['Filename'] + [f"{label}_Conf" for label in label_names] + ['Predictions']
    
    # 显示结果
    for i, result in enumerate(results, 1):
        filename = os.path.basename(result['image_path'])
        
        if multi_label:
            probs_dict = result['probabilities']
            predictions_dict = result['predictions']
            
            # 获取按顺序的置信度列表
            confidences = [probs_dict[label] for label in label_names]
            
            # 确定预测的标签
            predicted_labels = [label for label, info in predictions_dict.items() if info['value'] == 1]
            
            # 构建CSV行
            csv_row = [filename] + [f"{conf:.4f}" for conf in confidences] + [";".join(predicted_labels)]
            csv_rows.append(csv_row)
            
            # 打印信息（带进度）
            print(f"[{i}/{len(results)}] {filename}")
            print(f"   预测: {', '.join(predicted_labels) if predicted_labels else '无'}")
            for label, conf in zip(label_names, confidences):
                print(f"      - {label}: {conf:.2%}")
            print()

        else:
            pass
    # 保存CSV
    if output_csv:
        csv_path = os.path.join(image_dir, output_csv)
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)
                writer.writerows(csv_rows)
            print(f"✓ 预测详情已保存到: {csv_path}")
        except Exception as e:
            print(f"保存CSV失败: {e}")

    print("="*80)
    print(f"\n总计: {len(results)} 张图片")
    print(f"✓ 预测详情已保存到: {csv_path}")
    print(f"\n⏱️  统计信息:")
    print(f"   总耗时: {total_time:.2f} 秒")
    print(f"   平均单图预测耗时: {total_time / len(results):.4f} 秒")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='批量预测QR码图片的类别 (支持多标签)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_predict.py test_images
  python batch_predict.py test_images --checkpoint checkpoints/best_model.pth --move-files
  python batch_predict.py test_images --use-contrastive
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
        default='checkpoints/best_model_pico.pth',
        help='模型检查点路径'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default='predictions_detail.csv',
        help='输出CSV文件名'
    )

    parser.add_argument(
        '--multi-label',
        action='store_true',
        default=True, 
        help='强制开启多标签模式 (默认开启)'
    )
    
    parser.add_argument(
        '--use-contrastive',
        action='store_true',
        default=True,
        help='使用对比学习模型'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        print(f"错误：目录不存在: {args.image_dir}")
        sys.exit(1)
    
    batch_predict(args.image_dir, args.checkpoint, args.csv, args.multi_label, args.use_contrastive)

