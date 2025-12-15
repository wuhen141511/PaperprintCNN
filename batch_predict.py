"""
批量预测脚本 - 测试多张图片
"""

import os
import sys
import csv
import shutil
from src.inference import QRCodePredictor

def batch_predict(image_dir, checkpoint_path='checkpoints/best_model.pth', move_files=False, 
                  output_csv='predictions.csv', multi_label=True):
    """
    批量预测目录中的所有图片
    
    Args:
        image_dir: 包含测试图片的目录
        checkpoint_path: 模型检查点路径
        move_files: 是否将测试文件按预测结果移动到对应的子目录中（对于多标签，会复制到所有匹配的类目录）
        output_csv: 结果CSV文件路径
        multi_label: 是否启用多标签模式
    """
    # 创建预测器
    predictor = QRCodePredictor(
        checkpoint_path=checkpoint_path,
        model_name='convnextv2_tiny',
        backend='opencv',
        multi_label=multi_label
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
    
    # 批量预测
    results = predictor.predict_batch(image_files)
    
    # 准备CSV数据
    csv_rows = []
    csv_header = ['Filename'] + [f"{label}_Conf" for label in label_names] + ['Predictions']
    
    # 如果需要移动文件，创建输出目录
    if move_files:
        # 为每个标签创建目录
        for label in label_names:
            class_dir = os.path.join(image_dir, label)
            os.makedirs(class_dir, exist_ok=True)
        # 为无标签图片创建目录(可选)
        no_label_dir = os.path.join(image_dir, "uncategorized")
        os.makedirs(no_label_dir, exist_ok=True)
        print(f"\n已创建类别目录: {', '.join(label_names)} 和 uncategorized\n")
    
    # 显示结果并处理文件
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
            
            # 打印信息
            print(f"{i}. {filename}")
            print(f"   预测: {', '.join(predicted_labels) if predicted_labels else '无'}")
            for label, conf in zip(label_names, confidences):
                print(f"      - {label}: {conf:.2%}")
                
            # 处理文件移动/复制
            if move_files:
                # 构造新文件名: 原名_conf1_conf2_conf3.ext
                name_without_ext, ext = os.path.splitext(filename)
                conf_str = "_".join([f"{conf:.4f}" for conf in confidences])
                new_filename = f"{name_without_ext}_{conf_str}{ext}"
                
                # 如果有预测标签，复制到对应目录
                if predicted_labels:
                    for label in predicted_labels:
                        target_path = os.path.join(image_dir, label, new_filename)
                        shutil.copy2(result['image_path'], target_path)
                        print(f"   → 已复制到: {os.path.join(label, new_filename)}")
                else:
                    # 没有标签，移动到未分类
                    target_path = os.path.join(image_dir, "uncategorized", new_filename)
                    shutil.copy2(result['image_path'], target_path)
                    print(f"   → 已复制到: {os.path.join('uncategorized', new_filename)}")
                
                # 删除原文件(实现"移动"效果，但因为可能复制到多处，所以最后删除)
                try:
                    os.remove(result['image_path'])
                    print("   ✓ 原文件已删除")
                except OSError as e:
                    print(f"   ! 删除原文件失败: {e}")

        else:
            # 单标签逻辑保持兼容或可移除，这里保留基本兼容
            pass # 如果需要单标签支持，可以在这里保留，但用户请求针对多分类逻辑
        
        print()
    
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
    if move_files:
        print(f"✓ 文件处理完成")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='批量预测QR码图片的类别 (支持多标签)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_predict.py test_images
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
        help='模型检查点路径'
    )
    
    parser.add_argument(
        '--move-files',
        action='store_true',
        help='将文件按预测结果复制/移动到对应标签子目录，并重命名包含置信度'
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
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        print(f"错误：目录不存在: {args.image_dir}")
        sys.exit(1)
    
    batch_predict(args.image_dir, args.checkpoint, args.move_files, args.csv, args.multi_label)

