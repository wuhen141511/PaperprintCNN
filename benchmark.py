"""
性能测试脚本 - 统计预测耗时
测试模型推理速度和各个环节的时间消耗
"""

import os
import sys
import time
import torch
from PIL import Image
import numpy as np

from src.inference import QRCodePredictor
from src.utils import get_device


def benchmark_prediction(image_path, checkpoint_path='checkpoints/best_model.pth', num_runs=10):
    """
    测试预测性能
    
    Args:
        image_path: 测试图片路径
        checkpoint_path: 模型检查点路径
        num_runs: 运行次数（用于计算平均值）
    """
    print("="*80)
    print("QR Code Classification - 性能测试")
    print("="*80)
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片不存在: {image_path}")
        return
    
    # 设备信息
    device = get_device()
    print(f"\n设备信息:")
    print(f"  设备类型: {device}")
    if torch.cuda.is_available():
        print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")
    print()
    
    # 1. 测试模型加载时间
    print("1. 模型加载测试...")
    start_time = time.time()
    # 默认开启多标签
    predictor = QRCodePredictor(
        checkpoint_path=checkpoint_path,
        model_name='resnet50',
        multi_label=True
    )
    load_time = time.time() - start_time
    print(f"   模型加载时间: {load_time:.3f} 秒")
    print()
    
    # 2. 测试图片加载和预处理时间
    print("2. 图片预处理测试...")
    start_time = time.time()
    image = Image.open(image_path).convert('RGB')
    image_tensor = predictor.transform(image).unsqueeze(0).to(predictor.device)
    preprocess_time = time.time() - start_time
    print(f"   图片加载和预处理时间: {preprocess_time*1000:.2f} 毫秒")
    print()
    
    # 3. 测试首次推理时间（包含预热）
    print("3. 首次推理测试（包含预热）...")
    start_time = time.time()
    with torch.no_grad():
        outputs = predictor.model(image_tensor)
    first_inference_time = time.time() - start_time
    print(f"   首次推理时间: {first_inference_time*1000:.2f} 毫秒")
    print()
    
    # 4. 测试多次推理的平均时间
    print(f"4. 推理性能测试（{num_runs} 次运行）...")
    inference_times = []
    
    for i in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            outputs = predictor.model(image_tensor)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
    
    avg_inference_time = np.mean(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    std_inference_time = np.std(inference_times)
    
    print(f"   平均推理时间: {avg_inference_time*1000:.2f} 毫秒")
    print(f"   最快推理时间: {min_inference_time*1000:.2f} 毫秒")
    print(f"   最慢推理时间: {max_inference_time*1000:.2f} 毫秒")
    print(f"   标准差: {std_inference_time*1000:.2f} 毫秒")
    print()
    
    # 5. 测试完整预测流程（端到端）
    print(f"5. 端到端预测测试（{num_runs} 次运行）...")
    end_to_end_times = []
    
    for i in range(num_runs):
        start_time = time.time()
        result = predictor.predict_image(image_path, return_probs=True)
        end_to_end_time = time.time() - start_time
        end_to_end_times.append(end_to_end_time)
    
    avg_e2e_time = np.mean(end_to_end_times)
    min_e2e_time = np.min(end_to_end_times)
    max_e2e_time = np.max(end_to_end_times)
    
    print(f"   平均端到端时间: {avg_e2e_time*1000:.2f} 毫秒")
    print(f"   最快端到端时间: {min_e2e_time*1000:.2f} 毫秒")
    print(f"   最慢端到端时间: {max_e2e_time*1000:.2f} 毫秒")
    print()
    
    # 6. 显示预测结果
    print("6. 预测结果:")
    print(f"   图片: {os.path.basename(image_path)}")
    
    if predictor.multi_label:
        predictions_dict = result['predictions']
        print(f"   预测详情:")
        for label, info in predictions_dict.items():
            val = info['value']
            prob = info['probability']
            status = "YES" if val == 1 else "NO "
            print(f"      - [{status}] {label}: {prob:.2%}")
    else:
        print(f"   预测类别: {result['predicted_label']}")
        print(f"   置信度: {result['confidence']:.2%}")
        print(f"   类别概率:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"      - {class_name}: {prob:.2%}")
    print()
    
    # 7. 性能总结
    print("="*80)
    print("性能总结")
    print("="*80)
    print(f"模型加载时间:        {load_time:.3f} 秒")
    print(f"图片预处理时间:      {preprocess_time*1000:.2f} 毫秒")
    print(f"平均推理时间:        {avg_inference_time*1000:.2f} 毫秒")
    print(f"平均端到端时间:      {avg_e2e_time*1000:.2f} 毫秒")
    print(f"理论最大吞吐量:      {1/avg_e2e_time:.1f} 张/秒")
    print("="*80)
    
    # 8. 性能建议
    print("\n💡 性能建议:")
    if torch.cuda.is_available():
        print("  ✓ 已使用 GPU 加速")
    else:
        print("  ⚠ 当前使用 CPU，建议使用 GPU 可提升 5-10 倍速度")
    
    if avg_e2e_time < 0.1:
        print("  ✓ 推理速度优秀 (< 100ms)")
    elif avg_e2e_time < 0.5:
        print("  ✓ 推理速度良好 (< 500ms)")
    else:
        print("  ⚠ 推理速度较慢，建议使用更小的模型或 GPU 加速")
    
    print()
    
    return {
        'load_time': load_time,
        'preprocess_time': preprocess_time,
        'avg_inference_time': avg_inference_time,
        'avg_e2e_time': avg_e2e_time,
        'throughput': 1/avg_e2e_time
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n用法: python benchmark.py <图片路径> [运行次数]")
        print("\n示例:")
        print("  python benchmark.py test.jpg")
        print("  python benchmark.py test.jpg 20")
        print("\n说明:")
        print("  运行次数默认为 10，用于计算平均性能")
        sys.exit(1)
    
    image_path = sys.argv[1]
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    benchmark_prediction(image_path, num_runs=num_runs)
