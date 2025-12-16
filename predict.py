"""
Standalone prediction script.
Quick script for making predictions on QR code images.
"""

import sys
from src.inference import predict_single_image

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\nUsage: python predict.py <image_path> [checkpoint_path]")
        print("\nExample:")
        print("  python predict.py test_image.jpg")
        print("  python predict.py test_image.jpg checkpoints/best_model.pth")
        sys.exit(1)
    
    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/best_model.pth'
    
    print("\n" + "="*60)
    print("QR Code Classification - Prediction")
    print("="*60)
    
    result = predict_single_image(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        model_name='convnextv2_tiny',
        visualize=True
    )
    
    print("\n" + "="*60)
