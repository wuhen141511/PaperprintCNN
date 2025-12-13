"""
Example script demonstrating multi-label classification workflow.
This script shows how to:
1. Create annotation files
2. Train a multi-label model
3. Make predictions
"""

import os
import json
from pathlib import Path


def example_create_annotations():
    """Example: Create annotation files"""
    print("=" * 60)
    print("Step 1: Creating Annotation Files")
    print("=" * 60)
    
    # Example annotations for training data
    train_annotations = {
        "qr_001.jpg": {
            "is_copied": 0,  # Original
            "is_blurry": 0,  # Clear
            "is_low_light": 0  # Normal light
        },
        "qr_002.jpg": {
            "is_copied": 1,  # Copied
            "is_blurry": 0,  # Clear
            "is_low_light": 0  # Normal light
        },
        "qr_003.jpg": {
            "is_copied": 0,  # Original
            "is_blurry": 1,  # Blurry
            "is_low_light": 0  # Normal light
        },
        "qr_004.jpg": {
            "is_copied": 0,  # Original
            "is_blurry": 0,  # Clear
            "is_low_light": 1  # Low light
        },
        "qr_005.jpg": {
            "is_copied": 1,  # Copied
            "is_blurry": 1,  # Blurry
            "is_low_light": 0  # Normal light
        },
        "qr_006.jpg": {
            "is_copied": 1,  # Copied
            "is_blurry": 0,  # Clear
            "is_low_light": 1  # Low light
        },
        "qr_007.jpg": {
            "is_copied": 1,  # Copied
            "is_blurry": 1,  # Blurry
            "is_low_light": 1  # Low light (all three attributes!)
        }
    }
    
    # Create directories
    os.makedirs('data/train/images', exist_ok=True)
    os.makedirs('data/val/images', exist_ok=True)
    
    # Save training annotations
    with open('data/train/annotations.json', 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Created data/train/annotations.json with {len(train_annotations)} images")
    print("\nExample annotation:")
    print(json.dumps({"qr_007.jpg": train_annotations["qr_007.jpg"]}, indent=2))
    print("\nNote: This is just an example. You need to:")
    print("  1. Add your actual images to data/train/images/")
    print("  2. Update annotations.json with correct labels for your images")
    print("  3. Create data/val/annotations.json for validation data")


def example_train_model():
    """Example: Train a multi-label model"""
    print("\n" + "=" * 60)
    print("Step 2: Training Multi-Label Model")
    print("=" * 60)
    
    print("\nTraining code:")
    print("""
from src.train import train_model

history = train_model(
    train_dir='data/train',
    val_dir='data/val',
    model_name='resnet50',
    num_labels=3,
    batch_size=32,
    learning_rate=0.001,
    num_epochs=20,
    freeze_backbone=True,
    multi_label=True,
    label_names=["is_copied", "is_blurry", "is_low_light"]
)
""")
    
    print("\nNote: Make sure you have:")
    print("  ✓ Created annotations.json for both train and val directories")
    print("  ✓ Added images to data/train/images/ and data/val/images/")
    print("  ✓ Verified annotations with: python create_annotations.py validate data/train/annotations.json")


def example_inference():
    """Example: Make predictions with trained model"""
    print("\n" + "=" * 60)
    print("Step 3: Making Predictions")
    print("=" * 60)
    
    print("\nInference code:")
    print("""
from src.inference import QRCodePredictor

# Initialize predictor
predictor = QRCodePredictor(
    checkpoint_path='checkpoints/best_model.pth',
    model_name='resnet50',
    multi_label=True,
    label_names=["is_copied", "is_blurry", "is_low_light"],
    threshold=0.5
)

# Predict single image
result = predictor.predict_image('test_image.jpg')

# Print results
print("\\nPrediction Results:")
print(f"Image: {result['image_path']}")
print("\\nAttributes:")
for label_name, pred in result['predictions'].items():
    status = "✓ Yes" if pred['value'] == 1 else "✗ No"
    print(f"  {label_name}: {status} (confidence: {pred['probability']:.1%})")
""")
    
    print("\nExample output:")
    print("""
Prediction Results:
Image: test_image.jpg

Attributes:
  is_copied: ✓ Yes (confidence: 92.3%)
  is_blurry: ✗ No (confidence: 15.7%)
  is_low_light: ✓ Yes (confidence: 78.9%)
""")


def example_batch_inference():
    """Example: Batch prediction"""
    print("\n" + "=" * 60)
    print("Step 4: Batch Predictions")
    print("=" * 60)
    
    print("\nBatch inference code:")
    print("""
from src.inference import QRCodePredictor
import os

# Initialize predictor
predictor = QRCodePredictor(
    checkpoint_path='checkpoints/best_model.pth',
    multi_label=True
)

# Get all images in a directory
image_dir = 'test_images'
image_paths = [
    os.path.join(image_dir, f) 
    for f in os.listdir(image_dir) 
    if f.endswith(('.jpg', '.png'))
]

# Predict all images
results = predictor.predict_batch(image_paths)

# Save results to CSV
import csv

with open('predictions.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'is_copied', 'is_blurry', 'is_low_light'])
    
    for result in results:
        img_name = os.path.basename(result['image_path'])
        row = [img_name]
        for label in ['is_copied', 'is_blurry', 'is_low_light']:
            row.append(result['predictions'][label]['value'])
        writer.writerow(row)

print(f"Saved predictions for {len(results)} images to predictions.csv")
""")


def show_complete_workflow():
    """Show the complete workflow"""
    print("\n" + "=" * 60)
    print("COMPLETE MULTI-LABEL CLASSIFICATION WORKFLOW")
    print("=" * 60)
    
    example_create_annotations()
    example_train_model()
    example_inference()
    example_batch_inference()
    
    print("\n" + "=" * 60)
    print("Additional Resources")
    print("=" * 60)
    print("\n📖 Documentation:")
    print("  - MULTILABEL_GUIDE.md: Comprehensive guide")
    print("  - README.md: Project overview")
    print("\n🛠️ Helper Scripts:")
    print("  - create_annotations.py: Create and validate annotations")
    print("  - train_model.py: Quick training script")
    print("\n📊 Monitoring:")
    print("  - TensorBoard: tensorboard --logdir=logs")
    print("\n💡 Tips:")
    print("  1. Start with a small dataset to verify everything works")
    print("  2. Use data augmentation to improve generalization")
    print("  3. Monitor training with TensorBoard")
    print("  4. Adjust threshold (default 0.5) based on your needs")
    print("  5. Consider class imbalance when evaluating results")


if __name__ == '__main__':
    show_complete_workflow()
