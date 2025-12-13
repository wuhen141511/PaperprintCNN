"""
Test script for split_dataset.py
Creates sample data and tests the splitting functionality.
"""

import os
import json
import shutil
from pathlib import Path


def create_test_multilabel_data():
    """Create test data for multi-label mode"""
    print("Creating test data for multi-label mode...")
    
    # Create directory structure
    test_dir = Path('test_data/train')
    images_dir = test_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy image files
    annotations = {}
    for i in range(1, 21):  # 20 test images
        img_name = f"test_{i:03d}.jpg"
        img_path = images_dir / img_name
        
        # Create empty file (dummy image)
        img_path.touch()
        
        # Create random annotations
        import random
        annotations[img_name] = {
            "is_copied": random.choice([0, 1]),
            "is_blurry": random.choice([0, 1]),
            "is_low_light": random.choice([0, 1])
        }
    
    # Save annotations
    ann_file = test_dir / 'annotations.json'
    with open(ann_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Created {len(annotations)} test images in {images_dir}")
    print(f"✓ Created annotations file: {ann_file}")
    
    return str(test_dir)


def create_test_singleclass_data():
    """Create test data for single-class mode"""
    print("\nCreating test data for single-class mode...")
    
    # Create directory structure
    test_dir = Path('test_data_single/train')
    
    # Create two classes
    for class_name in ['class_a', 'class_b']:
        class_dir = test_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy image files
        for i in range(1, 11):  # 10 images per class
            img_path = class_dir / f"{class_name}_{i:03d}.jpg"
            img_path.touch()
    
    print(f"✓ Created test data in {test_dir}")
    print(f"  - class_a: 10 images")
    print(f"  - class_b: 10 images")
    
    return str(test_dir)


def test_multilabel_split():
    """Test multi-label splitting"""
    print("\n" + "=" * 60)
    print("Testing Multi-Label Split")
    print("=" * 60)
    
    train_dir = create_test_multilabel_data()
    val_dir = 'test_data/val'
    
    # Import and run split
    from split_dataset import split_multilabel_data
    
    stats = split_multilabel_data(
        train_dir=train_dir,
        val_dir=val_dir,
        val_ratio=0.3,
        seed=42,
        verbose=True
    )
    
    print("\n✅ Multi-label split test completed!")
    print(f"   Training: {stats['train_images']} images")
    print(f"   Validation: {stats['val_images']} images")
    
    # Verify files
    train_images = list(Path(train_dir).glob('images/*.jpg'))
    val_images = list(Path(val_dir).glob('images/*.jpg'))
    
    print(f"\nVerification:")
    print(f"   Train images on disk: {len(train_images)}")
    print(f"   Val images on disk: {len(val_images)}")
    
    # Check annotations
    with open(Path(train_dir) / 'annotations.json', 'r') as f:
        train_ann = json.load(f)
    with open(Path(val_dir) / 'annotations.json', 'r') as f:
        val_ann = json.load(f)
    
    print(f"   Train annotations: {len(train_ann)}")
    print(f"   Val annotations: {len(val_ann)}")
    
    # Cleanup
    cleanup = input("\nCleanup test data? (y/n): ")
    if cleanup.lower() == 'y':
        shutil.rmtree('test_data')
        print("✓ Test data cleaned up")


def test_singleclass_split():
    """Test single-class splitting"""
    print("\n" + "=" * 60)
    print("Testing Single-Class Split")
    print("=" * 60)
    
    train_dir = create_test_singleclass_data()
    val_dir = 'test_data_single/val'
    
    # Import and run split
    from split_dataset import split_singleclass_data
    
    stats = split_singleclass_data(
        train_dir=train_dir,
        val_dir=val_dir,
        val_ratio=0.3,
        seed=42,
        verbose=True
    )
    
    print("\n✅ Single-class split test completed!")
    
    # Cleanup
    cleanup = input("\nCleanup test data? (y/n): ")
    if cleanup.lower() == 'y':
        shutil.rmtree('test_data_single')
        print("✓ Test data cleaned up")


def main():
    print("=" * 60)
    print("Split Dataset Test Suite")
    print("=" * 60)
    print("\nThis script will create test data and verify the splitting functionality.")
    print("\nSelect test mode:")
    print("  1. Multi-label mode")
    print("  2. Single-class mode")
    print("  3. Both modes")
    
    choice = input("\nEnter choice (1/2/3): ")
    
    if choice == '1':
        test_multilabel_split()
    elif choice == '2':
        test_singleclass_split()
    elif choice == '3':
        test_multilabel_split()
        test_singleclass_split()
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
