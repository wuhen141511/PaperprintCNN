"""
Quick test script to verify the installation and setup.
Tests model creation and basic functionality without requiring data.
"""

import torch
import sys

def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import PIL
        import numpy
        import matplotlib
        import tqdm
        import yaml
        print("✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("⚠ CUDA not available - will use CPU (slower)")
    return True


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    try:
        from src.model import create_model
        from src.utils import get_device
        
        device = get_device()
        model = create_model(
            num_classes=2,
            model_name='resnet18',  # Use smaller model for testing
            pretrained=True,
            freeze_backbone=True,
            device=device
        )
        print("✓ Model created successfully")
        
        # Test forward pass
        print("\nTesting forward pass...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful - output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transforms():
    """Test image transforms."""
    print("\nTesting image transforms...")
    try:
        from src.dataset import get_transforms
        from PIL import Image
        import numpy as np
        
        # Create dummy image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test training transforms
        train_transform = get_transforms(image_size=224, augment=True)
        transformed = train_transform(dummy_img)
        print(f"✓ Training transform successful - output shape: {transformed.shape}")
        
        # Test validation transforms
        val_transform = get_transforms(image_size=224, augment=False)
        transformed = val_transform(dummy_img)
        print(f"✓ Validation transform successful - output shape: {transformed.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("QR Code Classification System - Installation Test")
    print("="*60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Package imports", test_imports()))
    results.append(("CUDA availability", test_cuda()))
    results.append(("Model creation", test_model_creation()))
    results.append(("Image transforms", test_transforms()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your data (see DATA_GUIDE.md)")
        print("2. Run: .venv\\Scripts\\python.exe create_data_structure.py")
        print("3. Add your images to the data directories")
        print("4. Run: .venv\\Scripts\\python.exe train_model.py")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
