"""
Example script to create sample data structure for testing.
This creates dummy directories to help you understand the required structure.
"""

import os

def create_sample_structure():
    """Create sample data directory structure."""
    
    # Define directory structure
    dirs = [
        'data/train/original',
        'data/train/copied',
        'data/val/original',
        'data/val/copied'
    ]
    
    print("Creating sample data directory structure...\n")
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    # Create README in data directory
    readme_content = """# Data Directory

Please organize your QR code images in this directory structure:

- train/original/  - Place original printed QR code images here
- train/copied/    - Place copied QR code images here
- val/original/    - Place validation original images here
- val/copied/      - Place validation copied images here

Supported formats: JPG, PNG, BMP, GIF

Recommended:
- At least 100 images per class for training
- At least 20 images per class for validation
- Image resolution: 224x224 or higher

See DATA_GUIDE.md for more details.
"""
    
    readme_path = 'data/README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✓ Created: {readme_path}")
    print("\n" + "="*60)
    print("Sample data structure created successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Add your QR code images to the appropriate directories")
    print("2. Run: .venv\\Scripts\\python.exe train_model.py")
    print("\nSee DATA_GUIDE.md for detailed instructions.")


if __name__ == '__main__':
    create_sample_structure()
