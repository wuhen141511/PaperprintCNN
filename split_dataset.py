"""
Dataset splitting tool for ImageNet classification project.
Splits training data into train and validation sets while maintaining class structure.

Usage:
    python split_dataset.py --train_dir /data/train --val_dir /data/val --ratio 0.2
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import split_train_val_data


def main():
    parser = argparse.ArgumentParser(
        description='Split training dataset into train and validation sets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--train_dir',
        type=str,
        default='data/train',
        help='Path to training directory containing class subdirectories'
    )
    
    parser.add_argument(
        '--val_dir',
        type=str,
        default='data/val',
        help='Path to validation directory (will be created if doesn\'t exist)'
    )
    
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.2,
        help='Ratio of data to move to validation set (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 < args.ratio < 1.0):
        print(f"Error: --ratio must be between 0.0 and 1.0, got {args.ratio}")
        sys.exit(1)
    
    # Convert to absolute paths if relative
    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    
    if not train_dir.exists():
        print(f"Error: Training directory does not exist: {train_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("Dataset Splitting Tool")
    print("=" * 60)
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Validation ratio: {args.ratio:.1%}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    print()
    
    # Confirm before proceeding
    response = input("⚠️  This will MOVE files from train to val directory. Continue? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Operation cancelled.")
        sys.exit(0)
    
    print()
    
    try:
        # Perform the split
        stats = split_train_val_data(
            train_dir=str(train_dir),
            val_dir=str(val_dir),
            val_ratio=args.ratio,
            seed=args.seed,
            verbose=not args.quiet
        )
        
        # Print detailed statistics
        if not args.quiet:
            print()
            print("=" * 60)
            print("Detailed Statistics by Class")
            print("=" * 60)
            for class_name, class_stats in sorted(stats['classes'].items()):
                print(f"{class_name}:")
                print(f"  Total: {class_stats['total']}")
                print(f"  Moved to validation: {class_stats['moved_to_val']}")
                print(f"  Remaining in training: {class_stats['remaining_in_train']}")
                print(f"  Actual validation ratio: {class_stats['actual_val_ratio']:.1%}")
                print()
        
        print("=" * 60)
        print("✅ Dataset split completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
