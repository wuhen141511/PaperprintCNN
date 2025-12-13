"""
Dataset splitting tool for multi-label classification project.
Splits training data into train and validation sets while maintaining annotations.

Supports two modes:
1. Single-class mode: Directory-based structure (class subdirectories)
2. Multi-label mode: JSON annotation-based structure (images/ + annotations.json)

Usage:
    # Multi-label mode (default)
    python split_dataset.py --train_dir data/train --val_dir data/val --ratio 0.2
    
    # Single-class mode
    python split_dataset.py --train_dir data/train --val_dir data/val --ratio 0.2 --mode single-class
"""

import argparse
import sys
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def split_multilabel_data(
    train_dir: str,
    val_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Split multi-label dataset with JSON annotations using Stratified Sampling.
    Uses Label Powerset Strategy: groups images by their unique label combinations
    and splits each group proportionally.
    
    Args:
        train_dir: Training directory containing images/ and annotations.json
        val_dir: Validation directory (will be created)
        val_ratio: Ratio of data to move to validation
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with split statistics
    """
    random.seed(seed)
    from collections import defaultdict
    
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    
    # Check structure
    train_images_dir = train_path / 'images'
    train_annotations_file = train_path / 'annotations.json'
    
    if not train_images_dir.exists():
        raise ValueError(f"Images directory not found: {train_images_dir}")
    if not train_annotations_file.exists():
        raise ValueError(f"Annotations file not found: {train_annotations_file}")
    
    # Load annotations
    with open(train_annotations_file, 'r', encoding='utf-8') as f:
        all_annotations = json.load(f)
    
    total_images = len(all_annotations)
    if total_images == 0:
        raise ValueError("No images found in annotations")
        
    # --- Stratified Sampling Logic ---
    if verbose:
        print(f"Found {total_images} images. Performing Stratified Sampling...")
        print(f"Target validation ratio: {val_ratio:.1%}")
    
    # 1. Group images by their label combinations
    label_groups = defaultdict(list)
    
    # Get sorted label names to ensure consistent signature
    first_ann = next(iter(all_annotations.values()))
    label_names = sorted(list(first_ann.keys()))
    
    for img_name, labels in all_annotations.items():
        # Create a signature tuple for the label combination (e.g., (1, 0, 1))
        # This represents a unique class in the Label Powerset transformation
        signature = tuple(labels[name] for name in label_names)
        label_groups[signature].append(img_name)
    
    val_images = []
    train_images = []
    
    if verbose:
        print(f"identified {len(label_groups)} unique label combinations:")
    
    # 2. Split each group proportionally
    for signature, group_images in sorted(label_groups.items()):
        # deterministic shuffle for this group
        random.shuffle(group_images)
        
        n_group = len(group_images)
        # Calculate validation size for this group (using rounding to nearest)
        n_val_group = int(n_group * val_ratio + 0.5)
        
        # Corner case: if ratio is small but group is small, n_val_group might be 0.
        # We generally prefer to keep samples in training if they are too few,
        # unless ratio is very high.
        
        # Ensure we don't accidentally empty the training set for rare combinations
        if n_val_group >= n_group and n_group > 1:
            n_val_group = n_group - 1
            
        val_subset = group_images[:n_val_group]
        train_subset = group_images[n_val_group:]
        
        val_images.extend(val_subset)
        train_images.extend(train_subset)
        
        if verbose:
            sig_str = ", ".join(f"{n[:4]}={v}" for n, v in zip(label_names, signature))
            print(f"  - [{sig_str}]: {n_group} images -> {len(val_subset)} val, {len(train_subset)} train")

    # 3. Final shuffle to mix the groups in the lists
    random.shuffle(val_images)
    random.shuffle(train_images)
    
    total_val = len(val_images)
    if verbose:
        print("-" * 60)
        print(f"Total to move: {total_val} images ({total_val/total_images:.1%})")
        print("-" * 60)
            
    # --- End Stratified Sampling ---
    
    # Create validation directory structure
    val_images_dir = val_path / 'images'
    val_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Split annotations and move files
    train_annotations = {}
    val_annotations = {}
    
    moved_count = 0
    # Process validation images
    for img_name in val_images:
        src_path = train_images_dir / img_name
        dst_path = val_images_dir / img_name
        
        if src_path.exists():
            # Handle duplicate filenames
            if dst_path.exists():
                base_name = Path(img_name).stem
                extension = Path(img_name).suffix
                counter = 1
                while dst_path.exists():
                    dst_path = val_images_dir / f"{base_name}_{counter}{extension}"
                    counter += 1
            
            shutil.move(str(src_path), str(dst_path))
            val_annotations[img_name] = all_annotations[img_name]
            moved_count += 1
        else:
            if verbose:
                print(f"Warning: Image not found: {img_name}")
    
    # Process training images (just keep annotations)
    for img_name in train_images:
        if img_name in all_annotations:
            train_annotations[img_name] = all_annotations[img_name]
    
    # Save split annotations
    train_annotations_file = train_path / 'annotations.json'
    val_annotations_file = val_path / 'annotations.json'
    
    with open(train_annotations_file, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=4, ensure_ascii=False)
    
    with open(val_annotations_file, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=4, ensure_ascii=False)
    
    if verbose:
        print(f"✓ Moved {moved_count} images to validation")
        print(f"✓ Training set: {len(train_annotations)} images")
        print(f"✓ Validation set: {len(val_annotations)} images")
        print(f"✓ Saved annotations to both directories")
    
    # Calculate label statistics
    if verbose and len(val_annotations) > 0:
        # Get label names from first annotation
        first_annotation = next(iter(all_annotations.values()))
        label_names = list(first_annotation.keys())
        
        print("\nLabel distribution in validation set:")
        for label_name in label_names:
            count = sum(1 for ann in val_annotations.values() if ann.get(label_name, 0) == 1)
            percentage = (count / len(val_annotations)) * 100
            print(f"  - {label_name}: {count}/{len(val_annotations)} ({percentage:.1f}%)")
    
    stats = {
        'total_images': total_images,
        'train_images': len(train_annotations),
        'val_images': len(val_annotations),
        'val_ratio_actual': len(val_annotations) / total_images if total_images > 0 else 0
    }
    
    return stats


def split_singleclass_data(
    train_dir: str,
    val_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Split single-class dataset with directory structure.
    Uses the existing split_train_val_data function from utils.
    """
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from utils import split_train_val_data
    
    return split_train_val_data(
        train_dir=train_dir,
        val_dir=val_dir,
        val_ratio=val_ratio,
        seed=seed,
        verbose=verbose
    )


def detect_dataset_mode(train_dir: str) -> str:
    """
    Auto-detect dataset mode based on directory structure.
    
    Returns:
        'multi-label' or 'single-class'
    """
    train_path = Path(train_dir)
    
    # Check for multi-label structure
    if (train_path / 'images').exists() and (train_path / 'annotations.json').exists():
        return 'multi-label'
    
    # Check for single-class structure (class subdirectories)
    subdirs = [d for d in train_path.iterdir() if d.is_dir()]
    if len(subdirs) > 0:
        # Check if subdirectories contain images
        for subdir in subdirs:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            has_images = any(f.suffix.lower() in image_extensions for f in subdir.iterdir() if f.is_file())
            if has_images:
                return 'single-class'
    
    # Default to multi-label
    return 'multi-label'


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train and validation sets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--train_dir',
        type=str,
        default='data/train',
        help='Path to training directory'
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
        '--mode',
        type=str,
        choices=['auto', 'multi-label', 'single-class'],
        default='auto',
        help='Dataset mode: auto-detect, multi-label (JSON), or single-class (directories)'
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
    
    # Convert to absolute paths
    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    
    if not train_dir.exists():
        print(f"Error: Training directory does not exist: {train_dir}")
        sys.exit(1)
    
    # Auto-detect mode if needed
    if args.mode == 'auto':
        mode = detect_dataset_mode(str(train_dir))
        if not args.quiet:
            print(f"Auto-detected mode: {mode}")
    else:
        mode = args.mode
    
    print("=" * 60)
    print("Dataset Splitting Tool")
    print("=" * 60)
    print(f"Mode: {mode}")
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
        # Perform the split based on mode
        if mode == 'multi-label':
            stats = split_multilabel_data(
                train_dir=str(train_dir),
                val_dir=str(val_dir),
                val_ratio=args.ratio,
                seed=args.seed,
                verbose=not args.quiet
            )
            
            if not args.quiet:
                print()
                print("=" * 60)
                print("Split Summary")
                print("=" * 60)
                print(f"Total images: {stats['total_images']}")
                print(f"Training set: {stats['train_images']}")
                print(f"Validation set: {stats['val_images']}")
                print(f"Actual validation ratio: {stats['val_ratio_actual']:.1%}")
        
        else:  # single-class
            stats = split_singleclass_data(
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
