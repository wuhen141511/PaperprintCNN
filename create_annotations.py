"""
Helper script to create annotation files for multi-label classification.
Creates a template annotations.json file that users can fill in.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def parse_filename_labels(filename: str, label_names: List[str]) -> Tuple[bool, Dict[str, int]]:
    """
    Parse labels from filename pattern _XYZ where X, Y, Z are 0 or 1.

    Args:
        filename: Image filename (e.g., "image_101.jpg")
        label_names: List of label names (should have 3 elements: [is_copied, is_low_light, is_blurry])

    Returns:
        Tuple of (is_valid, labels_dict)
        - is_valid: True if filename matches pattern, False otherwise
        - labels_dict: Dictionary mapping label names to values (0 or 1)
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]

    # Pattern: ends with _XXX where X is 0 or 1
    pattern = r'_([01])([01])([01])$'
    match = re.search(pattern, name_without_ext)

    if not match:
        return False, {}

    # Extract the three digits
    # Position mapping: is_copied, is_low_light, is_blurry
    values = [int(match.group(1)), int(match.group(2)), int(match.group(3))]

    # Create labels dictionary
    labels = {}
    for i, label_name in enumerate(label_names):
        if i < len(values):
            labels[label_name] = values[i]
        else:
            labels[label_name] = 0

    return True, labels


def create_annotation_template(
    image_dir: str,
    output_file: str = None,
    label_names: List[str] = None
) -> str:
    """
    Create an annotation template file for all images in a directory.
    Automatically sets labels based on filename pattern _XYZ where X,Y,Z are 0 or 1.
    Invalid files (without pattern) are moved to invalid/ subdirectory.

    Args:
        image_dir: Directory containing images
        output_file: Path to save annotations.json (default: image_dir/annotations.json)
        label_names: List of label names (default: ["is_copied", "is_low_light", "is_blurry"])

    Returns:
        Path to the created annotation file
    """
    if label_names is None:
        label_names = ["is_copied", "is_low_light", "is_blurry"]

    if output_file is None:
        output_file = os.path.join(image_dir, 'annotations.json')

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_dir_path = Path(image_dir).resolve()  # Convert to absolute path

    # Check if images are in a subdirectory called 'images'
    if (image_dir_path / 'images').exists():
        search_dir = image_dir_path / 'images'
        use_images_subdir = True
    else:
        search_dir = image_dir_path
        use_images_subdir = False

    # Create invalid directory path
    invalid_dir = search_dir / 'invalid'

    # Collect image files and remove duplicates
    image_files = set()  # Use set to automatically handle duplicates
    for ext in image_extensions:
        # glob() on Windows may return same file for both .jpg and .JPG
        image_files.update(search_dir.glob(f'*{ext}'))
        image_files.update(search_dir.glob(f'*{ext.upper()}'))

    # Convert set back to list
    image_files = list(image_files)

    if len(image_files) == 0:
        print(f"Warning: No images found in {search_dir}")
        return None

    # Statistics
    stats = {
        'total_files': len(image_files),
        'valid_files': 0,
        'invalid_files': 0,
        'label_counts': {label: 0 for label in label_names}
    }

    # Create invalid directory if needed
    invalid_dir = search_dir / 'invalid'

    # Create annotations dictionary
    annotations = {}
    invalid_files = []

    for img_path in sorted(image_files):
        img_name = img_path.name

        # Parse filename to get labels
        is_valid, labels = parse_filename_labels(img_name, label_names)

        if is_valid:
            # Valid filename - add to annotations
            annotations[img_name] = labels
            stats['valid_files'] += 1

            # Update label statistics
            for label_name, value in labels.items():
                if value == 1:
                    stats['label_counts'][label_name] += 1
        else:
            # Invalid filename - mark for moving
            invalid_files.append(img_path)
            stats['invalid_files'] += 1

    # Move invalid files to invalid directory
    if invalid_files:
        os.makedirs(invalid_dir, exist_ok=True)
        print(f"\n⚠️  Moving {len(invalid_files)} invalid files to {invalid_dir}:")
        for img_path in invalid_files:
            # Check if source file still exists (may have been moved in a previous run)
            if not img_path.exists():
                print(f"  ✗ Skipped {img_path.name}: file not found (may have been moved already)")
                stats['invalid_files'] -= 1
                continue

            dest_path = invalid_dir / img_path.name
            # Handle duplicate filenames
            if dest_path.exists():
                base_name = img_path.stem
                ext = img_path.suffix
                counter = 1
                while (invalid_dir / f"{base_name}_{counter}{ext}").exists():
                    counter += 1
                dest_path = invalid_dir / f"{base_name}_{counter}{ext}"

            try:
                shutil.move(str(img_path), str(dest_path))
                print(f"  - {img_path.name} -> {dest_path.name}")
            except Exception as e:
                print(f"  ✗ Failed to move {img_path.name}: {e}")
                stats['invalid_files'] -= 1  # Adjust count if move failed

    # Save to file
    if annotations:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=4, ensure_ascii=False)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Annotation Creation Summary")
    print(f"{'='*60}")
    print(f"Total files processed:     {stats['total_files']}")
    print(f"Valid files:               {stats['valid_files']}")
    print(f"Invalid files (moved):     {stats['invalid_files']}")

    if stats['valid_files'] > 0:
        print(f"\nLabel Statistics:")
        for label_name, count in stats['label_counts'].items():
            percentage = (count / stats['valid_files']) * 100
            print(f"  - {label_name:15s}: {count:4d} ({percentage:5.1f}%)")

        print(f"\n✓ Created annotation file: {output_file}")
        print(f"  Total annotations: {len(annotations)}")
    else:
        print(f"\n⚠️  No valid files found. Annotation file not created.")

    if stats['invalid_files'] > 0:
        print(f"\n⚠️  Invalid files moved to: {invalid_dir}")

    print(f"{'='*60}\n")

    return output_file if annotations else None


def validate_annotations(annotation_file: str, label_names: List[str] = None) -> bool:
    """
    Validate an annotation file.
    
    Args:
        annotation_file: Path to annotations.json
        label_names: Expected label names
        
    Returns:
        True if valid, False otherwise
    """
    if label_names is None:
        label_names = ["is_copied", "is_blurry", "is_low_light"]
    
    if not os.path.exists(annotation_file):
        print(f"❌ Error: Annotation file not found: {annotation_file}")
        return False
    
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {e}")
        return False
    
    if not isinstance(annotations, dict):
        print(f"❌ Error: Annotations must be a dictionary")
        return False
    
    errors = []
    warnings = []
    
    for img_name, labels in annotations.items():
        if not isinstance(labels, dict):
            errors.append(f"  - {img_name}: labels must be a dictionary")
            continue
        
        # Check if all required labels are present
        for label_name in label_names:
            if label_name not in labels:
                errors.append(f"  - {img_name}: missing label '{label_name}'")
        
        # Check if all label values are 0 or 1
        for label_name, value in labels.items():
            if value not in [0, 1]:
                errors.append(f"  - {img_name}: label '{label_name}' must be 0 or 1, got {value}")
        
        # Check if all labels are 0 (might be unfilled template)
        if all(v == 0 for v in labels.values()):
            warnings.append(f"  - {img_name}: all labels are 0 (is this intentional?)")
    
    # Print results
    if errors:
        print(f"❌ Validation failed with {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(error)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    
    if warnings:
        print(f"⚠️  Validation passed with {len(warnings)} warnings:")
        for warning in warnings[:5]:  # Show first 5 warnings
            print(warning)
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more warnings")
    
    print(f"✓ Validation passed!")
    print(f"  Total images: {len(annotations)}")
    
    # Calculate label statistics
    label_counts = {label: 0 for label in label_names}
    for labels in annotations.values():
        for label_name in label_names:
            if labels.get(label_name, 0) == 1:
                label_counts[label_name] += 1
    
    print(f"\nLabel statistics:")
    for label_name, count in label_counts.items():
        percentage = (count / len(annotations)) * 100
        print(f"  - {label_name}: {count} ({percentage:.1f}%)")
    
    return True


def merge_annotations(
    annotation_files: List[str],
    output_file: str
) -> str:
    """
    Merge multiple annotation files into one.
    
    Args:
        annotation_files: List of annotation file paths
        output_file: Path to save merged annotations
        
    Returns:
        Path to merged annotation file
    """
    merged = {}
    
    for ann_file in annotation_files:
        if not os.path.exists(ann_file):
            print(f"Warning: File not found: {ann_file}, skipping...")
            continue
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Check for duplicates
        for img_name in annotations:
            if img_name in merged:
                print(f"Warning: Duplicate image '{img_name}' in {ann_file}, overwriting...")
        
        merged.update(annotations)
    
    # Save merged annotations
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Merged {len(annotation_files)} annotation files")
    print(f"  Total images: {len(merged)}")
    print(f"  Output: {output_file}")
    
    return output_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create and manage annotation files for multi-label classification')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create template command
    create_parser = subparsers.add_parser('create', help='Create annotation template')
    create_parser.add_argument('image_dir', help='Directory containing images')
    create_parser.add_argument('--output', '-o', help='Output annotation file path')
    create_parser.add_argument('--labels', '-l', nargs='+', help='Label names')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate annotation file')
    validate_parser.add_argument('annotation_file', help='Path to annotations.json')
    validate_parser.add_argument('--labels', '-l', nargs='+', help='Expected label names')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple annotation files')
    merge_parser.add_argument('annotation_files', nargs='+', help='Annotation files to merge')
    merge_parser.add_argument('--output', '-o', required=True, help='Output merged annotation file')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        create_annotation_template(
            args.image_dir,
            args.output,
            args.labels
        )
    elif args.command == 'validate':
        validate_annotations(
            args.annotation_file,
            args.labels
        )
    elif args.command == 'merge':
        merge_annotations(
            args.annotation_files,
            args.output
        )
    else:
        parser.print_help()
