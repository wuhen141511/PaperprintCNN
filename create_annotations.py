"""
Helper script to create annotation files for multi-label classification.
Creates a template annotations.json file that users can fill in.
"""

import json
import os
from pathlib import Path
from typing import List, Dict


def create_annotation_template(
    image_dir: str,
    output_file: str = None,
    label_names: List[str] = None
) -> str:
    """
    Create an annotation template file for all images in a directory.
    
    Args:
        image_dir: Directory containing images
        output_file: Path to save annotations.json (default: image_dir/annotations.json)
        label_names: List of label names (default: ["is_copied", "is_blurry", "is_low_light"])
        
    Returns:
        Path to the created annotation file
    """
    if label_names is None:
        label_names = ["is_copied", "is_blurry", "is_low_light"]
    
    if output_file is None:
        output_file = os.path.join(image_dir, 'annotations.json')
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_dir_path = Path(image_dir)
    
    # Check if images are in a subdirectory called 'images'
    if (image_dir_path / 'images').exists():
        search_dir = image_dir_path / 'images'
        use_images_subdir = True
    else:
        search_dir = image_dir_path
        use_images_subdir = False
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(search_dir.glob(f'*{ext}'))
        image_files.extend(search_dir.glob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {search_dir}")
        return None
    
    # Create annotations dictionary
    annotations = {}
    for img_path in sorted(image_files):
        # Use relative path if images are in subdirectory
        if use_images_subdir:
            img_name = img_path.name
        else:
            img_name = img_path.name
        
        # Initialize all labels to 0 (user needs to fill in correct values)
        annotations[img_name] = {label: 0 for label in label_names}
    
    # Save to file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Created annotation template: {output_file}")
    print(f"  Found {len(image_files)} images")
    print(f"  Labels: {label_names}")
    print(f"\nNext steps:")
    print(f"  1. Open {output_file}")
    print(f"  2. Set label values to 0 or 1 for each image")
    print(f"  3. Save the file")
    
    return output_file


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
