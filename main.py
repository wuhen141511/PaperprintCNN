"""
QR Code Classification System - Main Entry Point
Provides command-line interface for training and inference.
"""

import argparse
import os
import yaml

from src.train import train_model
from src.inference import predict_single_image, QRCodePredictor


def load_config(config_path: str = 'config.yaml'):
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def train_command(args):
    """Execute training command."""
    # Load config if available
    config = load_config(args.config) if args.config else {}
    
    # Override config with command-line arguments
    train_dir = args.train_dir or config.get('data', {}).get('train_dir', 'data/train')
    val_dir = args.val_dir or config.get('data', {}).get('val_dir', 'data/val')
    model_name = args.model or config.get('model', {}).get('name', 'resnet50')
    num_labels = args.num_labels or config.get('model', {}).get('num_labels', 3)
    batch_size = args.batch_size or config.get('data', {}).get('batch_size', 32)
    learning_rate = args.lr or config.get('training', {}).get('learning_rate', 0.001)
    num_epochs = args.epochs or config.get('training', {}).get('num_epochs', 20)
    image_size = args.image_size or config.get('data', {}).get('image_size', 224)
    
    multi_label = args.multi_label
    # If not explicitly set via CLI, check config or default to True
    if not args.multi_label_set: # Custom flag I'll handle internally or just rely on default
         multi_label = config.get('model', {}).get('multi_label', True)

    print("\n" + "="*60)
    print("QR Code Classification - Training Mode")
    print("="*60)
    print(f"Train directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Model: {model_name}")
    print(f"Number of labels: {num_labels} (Multi-label: {multi_label})")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Image size: {image_size}")
    print("="*60 + "\n")
    
    # Train model
    train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        model_name=model_name,
        num_labels=num_labels,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        image_size=image_size,
        freeze_backbone=True,
        checkpoint_dir='checkpoints',
        log_dir='logs',
        multi_label=multi_label
    )


def predict_command(args):
    """Execute prediction command."""
    print("\n" + "="*60)
    print("QR Code Classification - Prediction Mode")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print(f"Multi-label: {args.multi_label}")
    print("="*60 + "\n")
    
    # Predict
    result = predict_single_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        visualize=args.visualize,
        multi_label=args.multi_label
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='QR Code Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --train-dir data/train --val-dir data/val --epochs 20
  
  # Predict on a single image
  python main.py predict --image test.jpg --checkpoint checkpoints/best_model.pth
  
  # Use config file
  python main.py train --config config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--train-dir', type=str, help='Path to training data directory')
    train_parser.add_argument('--val-dir', type=str, help='Path to validation data directory')
    train_parser.add_argument('--model', type=str, default='resnet50',
                            choices=['resnet50', 'resnet18', 'efficientnet_b0', 'mobilenet_v3_small'],
                            help='Model architecture')
    train_parser.add_argument('--num-labels', type=int, help='Number of labels/classes')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--image-size', type=int, help='Input image size')
    train_parser.add_argument('--config', type=str, help='Path to config YAML file')
    
    # Multi-label flags
    train_parser.add_argument('--multi-label', action='store_true', default=True, help='Enable multi-label mode (default: True)')
    train_parser.add_argument('--single-label', action='store_false', dest='multi_label', help='Enable single-class mode')
    train_parser.set_defaults(multi_label_set=True) # Hack to detect if set, though logic above is simplified

    train_parser.set_defaults(func=train_command)
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Predict on an image')
    predict_parser.add_argument('--image', type=str, required=True, help='Path to image file')
    predict_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    predict_parser.add_argument('--model', type=str, default='resnet50',
                              choices=['resnet50', 'resnet18', 'efficientnet_b0', 'mobilenet_v3_small'],
                              help='Model architecture')
    predict_parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                              help='Disable visualization')
    predict_parser.add_argument('--multi-label', action='store_true', default=True, help='Use multi-label prediction (default: True)')
    predict_parser.add_argument('--single-label', action='store_false', dest='multi_label', help='Use single-class prediction')
    
    predict_parser.set_defaults(func=predict_command, visualize=True)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()

