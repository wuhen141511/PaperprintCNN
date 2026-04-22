"""
Convenient script to export PyTorch models to ONNX format.
This script provides a simple command-line interface to convert trained models
to ONNX format for deployment in C++ and OpenCV applications.

Usage:
    # Export default model (checkpoints/best_model.pth)
    python export_to_onnx.py
    
    # Export specific model
    python export_to_onnx.py --model checkpoints/epoch_10.pth
    
    # Export with custom output path
    python export_to_onnx.py --model checkpoints/best_model.pth --output models/qrcode.onnx
    
    # Export with custom input size
    python export_to_onnx.py --input-size 224
    
    # Export with specific opset version
    python export_to_onnx.py --opset 13
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import export_to_onnx


def main():
    parser = argparse.ArgumentParser(
        description='Export PyTorch model to ONNX format for C++ and OpenCV deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export default model
  python export_to_onnx.py
  
  # Export specific model
  python export_to_onnx.py --model checkpoints/epoch_10.pth
  
  # Export with custom output path
  python export_to_onnx.py --model checkpoints/best_model.pth --output models/qrcode.onnx
  
  # Export with custom input size (224x224)
  python export_to_onnx.py --input-size 224
  
  # Export with specific opset version for newer OpenCV
  python export_to_onnx.py --opset 13
  
  # Export without dynamic batch size
  python export_to_onnx.py --no-dynamic
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to PyTorch model checkpoint (.pth file). Default: checkpoints/best_model.pth'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for ONNX model. Default: same directory as input model with .onnx extension'
    )
    
    parser.add_argument(
        '--input-size', '-s',
        type=int,
        default=384,
        help='Input image size (assumes square images). Default: 224'
    )

    parser.add_argument(
        '--input-channels', '-c',
        type=int,
        default=4,
        help='Number of input channels. Default: 4'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='Batch size for dummy input. Default: 1'
    )
    
    parser.add_argument(
        '--opset',
        type=int,
        default=14,
        help='ONNX opset version. Default: 11 (compatible with most OpenCV versions)'
    )
    
    parser.add_argument(
        '--no-dynamic',
        action='store_true',
        help='Disable dynamic batch size (use fixed batch size)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    parser.add_argument(
        '--version', '-v',
        type=str,
        default=None,
        help='Add version number to ONNX model metadata'
    )
    
    args = parser.parse_args()
    
    # Prepare arguments
    input_size = (args.input_channels, args.input_size, args.input_size)
    dynamic_axes = not args.no_dynamic
    verbose = not args.quiet
    
    try:
        # Export model
        onnx_path = export_to_onnx(
            model_path=args.model,
            output_path=args.output,
            input_size=input_size,
            batch_size=args.batch_size,
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            version=args.version
        )
        
        if not verbose:
            print(f"✓ ONNX model exported successfully: {onnx_path}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        print("\nMake sure you have trained a model first:", file=sys.stderr)
        print("  python main.py", file=sys.stderr)
        return 1
        
    except Exception as e:
        print(f"❌ Export failed: {e}", file=sys.stderr)
        import traceback
        if verbose:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
