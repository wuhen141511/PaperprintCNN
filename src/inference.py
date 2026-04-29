"""
Inference script for QR code classification.
Provides functions for single image and batch prediction.
Supports both single-class and multi-label classification.
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
import json
from typing import List, Tuple, Dict
import numpy as np

from src.model import load_model_for_inference
from src.dataset import get_inference_transform
from src.utils import get_device
import cv2
try:
    from src.qrcode_utils import QRCodeRegistrator
except ImportError:
    # Handle case where package is run from different root
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.qrcode_utils import QRCodeRegistrator


class QRCodePredictor:
    """Predictor for QR code classification (single-class or multi-label)."""
    
    def __init__(
        self,
        checkpoint_path: str,
        label_names: List[str] = None,
        model_name: str = 'convnextv2_tiny',
        image_size: int = 384,
        device: str = None,
        backend: str = 'opencv',
        multi_label: bool = False,
        threshold: float = 0.5,
        register_dir: str = 'data/register',
        wqmodules_dir: str = 'wqmodules',
        use_contrastive: bool = False,
        use_gray: bool = False,
    ):
        """
        Initialize predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            label_names: List of label/class names (if None, will try to load from checkpoint dir)
            model_name: Name of the model architecture
            image_size: Input image size
            device: Device to use (None for auto-detection)
            backend: Preprocessing backend ('pil' or 'opencv')
            multi_label: Whether this is multi-label classification
            threshold: Threshold for multi-label binary predictions (default: 0.5)
            register_dir: Directory containing reference images
            use_contrastive: Whether to use contrastive learning model
        """
        self.device = get_device() if device is None else torch.device(device)
        self.image_size = image_size
        self.multi_label = multi_label
        self.threshold = threshold
        self.register_dir = register_dir
        self.wqmodules_dir = wqmodules_dir
        self.use_contrastive = use_contrastive
        
        # Initialize registrator
        self.registrator = QRCodeRegistrator(wqmodules_dir)
        
        # Use the requested backend
        self.transform = get_inference_transform(image_size, backend=backend, use_contrastive=use_contrastive, use_gray=use_gray)
        self.backend = backend
        
        # Load label/class names
        if label_names is None:
            class_names_path = os.path.join(os.path.dirname(checkpoint_path), 'class_names.json')
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    label_names = json.load(f)
            else:
                if multi_label:
                    label_names = ["is_copied", "is_low_light", "is_blurry"]  # Default for multi-label
                else:
                    label_names = ['Class 0', 'Class 1']  # Default for single-class
        
        self.label_names = label_names
        num_labels = len(label_names)
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = load_model_for_inference(
            checkpoint_path=checkpoint_path,
            num_labels=num_labels,
            model_name=model_name,
            device=self.device,
            use_contrastive=self.use_contrastive
        )
        
        print(f"Predictor ready!")
        if multi_label:
            print(f"Labels: {self.label_names} (multi-label mode)")
            if use_contrastive:
                print(f"Using contrastive learning model")
        else:
            print(f"Classes: {self.label_names} (single-class mode)")
    
    def predict_image(self, image_path: str, return_probs: bool = True) -> Dict:
        """
        Predict class/labels for a single image.
        
        Args:
            image_path: Path to image file
            return_probs: Whether to return probabilities
            
        Returns:
            Dictionary containing prediction results
        """
        # Load image
        image = Image.open(image_path)
        
        if image.mode == 'RGBA':
            # Directly use 4-channel image (e.g., PNG with alpha)
            image_4c = image
        else:
            # Handle 1-channel or 3-channel images (e.g., JPGs) by registering 4th channel
            image_4c = image.convert('RGB')         
        
        image_tensor = self.transform(image_4c).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            model_output = self.model(image_tensor)
            
            # Handle contrastive model output (returns 3 values: output, rgb_feat, ref_feat)
            if self.use_contrastive:
                outputs = model_output[0]  # Only use classification output
            else:
                outputs = model_output
            
            if self.multi_label:
                # Multi-label classification
                probs = torch.sigmoid(outputs)
                predictions = (probs >= self.threshold).float()
                
                result = {
                    'image_path': image_path,
                    'predictions': {}
                }
                
                # Add per-label predictions
                for i, label_name in enumerate(self.label_names):
                    result['predictions'][label_name] = {
                        'value': int(predictions[0][i].item()),
                        'probability': float(probs[0][i].item())
                    }
                
                if return_probs:
                    result['probabilities'] = {
                        self.label_names[i]: float(probs[0][i].item())
                        for i in range(len(self.label_names))
                    }
            else:
                # Single-class classification
                probs = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                predicted_class = predicted.item()
                confidence_score = confidence.item()
                
                result = {
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'predicted_label': self.label_names[predicted_class],
                    'confidence': confidence_score
                }
                
                if return_probs:
                    class_probs = {
                        self.label_names[i]: probs[0][i].item()
                        for i in range(len(self.label_names))
                    }
                    result['class_probabilities'] = class_probs
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            results.append(result)
        return results


def predict_single_image(
    image_path: str,
    checkpoint_path: str,
    model_name: str = 'convnextv2_tiny',
    backend: str = 'opencv',
    multi_label: bool = True
) -> Dict:
    """
    Convenience function to predict a single image.
    
    Args:
        image_path: Path to image file
        checkpoint_path: Path to model checkpoint
        model_name: Name of the model architecture
        backend: 'pil' or 'opencv'
        multi_label: Whether to use multi-label mode
        
    Returns:
        Prediction dictionary
    """
    predictor = QRCodePredictor(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        backend=backend,
        multi_label=multi_label
    )
    
    result = predictor.predict_image(image_path)
    
    print(f"\nPrediction for: {image_path}")
    
    if multi_label:
        print("Predictions:")
        predictions_dict = result['predictions']
        probs_dict = result.get('probabilities', {})
        
        has_positive = False
        for label, info in predictions_dict.items():
            val = info['value']
            prob = info['probability']
            status = "YES" if val == 1 else "NO "
            print(f"  [{status}] {label}: {prob:.2%}")
            if val == 1:
                has_positive = True
        
        if not has_positive:
            print("  (No labels detected above threshold)")
            
    else:
        print(f"Predicted Class: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nClass Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")
    
    return result


if __name__ == '__main__':
    predict_single_image(
        image_path='path/to/test/image.jpg',
        checkpoint_path='checkpoints/best_model.pth',
        model_name='convnextv2_tiny',
        multi_label=True
    )
