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
import matplotlib.pyplot as plt
import numpy as np

from src.model import load_model_for_inference
from src.dataset import get_inference_transform
from src.utils import get_device


class QRCodePredictor:
    """Predictor for QR code classification (single-class or multi-label)."""
    
    def __init__(
        self,
        checkpoint_path: str,
        label_names: List[str] = None,
        model_name: str = 'convnextv2_tiny',
        image_size: int = 224,
        device: str = None,
        backend: str = 'opencv',
        multi_label: bool = False,
        threshold: float = 0.5
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
        """
        self.device = get_device() if device is None else torch.device(device)
        self.image_size = image_size
        self.multi_label = multi_label
        self.threshold = threshold
        # Use the requested backend
        self.transform = get_inference_transform(image_size, backend=backend)
        self.backend = backend
        
        # Load label/class names
        if label_names is None:
            class_names_path = os.path.join(os.path.dirname(checkpoint_path), 'class_names.json')
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    label_names = json.load(f)
            else:
                if multi_label:
                    label_names = ["is_copied", "is_blurry", "is_low_light"]  # Default for multi-label
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
            device=self.device
        )
        
        print(f"Predictor ready!")
        if multi_label:
            print(f"Labels: {self.label_names} (multi-label mode)")
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
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
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
    
    def visualize_prediction(self, image_path: str, save_path: str = None):
        """
        Visualize prediction for a single image.
        
        Args:
            image_path: Path to image file
            save_path: Optional path to save visualization
        """
        # Get prediction
        result = self.predict_image(image_path)
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f"Input Image\n{os.path.basename(image_path)}", fontsize=12)
        
        # Display prediction
        # Display prediction
        if self.multi_label:
            class_probs = result['probabilities']
            classes = list(class_probs.keys())
            probs = list(class_probs.values())
            
            # Highlight positive predictions
            predictions_dict = result['predictions']
            colors = ['green' if predictions_dict[c]['value'] == 1 else 'gray' for c in classes]
            
            # Text summary
            positives = [label for label, info in predictions_dict.items() if info['value'] == 1]
            pred_text = f"Predictions: {', '.join(positives) if positives else 'None'}"
        else:
            class_probs = result['class_probabilities']
            classes = list(class_probs.keys())
            probs = list(class_probs.values())
            
            colors = ['green' if c == result['predicted_label'] else 'gray' for c in classes]
            
            pred_text = f"Prediction: {result['predicted_label']}\nConfidence: {result['confidence']:.2%}"

        bars = ax2.barh(classes, probs, color=colors)
        ax2.set_xlabel('Probability', fontsize=11)
        ax2.set_title('Class Probabilities', fontsize=12)
        ax2.set_xlim([0, 1])
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{prob:.2%}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Add prediction text
        fig.text(0.5, 0.02, pred_text, ha='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def predict_single_image(
    image_path: str,
    checkpoint_path: str,
    model_name: str = 'convnextv2_tiny',
    visualize: bool = True,
    backend: str = 'opencv',
    multi_label: bool = True
) -> Dict:
    """
    Convenience function to predict a single image.
    
    Args:
        image_path: Path to image file
        checkpoint_path: Path to model checkpoint
        model_name: Name of the model architecture
        visualize: Whether to visualize the prediction
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
    
    if visualize:
        predictor.visualize_prediction(image_path)
    
    return result


if __name__ == '__main__':
    # Example usage
    predict_single_image(
        image_path='path/to/test/image.jpg',
        checkpoint_path='checkpoints/best_model.pth',
        model_name='convnextv2_tiny',
        visualize=True
    )
