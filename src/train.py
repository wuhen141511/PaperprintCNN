"""
Training script for QR code multi-label classification model.
Implements the complete training pipeline with validation and checkpointing.
Supports both single-class and multi-label classification.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, List, Tuple
import json

from src.model import create_model
from src.dataset import create_dataloaders
from src.utils import set_seed, get_device, save_checkpoint, load_checkpoint, calculate_metrics, calculate_multilabel_metrics, QRCodeContrastiveLoss


class Trainer:
    """Training manager for QR code classification (single-class or multi-label)."""
    
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        model_name: str = 'resnet50',
        num_labels: int = 4,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 20,
        image_size: int = 224,
        freeze_backbone: bool = True,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        device: str = None,
        seed: int = 42,
        multi_label: bool = True,
        label_names: List[str] = None,
        load_checkpoint_path: str = None,
        pretrained_path: str = None,
        use_contrastive: bool = False,
        contrastive_weight: float = 0.3,
        classification_weight: float = 1.0,
        use_gray: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            model_name: Name of the model architecture
            num_labels: Number of labels (for multi-label) or classes (for single-class)
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            num_epochs: Number of training epochs
            image_size: Input image size
            freeze_backbone: Whether to freeze backbone initially
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            device: Device to use (None for auto-detection)
            seed: Random seed for reproducibility
            multi_label: Whether to use multi-label classification
            label_names: List of label names for multi-label classification
            load_checkpoint_path: Optional path to checkpoint file to resume training
            use_contrastive: Whether to use contrastive learning
            contrastive_weight: Weight for contrastive loss
            classification_weight: Weight for classification loss
        """
        # Set random seed
        set_seed(seed)
        
        # Store configuration
        self.multi_label = multi_label
        self.label_names = label_names
        self.use_contrastive = use_contrastive
        
        # Setup device
        self.device = get_device() if device is None else torch.device(device)
        
        # Create data loaders
        print("\nLoading datasets...")
        self.train_loader, self.val_loader, self.class_names = create_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=0,  # Set to 0 for Windows compatibility
            multi_label=multi_label,
            label_names=label_names,
            use_contrastive=use_contrastive,
            use_gray=use_gray
        )
        
        # Create model
        print("\nCreating model...")
        self.model_name = model_name
        self.model_type = 'contrastive' if use_contrastive else 'regular'
        self.num_labels = num_labels
        self.model = create_model(
            num_labels=num_labels,
            model_name=model_name,
            pretrained=True,
            freeze_backbone=freeze_backbone,
            device=self.device,
            pretrained_path=pretrained_path,
            use_contrastive=use_contrastive
        )
        
        # Setup training components
        if multi_label:
            if use_contrastive:
                # Contrastive learning: Use both contrastive and classification loss
                self.contrastive_criterion = QRCodeContrastiveLoss(margin=1.0, temperature=0.5)
                self.classification_criterion = nn.BCEWithLogitsLoss()
                self.contrastive_weight = contrastive_weight
                self.classification_weight = classification_weight
                print("Using contrastive learning with QRCodeContrastiveLoss")
            else:
                # Standard multi-label classification: BCEWithLogitsLoss
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            # Single-class classification: CrossEntropyLoss
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Training parameters
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.start_epoch = 1
        
        # Load checkpoint if provided
        if load_checkpoint_path is not None:
            if os.path.exists(load_checkpoint_path):
                self.model, self.optimizer, loaded_epoch, loaded_loss, loaded_acc = load_checkpoint(
                    self.model, self.optimizer, load_checkpoint_path, self.device
                )
                self.start_epoch = loaded_epoch + 1  # Resume from next epoch
                self.best_val_acc = loaded_acc
                self.best_epoch = loaded_epoch
                print(f"Resuming training from epoch {self.start_epoch}")
                
                # Load training history if available
                history_path = os.path.join(os.path.dirname(load_checkpoint_path), 'training_history.json')
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        self.history = json.load(f)
                    print(f"Loaded training history from {history_path}")
            else:
                print(f"Warning: Checkpoint file not found at {load_checkpoint_path}. Starting from scratch.")
        
        print(f"\nClass names: {self.class_names}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        running_contrastive_loss = 0.0
        running_classification_loss = 0.0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Train]')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_contrastive:
                # Contrastive learning: model returns (output, rgb_feat, ref_feat)
                outputs, rgb_feat, ref_feat = self.model(images)
                
                # Calculate losses
                contrastive_loss = self.contrastive_criterion(rgb_feat, ref_feat, labels)
                classification_loss = self.classification_criterion(outputs, labels)
                
                # Weighted sum
                loss = (self.contrastive_weight * contrastive_loss + 
                         self.classification_weight * classification_loss)
                
                # Track losses
                running_contrastive_loss += contrastive_loss.item() * images.size(0)
                running_classification_loss += classification_loss.item() * images.size(0)
            else:
                # Standard training
                outputs = self.model(images)
                
                # For multi-label, labels should be float; for single-class, long
                if self.multi_label:
                    loss = self.criterion(outputs, labels)
                else:
                    loss = self.criterion(outputs, labels.long())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            all_predictions.append(outputs.detach())
            all_labels.append(labels.detach())
            
            # Update progress bar
            if self.use_contrastive:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'contrastive': contrastive_loss.item(),
                    'cls': classification_loss.item()
                })
            else:
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader.dataset)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        if self.multi_label:
            metrics = calculate_multilabel_metrics(all_predictions, all_labels)
            epoch_acc = metrics['mean_accuracy']
        else:
            metrics = calculate_metrics(all_predictions, all_labels)
            epoch_acc = metrics['accuracy']
        
        if self.use_contrastive:
            return epoch_loss, epoch_acc, {
                'contrastive_loss': running_contrastive_loss / len(self.train_loader.dataset),
                'classification_loss': running_classification_loss / len(self.train_loader.dataset)
            }
        else:
            return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Val]  ')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.use_contrastive:
                    # Contrastive learning: model returns (output, rgb_feat, ref_feat)
                    outputs, _, _ = self.model(images)
                else:
                    outputs = self.model(images)
                
                # For multi-label, labels should be float; for single-class, long
                if self.multi_label:
                    if self.use_contrastive:
                        loss = self.classification_criterion(outputs, labels)
                    else:
                        loss = self.criterion(outputs, labels)
                else:
                    loss = self.criterion(outputs, labels.long())
                
                # Track metrics
                running_loss += loss.item() * images.size(0)
                all_predictions.append(outputs)
                all_labels.append(labels)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader.dataset)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        if self.multi_label:
            metrics = calculate_multilabel_metrics(all_predictions, all_labels)
            epoch_acc = metrics['mean_accuracy']
        else:
            metrics = calculate_metrics(all_predictions, all_labels)
            epoch_acc = metrics['accuracy']
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        if self.start_epoch > 1:
            print(f"Resuming training from epoch {self.start_epoch} to {self.num_epochs}")
        else:
            print(f"Starting training for {self.num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            # Train
            if self.use_contrastive:
                train_loss, train_acc, train_losses_dict = self.train_epoch(epoch)
            else:
                train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Log contrastive losses if using contrastive learning
            if self.use_contrastive:
                self.writer.add_scalar('Loss/contrastive', train_losses_dict['contrastive_loss'], epoch)
                self.writer.add_scalar('Loss/classification', train_losses_dict['classification_loss'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            if self.use_contrastive:
                print(f"    - Contrastive Loss: {train_losses_dict['contrastive_loss']:.4f}")
                print(f"    - Classification Loss: {train_losses_dict['classification_loss']:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(
                self.model, self.optimizer, epoch, val_loss, val_acc, checkpoint_path,
                model_type=self.model_type, model_name=self.model_name, num_labels=self.num_labels
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, val_acc, best_path,
                    model_type=self.model_type, model_name=self.model_name, num_labels=self.num_labels
                )
                print(f"  ✓ New best model saved! (Acc: {val_acc:.4f})")
            
            print(f"  Best Val Acc: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
            print(f"{'-'*60}\n")
        
        # Training complete
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        
        # Save training history
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
        # Save class names
        class_names_path = os.path.join(self.checkpoint_dir, 'class_names.json')
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f, indent=2)
        print(f"Class names saved to {class_names_path}")
        
        self.writer.close()
        
        return self.history


def train_model(
    train_dir: str,
    val_dir: str,
    model_name: str = 'resnet50',
    num_labels: int = 4,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: int = 20,
    image_size: int = 224,
    freeze_backbone: bool = True,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs',
    multi_label: bool = True,
    label_names: List[str] = None,
    load_checkpoint_path: str = None,
    pretrained_path: str = None,
    use_contrastive: bool = False,
    contrastive_weight: float = 0.3,
    classification_weight: float = 1.0,
    use_gray: bool = False
):
    """
    Convenience function to train a model.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        model_name: Name of the model architecture
        num_labels: Number of labels (for multi-label) or classes (for single-class)
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        image_size: Input image size
        freeze_backbone: Whether to freeze backbone initially
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        multi_label: Whether to use multi-label classification
        label_names: List of label names for multi-label classification
        load_checkpoint_path: Optional path to checkpoint file to resume training
        pretrained_path: Optional path to local pretrained weights file
        use_contrastive: Whether to use contrastive learning
        contrastive_weight: Weight for contrastive loss
        classification_weight: Weight for classification loss
    """
    trainer = Trainer(
        train_dir=train_dir,
        val_dir=val_dir,
        model_name=model_name,
        num_labels=num_labels,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        image_size=image_size,
        freeze_backbone=freeze_backbone,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        multi_label=multi_label,
        label_names=label_names,
        load_checkpoint_path=load_checkpoint_path,
        pretrained_path=pretrained_path,
        use_contrastive=use_contrastive,
        contrastive_weight=contrastive_weight,
        classification_weight=classification_weight,
        use_gray=use_gray
    )
    
    history = trainer.train()
    return history


if __name__ == '__main__':
    # Example usage for multi-label classification
    train_model(
        train_dir='data/train',
        val_dir='data/val',
        model_name='resnet50',
        num_labels=3,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=20,
        freeze_backbone=True,
        multi_label=True,
        label_names=["is_copied", "is_low_light", "is_blurry"]
    )
