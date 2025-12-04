"""
Training script for QR code classification model.
Implements the complete training pipeline with validation and checkpointing.
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
from src.utils import set_seed, get_device, save_checkpoint, calculate_metrics


class Trainer:
    """Training manager for QR code classification."""
    
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        model_name: str = 'resnet50',
        num_classes: int = 2,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 20,
        image_size: int = 224,
        freeze_backbone: bool = True,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        device: str = None,
        seed: int = 42
    ):
        """
        Initialize trainer.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            model_name: Name of the model architecture
            num_classes: Number of classes
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            num_epochs: Number of training epochs
            image_size: Input image size
            freeze_backbone: Whether to freeze backbone initially
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            device: Device to use (None for auto-detection)
            seed: Random seed for reproducibility
        """
        # Set random seed
        set_seed(seed)
        
        # Setup device
        self.device = get_device() if device is None else torch.device(device)
        
        # Create data loaders
        print("\nLoading datasets...")
        self.train_loader, self.val_loader, self.class_names = create_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        # Create model
        print("\nCreating model...")
        self.model = create_model(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=True,
            freeze_backbone=freeze_backbone,
            device=self.device
        )
        
        # Setup training components
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
        
        print(f"\nClass names: {self.class_names}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs} [Train]')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            all_predictions.append(outputs.detach())
            all_labels.append(labels.detach())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader.dataset)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metrics = calculate_metrics(all_predictions, all_labels)
        epoch_acc = metrics['accuracy']
        
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
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
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
        metrics = calculate_metrics(all_predictions, all_labels)
        epoch_acc = metrics['accuracy']
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.num_epochs + 1):
            # Train
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
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(
                self.model, self.optimizer, epoch, val_loss, val_acc, checkpoint_path
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, val_acc, best_path
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
    num_classes: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: int = 20,
    image_size: int = 224,
    freeze_backbone: bool = True,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs'
):
    """
    Convenience function to train a model.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        model_name: Name of the model architecture
        num_classes: Number of classes
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        image_size: Input image size
        freeze_backbone: Whether to freeze backbone initially
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    """
    trainer = Trainer(
        train_dir=train_dir,
        val_dir=val_dir,
        model_name=model_name,
        num_classes=num_classes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        image_size=image_size,
        freeze_backbone=freeze_backbone,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    history = trainer.train()
    return history


if __name__ == '__main__':
    # Example usage
    train_model(
        train_dir='data/train',
        val_dir='data/val',
        model_name='resnet50',
        num_classes=2,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=20,
        freeze_backbone=True
    )
