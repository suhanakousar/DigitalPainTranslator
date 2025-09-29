"""
Training pipeline for pain assessment models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import time

from .sequence_model import PainAssessmentModel, CNN1DPainModel, save_model_checkpoint
from ..utils import Timer


class PainAssessmentDataset(Dataset):
    """Dataset class for pain assessment training data."""
    
    def __init__(
        self,
        features_data: List[List[float]],
        labels: List[float],
        sequence_length: int = 1,
        augment: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            features_data: List of feature vectors
            labels: List of pain scores (0-10)
            sequence_length: Length of sequences for temporal models
            augment: Whether to apply data augmentation
        """
        self.features_data = features_data
        self.labels = labels
        self.sequence_length = sequence_length
        self.augment = augment
        
        # Normalize labels to [0, 1] for training
        self.normalized_labels = [label / 10.0 for label in labels]
        
        # Create sequences if needed
        if sequence_length > 1:
            self.sequences, self.sequence_labels = self._create_sequences()
        else:
            self.sequences = [[features] for features in features_data]
            self.sequence_labels = self.normalized_labels
    
    def _create_sequences(self) -> Tuple[List[List[List[float]]], List[float]]:
        """Create sequences from individual feature vectors."""
        sequences = []
        sequence_labels = []
        
        for i in range(len(self.features_data) - self.sequence_length + 1):
            sequence = self.features_data[i:i + self.sequence_length]
            label = self.normalized_labels[i + self.sequence_length - 1]  # Use last label
            
            sequences.append(sequence)
            sequence_labels.append(label)
        
        return sequences, sequence_labels
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        label = self.sequence_labels[idx]
        
        # Convert to tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        # Apply augmentation if enabled
        if self.augment and self.training:
            sequence_tensor = self._augment_sequence(sequence_tensor)
        
        return sequence_tensor, label_tensor
    
    def _augment_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to sequence."""
        # Add small random noise
        noise = torch.randn_like(sequence) * 0.01
        augmented = sequence + noise
        
        # Clamp values to valid ranges
        augmented = torch.clamp(augmented, 0.0, 1.0)
        
        return augmented
    
    def train(self):
        """Set dataset to training mode."""
        self.training = True
    
    def eval(self):
        """Set dataset to evaluation mode."""
        self.training = False


class PainAssessmentTrainer:
    """Trainer class for pain assessment models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Loss function - MSE for regression
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions, confidence = self.model(sequences)
            
            # Calculate loss
            loss = self.criterion(predictions, labels)
            
            # Add confidence regularization
            confidence_loss = torch.mean((confidence - 0.8) ** 2)  # Encourage high confidence
            total_loss_value = loss + 0.01 * confidence_loss
            
            # Backward pass
            total_loss_value.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                predictions, confidence = self.model(sequences)
                
                # Calculate loss
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                # Store for metrics calculation
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, predictions: List[float], labels: List[float]) -> Dict[str, float]:
        """Calculate validation metrics."""
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Convert back to 0-10 scale for interpretable metrics
        pred_scores = predictions * 10.0
        true_scores = labels * 10.0
        
        # Mean Absolute Error
        mae = np.mean(np.abs(pred_scores - true_scores))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((pred_scores - true_scores) ** 2))
        
        # Correlation coefficient
        correlation = np.corrcoef(pred_scores, true_scores)[0, 1]
        
        # Accuracy within thresholds
        acc_1 = np.mean(np.abs(pred_scores - true_scores) <= 1.0)  # Within 1 point
        acc_2 = np.mean(np.abs(pred_scores - true_scores) <= 2.0)  # Within 2 points
        
        return {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation if not np.isnan(correlation) else 0.0,
            'accuracy_1pt': acc_1,
            'accuracy_2pt': acc_2
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 20
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_path: Path to save best model
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate epoch
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                if save_path:
                    save_model_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_loss,
                        save_path,
                        model_version="1.0"
                    )
            else:
                patience_counter += 1
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val MAE: {val_metrics['mae']:.2f}")
            print(f"  Val Correlation: {val_metrics['correlation']:.3f}")
            print(f"  Time: {epoch_time:.1f}s")
            print()
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on test set."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                predictions, confidence = self.model(sequences)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_labels)
        
        # Add confidence statistics
        metrics['avg_confidence'] = float(np.mean(all_confidences))
        metrics['std_confidence'] = float(np.std(all_confidences))
        
        return metrics


def load_training_data(data_path: str) -> Tuple[List[List[float]], List[float]]:
    """
    Load training data from file.
    
    Expected format: JSON file with 'features' and 'labels' keys.
    Features should be list of feature vectors.
    Labels should be list of pain scores (0-10).
    
    Args:
        data_path: Path to training data file
        
    Returns:
        Tuple of (features, labels)
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    if 'features' not in data or 'labels' not in data:
        raise ValueError("Training data must contain 'features' and 'labels' keys")
    
    features = data['features']
    labels = data['labels']
    
    if len(features) != len(labels):
        raise ValueError("Number of features and labels must match")
    
    return features, labels


def create_data_loaders(
    features: List[List[float]],
    labels: List[float],
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    sequence_length: int = 1,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        features: Feature data
        labels: Label data
        batch_size: Batch size for data loaders
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        sequence_length: Sequence length for temporal models
        augment_train: Whether to augment training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    n_samples = len(features)
    indices = np.random.permutation(n_samples)
    
    # Split indices
    test_size = int(n_samples * test_split)
    val_size = int(n_samples * val_split)
    train_size = n_samples - test_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_features = [features[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_dataset = PainAssessmentDataset(
        train_features, train_labels, sequence_length, augment=augment_train
    )
    train_dataset.train()
    
    val_features = [features[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    val_dataset = PainAssessmentDataset(
        val_features, val_labels, sequence_length, augment=False
    )
    val_dataset.eval()
    
    test_features = [features[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_dataset = PainAssessmentDataset(
        test_features, test_labels, sequence_length, augment=False
    )
    test_dataset.eval()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader, test_loader