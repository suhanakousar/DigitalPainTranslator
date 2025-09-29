"""
PyTorch sequence model for pain assessment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PainAssessmentModel(nn.Module):
    """
    LSTM-based sequence model for pain assessment from facial features.
    """
    
    def __init__(
        self,
        input_features: int = 8,  # 5 facial features + 3 caregiver inputs
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize the pain assessment model.
        
        Args:
            input_features: Number of input features
            hidden_size: Hidden state size for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_size: Output dimension (1 for pain score)
        """
        super(PainAssessmentModel, self).__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_features)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature importance layer
        self.feature_importance = nn.Linear(hidden_size * 2, input_features)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Confidence estimation layer
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)
               For single inference, sequence_length can be 1
            
        Returns:
            Tuple of (pain_score, confidence)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Handle single time-step input by expanding to sequence
        if seq_len == 1:
            # For single time-step, repeat to create a small sequence
            x = x.repeat(1, 3, 1)  # Repeat 3 times
            seq_len = 3
        
        # Reshape for batch normalization
        x_reshaped = x.view(-1, self.input_features)
        x_norm = self.input_norm(x_reshaped)
        x_norm = x_norm.view(batch_size, seq_len, self.input_features)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x_norm)
        
        # Apply attention mechanism
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last attended output for classification
        final_output = attended_out[:, -1, :]  # Shape: (batch_size, hidden_size * 2)
        
        # Pain score prediction (0-10 scale)
        pain_logits = self.classifier(final_output)
        pain_score = torch.sigmoid(pain_logits) * 10.0  # Scale to 0-10
        
        # Confidence estimation
        confidence = self.confidence_estimator(final_output)
        
        return pain_score.squeeze(), confidence.squeeze()
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature importance scores for explainability.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature importance scores
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        if seq_len == 1:
            x = x.repeat(1, 3, 1)
            seq_len = 3
        
        # Forward pass through LSTM
        x_reshaped = x.view(-1, self.input_features)
        x_norm = self.input_norm(x_reshaped)
        x_norm = x_norm.view(batch_size, seq_len, self.input_features)
        
        lstm_out, _ = self.lstm(x_norm)
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        final_output = attended_out[:, -1, :]
        
        # Generate feature importance
        importance = self.feature_importance(final_output)
        importance = torch.softmax(importance, dim=-1)
        
        return importance


class CNN1DPainModel(nn.Module):
    """
    Alternative 1D CNN model for pain assessment.
    """
    
    def __init__(
        self,
        input_features: int = 8,
        num_filters: int = 64,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.2
    ):
        """
        Initialize 1D CNN model.
        
        Args:
            input_features: Number of input features
            num_filters: Number of convolutional filters
            kernel_sizes: Tuple of kernel sizes for different conv layers
            dropout: Dropout probability
        """
        super(CNN1DPainModel, self).__init__()
        
        self.input_features = input_features
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_features)
        
        # Multiple convolutional branches with different kernel sizes
        self.conv_branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(input_features, num_filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.conv_branches.append(branch)
        
        # Combine features from all branches
        combined_features = num_filters * len(kernel_sizes)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, combined_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_features // 2, combined_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_features // 4, 1)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(combined_features, combined_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_features // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CNN model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)
            
        Returns:
            Tuple of (pain_score, confidence)
        """
        batch_size, seq_len, _ = x.shape
        
        # Transpose for conv1d: (batch_size, input_features, sequence_length)
        x = x.transpose(1, 2)
        
        # Normalize input
        x = self.input_norm(x)
        
        # Apply each convolutional branch
        branch_outputs = []
        for branch in self.conv_branches:
            branch_out = branch(x)  # Shape: (batch_size, num_filters, 1)
            branch_out = branch_out.squeeze(-1)  # Shape: (batch_size, num_filters)
            branch_outputs.append(branch_out)
        
        # Concatenate branch outputs
        combined = torch.cat(branch_outputs, dim=1)
        
        # Pain score prediction
        pain_logits = self.classifier(combined)
        pain_score = torch.sigmoid(pain_logits) * 10.0
        
        # Confidence estimation
        confidence = self.confidence_estimator(combined)
        
        return pain_score.squeeze(), confidence.squeeze()


def create_model(model_type: str = "lstm", **kwargs) -> nn.Module:
    """
    Factory function to create different model types.
    
    Args:
        model_type: Type of model ('lstm' or 'cnn')
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model
    """
    if model_type.lower() == "lstm":
        return PainAssessmentModel(**kwargs)
    elif model_type.lower() == "cnn":
        return CNN1DPainModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    file_path: str,
    model_version: str = "1.0"
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        file_path: Path to save checkpoint
        model_version: Model version string
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'version': model_version,
        'model_config': {
            'input_features': getattr(model, 'input_features', 8),
            'hidden_size': getattr(model, 'hidden_size', 64),
            'num_layers': getattr(model, 'num_layers', 2),
        }
    }
    
    torch.save(checkpoint, file_path)


def load_model_checkpoint(file_path: str, device: torch.device) -> dict:
    """
    Load model checkpoint.
    
    Args:
        file_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(file_path, map_location=device)
    return checkpoint