#!/usr/bin/env python3
"""
CLI script for training pain assessment models.

This script expects user-supplied training data in the specified format.
See demo_data/README.md for data format instructions.
"""
import argparse
import torch
import json
from pathlib import Path
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.models.sequence_model import create_model, count_parameters
from app.models.trainer import PainAssessmentTrainer, load_training_data, create_data_loaders
from app.config import settings


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train pain assessment models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "cnn"],
        default="lstm",
        help="Type of model to train"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=settings.hidden_size,
        help="Hidden layer size for LSTM models"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=settings.num_layers,
        help="Number of layers for LSTM models"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=settings.dropout,
        help="Dropout probability"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.batch_size,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=settings.epochs,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=settings.learning_rate,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=settings.sequence_length,
        help="Sequence length for temporal models"
    )
    
    # Data split arguments
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data for testing"
    )
    
    # Training options
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable data augmentation"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to resume training from checkpoint"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup training device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    return device


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def load_and_validate_data(data_path: str):
    """Load and validate training data."""
    print(f"Loading training data from: {data_path}")
    
    try:
        features, labels = load_training_data(data_path)
    except FileNotFoundError:
        print(f"Error: Training data file not found: {data_path}")
        print("Please ensure the data file exists and contains 'features' and 'labels' keys.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(features)} samples")
    print(f"Feature dimension: {len(features[0]) if features else 0}")
    print(f"Label range: {min(labels):.1f} - {max(labels):.1f}")
    
    return features, labels


def create_output_directory(output_dir: str) -> Path:
    """Create output directory for models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_training_config(args, output_path: Path):
    """Save training configuration."""
    config = {
        "model_type": args.model_type,
        "model_config": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "sequence_length": args.sequence_length
        },
        "training_config": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "augmentation": not args.no_augmentation,
            "early_stopping_patience": args.early_stopping_patience
        },
        "data_config": {
            "data_path": args.data_path,
            "seed": args.seed
        }
    }
    
    config_path = output_path / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training configuration saved to: {config_path}")


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    output_path = create_output_directory(args.output_dir)
    
    # Save training configuration
    save_training_config(args, output_path)
    
    # Load and validate data
    features, labels = load_and_validate_data(args.data_path)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        features=features,
        labels=labels,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        sequence_length=args.sequence_length,
        augment_train=not args.no_augmentation
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model_kwargs = {
        "input_features": settings.input_features,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout
    }
    
    model = create_model(args.model_type, **model_kwargs)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = PainAssessmentTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate
    )
    
    # Load checkpoint if provided
    if args.checkpoint_path:
        print(f"Loading checkpoint from: {args.checkpoint_path}")
        # Implementation would load checkpoint state
    
    # Train model
    print("Starting training...")
    model_path = output_path / f"pain_assessment_model_{args.model_type}.pt"
    
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_path=str(model_path),
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    print("\nFinal Test Results:")
    print(f"  MAE: {test_metrics['mae']:.2f}")
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  Correlation: {test_metrics['correlation']:.3f}")
    print(f"  Accuracy (±1): {test_metrics['accuracy_1pt']:.1%}")
    print(f"  Accuracy (±2): {test_metrics['accuracy_2pt']:.1%}")
    print(f"  Average Confidence: {test_metrics['avg_confidence']:.3f}")
    
    # Save test results
    results_path = output_path / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")
    
    # Print usage instructions
    print(f"\nTo use the trained model:")
    print(f"  1. Copy {model_path} to your deployment directory")
    print(f"  2. Update MODEL_PATH environment variable")
    print(f"  3. Restart the inference service")


if __name__ == "__main__":
    main()