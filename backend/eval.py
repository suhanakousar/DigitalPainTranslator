#!/usr/bin/env python3
"""
CLI script for evaluating and exporting pain assessment models.

This script provides model evaluation and export functionality.
"""
import argparse
import torch
import json
import onnx
from pathlib import Path
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.models.sequence_model import create_model, load_model_checkpoint
from app.models.trainer import load_training_data, create_data_loaders, PainAssessmentTrainer
from app.config import settings


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate and export pain assessment models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to evaluation data JSON file"
    )
    
    # Export arguments
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export model to ONNX format"
    )
    parser.add_argument(
        "--export-torchscript",
        action="store_true",
        help="Export model to TorchScript format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="Directory to save exported models"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1,
        help="Sequence length for evaluation"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for evaluation"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup evaluation device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def load_model_from_checkpoint(model_path: str, device: torch.device):
    """Load model from checkpoint file."""
    print(f"Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        # Load checkpoint
        checkpoint = load_model_checkpoint(model_path, device)
        
        # Extract model configuration
        model_config = checkpoint.get('model_config', {})
        model_type = model_config.get('type', 'lstm')  # Default to LSTM
        
        # Create model with configuration
        model_kwargs = {
            "input_features": model_config.get('input_features', settings.input_features),
            "hidden_size": model_config.get('hidden_size', settings.hidden_size),
            "num_layers": model_config.get('num_layers', settings.num_layers),
            "dropout": model_config.get('dropout', settings.dropout)
        }
        
        model = create_model(model_type, **model_kwargs)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully")
        print(f"  Version: {checkpoint.get('version', 'unknown')}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Loss: {checkpoint.get('loss', 'unknown')}")
        
        return model, model_config
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def evaluate_model(model, data_path: str, device: torch.device, batch_size: int, sequence_length: int):
    """Evaluate model on test data."""
    if not data_path:
        print("No evaluation data provided, skipping evaluation")
        return None
    
    print(f"Loading evaluation data from: {data_path}")
    
    try:
        features, labels = load_training_data(data_path)
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return None
    
    # Create data loader
    _, _, test_loader = create_data_loaders(
        features=features,
        labels=labels,
        batch_size=batch_size,
        val_split=0.0,
        test_split=1.0,  # Use all data for testing
        sequence_length=sequence_length,
        augment_train=False
    )
    
    # Create trainer for evaluation
    trainer = PainAssessmentTrainer(model=model, device=device)
    
    # Evaluate
    print("Evaluating model...")
    metrics = trainer.evaluate(test_loader)
    
    print("\nEvaluation Results:")
    print(f"  Samples: {len(test_loader.dataset)}")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  Correlation: {metrics['correlation']:.3f}")
    print(f"  Accuracy (±1): {metrics['accuracy_1pt']:.1%}")
    print(f"  Accuracy (±2): {metrics['accuracy_2pt']:.1%}")
    print(f"  Average Confidence: {metrics['avg_confidence']:.3f}")
    print(f"  Std Confidence: {metrics['std_confidence']:.3f}")
    
    return metrics


def export_onnx(model, output_dir: str, sequence_length: int):
    """Export model to ONNX format."""
    try:
        import onnx
        import torch.onnx
    except ImportError:
        print("ONNX not available, skipping ONNX export")
        return False
    
    print("Exporting to ONNX format...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_path / "pain_assessment_model.onnx"
    
    # Create dummy input
    dummy_input = torch.randn(1, sequence_length, settings.input_features)
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['features'],
            output_names=['pain_score', 'confidence'],
            dynamic_axes={
                'features': {0: 'batch_size', 1: 'sequence_length'},
                'pain_score': {0: 'batch_size'},
                'confidence': {0: 'batch_size'}
            }
        )
        
        # Verify the ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        print(f"ONNX model exported to: {onnx_path}")
        return True
        
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False


def export_torchscript(model, output_dir: str, sequence_length: int):
    """Export model to TorchScript format."""
    print("Exporting to TorchScript format...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    script_path = output_path / "pain_assessment_model.pt"
    
    # Create dummy input
    dummy_input = torch.randn(1, sequence_length, settings.input_features)
    
    try:
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Save traced model
        traced_model.save(str(script_path))
        
        # Test the traced model
        with torch.no_grad():
            original_output = model(dummy_input)
            traced_output = traced_model(dummy_input)
            
            # Check if outputs are close
            if isinstance(original_output, tuple):
                score_diff = torch.abs(original_output[0] - traced_output[0]).max()
                conf_diff = torch.abs(original_output[1] - traced_output[1]).max()
                max_diff = max(score_diff, conf_diff)
            else:
                max_diff = torch.abs(original_output - traced_output).max()
            
            if max_diff < 1e-5:
                print(f"TorchScript model exported to: {script_path}")
                return True
            else:
                print(f"Warning: TorchScript model outputs differ by {max_diff}")
                return False
        
    except Exception as e:
        print(f"TorchScript export failed: {e}")
        return False


def save_model_info(model_config: dict, metrics: dict, output_dir: str):
    """Save model information and metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    info_path = output_path / "model_info.json"
    
    model_info = {
        "model_config": model_config,
        "evaluation_metrics": metrics or {},
        "export_timestamp": str(torch.tensor(0).item()),  # Current timestamp placeholder
        "framework_version": torch.__version__,
        "usage_instructions": {
            "onnx": "Load with onnxruntime for cross-platform inference",
            "torchscript": "Load with torch.jit.load() for PyTorch inference",
            "pytorch": "Load with torch.load() and model.load_state_dict()"
        }
    }
    
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model information saved to: {info_path}")


def main():
    """Main evaluation and export function."""
    args = parse_arguments()
    
    # Setup device
    device = setup_device(args.device)
    
    # Load model
    model, model_config = load_model_from_checkpoint(args.model_path, device)
    
    # Evaluate model if data provided
    metrics = None
    if args.data_path:
        metrics = evaluate_model(
            model=model,
            data_path=args.data_path,
            device=device,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length
        )
    
    # Export models if requested
    if args.export_onnx or args.export_torchscript:
        print(f"\nExporting models to: {args.output_dir}")
        
        if args.export_onnx:
            export_onnx(model, args.output_dir, args.sequence_length)
        
        if args.export_torchscript:
            export_torchscript(model, args.output_dir, args.sequence_length)
        
        # Save model information
        save_model_info(model_config, metrics, args.output_dir)
        
        print("\nExport completed!")
        print(f"Exported files available in: {args.output_dir}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()