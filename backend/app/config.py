"""
Configuration settings for the application.
"""
from typing import List
import os
from pathlib import Path


class Settings:
    """Application settings."""
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: List[str] = [
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ]
    
    # Model configuration
    model_path: str = os.getenv("MODEL_PATH", "models/pain_assessment_model.pt")
    baseline_weights_path: str = os.getenv("BASELINE_WEIGHTS_PATH", "models/baseline_weights.json")
    
    # Data configuration
    records_db_path: str = os.getenv("RECORDS_DB_PATH", "data/records.json")
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    sequence_length: int = 30
    
    # Model architecture
    input_features: int = 5  # mouthOpen, eyeClosureAvg, browFurrowAvg, headTiltVar, microMovementVar
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    
    # Inference configuration
    confidence_threshold: float = 0.7
    max_inference_time_ms: int = 1000
    
    # Feature extraction
    landmark_indices = {
        "mouth": list(range(61, 68)),  # Mouth landmarks
        "left_eye": list(range(362, 382)),  # Left eye landmarks
        "right_eye": list(range(33, 133)),  # Right eye landmarks
        "left_eyebrow": list(range(46, 53)),  # Left eyebrow landmarks
        "right_eyebrow": list(range(276, 283)),  # Right eyebrow landmarks
    }


settings = Settings()