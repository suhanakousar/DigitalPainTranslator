"""
Pain assessment predictor with model loading, preprocessing, and explainability.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from pathlib import Path

from ..schemas import (
    FacialFeatures, CaregiverInputs, InferenceResponse, 
    ExplanationItem, InferenceRequest
)
from ..utils import Timer, generate_session_id, load_json_file
from ..config import settings
from .feature_extractor import FacialFeatureExtractor


class BaselinePredictor:
    """Baseline deterministic predictor with explicit feature weights."""
    
    def __init__(self, weights_path: Optional[str] = None):
        """Initialize baseline predictor with weights."""
        self.weights = self._load_baseline_weights(weights_path)
        self.model_version = "baseline-1.0"
    
    def _load_baseline_weights(self, weights_path: Optional[str] = None) -> Dict[str, float]:
        """Load baseline feature weights."""
        if weights_path and Path(weights_path).exists():
            weights = load_json_file(weights_path)
            if weights:
                return weights
        
        # Default weights based on pain research literature
        return {
            "mouthOpen": 0.15,
            "eyeClosureAvg": 0.20,
            "browFurrowAvg": 0.25,
            "headTiltVar": 0.10,
            "microMovementVar": 0.15,
            "grimace": 0.08,
            "breathing": 0.04,
            "restlessness": 0.03
        }
    
    def predict(self, features: FacialFeatures, caregiver_inputs: CaregiverInputs) -> Tuple[float, float, List[ExplanationItem]]:
        """
        Predict pain score using baseline model.
        
        Returns:
            Tuple of (score, confidence, explanations)
        """
        # Normalize caregiver inputs to [0, 1]
        normalized_caregiver = {
            "grimace": caregiver_inputs.grimace / 5.0,
            "breathing": caregiver_inputs.breathing / 5.0,
            "restlessness": caregiver_inputs.restlessness / 5.0
        }
        
        # Calculate weighted sum
        contributions = {}
        total_score = 0.0
        
        # Facial features
        for feature_name in ["mouthOpen", "eyeClosureAvg", "browFurrowAvg", "headTiltVar", "microMovementVar"]:
            value = getattr(features, feature_name)
            weight = self.weights.get(feature_name, 0.0)
            contribution = weight * value
            contributions[feature_name] = contribution
            total_score += contribution
        
        # Caregiver inputs
        for input_name, value in normalized_caregiver.items():
            weight = self.weights.get(input_name, 0.0)
            contribution = weight * value
            contributions[input_name] = contribution
            total_score += contribution
        
        # Scale to 0-10 range
        pain_score = min(10.0, max(0.0, total_score * 10.0))
        
        # Calculate confidence based on input consistency
        confidence = self._calculate_baseline_confidence(features, caregiver_inputs, contributions)
        
        # Generate explanations
        explanations = self._generate_baseline_explanations(contributions)
        
        return pain_score, confidence, explanations
    
    def _calculate_baseline_confidence(self, features: FacialFeatures, caregiver_inputs: CaregiverInputs, contributions: Dict[str, float]) -> float:
        """Calculate prediction confidence for baseline model."""
        # Confidence based on input magnitude and consistency
        feature_values = [
            features.mouthOpen, features.eyeClosureAvg, features.browFurrowAvg,
            features.headTiltVar, features.microMovementVar
        ]
        
        caregiver_values = [
            caregiver_inputs.grimace / 5.0,
            caregiver_inputs.breathing / 5.0,
            caregiver_inputs.restlessness / 5.0
        ]
        
        # Higher confidence when values are more decisive (not near 0.5)
        feature_confidence = np.mean([abs(v - 0.5) * 2 for v in feature_values])
        caregiver_confidence = np.mean([abs(v - 0.5) * 2 for v in caregiver_values])
        
        overall_confidence = (feature_confidence + caregiver_confidence) / 2.0
        return float(np.clip(overall_confidence, 0.1, 1.0))
    
    def _generate_baseline_explanations(self, contributions: Dict[str, float]) -> List[ExplanationItem]:
        """Generate explanations for baseline predictions."""
        explanations = []
        total_abs_contribution = sum(abs(c) for c in contributions.values())
        
        if total_abs_contribution == 0:
            return explanations
        
        for signal, contribution in contributions.items():
            importance = abs(contribution) / total_abs_contribution
            
            explanations.append(ExplanationItem(
                signal=signal,
                contribution=float(contribution),
                importance=float(importance)
            ))
        
        # Sort by importance (descending)
        explanations.sort(key=lambda x: x.importance, reverse=True)
        return explanations


class MLPredictor:
    """Machine learning predictor using PyTorch models."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize ML predictor."""
        self.model = None
        self.model_version = "unknown"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load PyTorch model from file."""
        try:
            path = Path(model_path)
            if not path.exists():
                return False
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model architecture and weights
            if isinstance(checkpoint, dict):
                model_state = checkpoint.get('model_state_dict', checkpoint)
                self.model_version = checkpoint.get('version', 'ml-1.0')
            else:
                model_state = checkpoint
                self.model_version = 'ml-1.0'
            
            # Initialize model (will be implemented in models/sequence_model.py)
            from ..models.sequence_model import PainAssessmentModel
            self.model = PainAssessmentModel(
                input_features=settings.input_features,
                hidden_size=settings.hidden_size,
                num_layers=settings.num_layers,
                dropout=settings.dropout
            )
            
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
            
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def predict(self, features: FacialFeatures, caregiver_inputs: CaregiverInputs) -> Tuple[float, float, List[ExplanationItem]]:
        """
        Predict pain score using ML model.
        
        Returns:
            Tuple of (score, confidence, explanations)
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Prepare input tensor
        input_tensor = self._prepare_input_tensor(features, caregiver_inputs)
        
        with torch.no_grad():
            # Forward pass
            output = self.model(input_tensor)
            
            # Extract predictions
            if isinstance(output, tuple):
                pain_score, confidence = output
            else:
                pain_score = output
                confidence = torch.tensor(0.8)  # Default confidence
            
            pain_score = float(pain_score.item())
            confidence_score = float(confidence.item())
        
        # Generate explanations using gradient-based attribution
        explanations = self._generate_ml_explanations(input_tensor, features, caregiver_inputs)
        
        return pain_score, confidence_score, explanations
    
    def _prepare_input_tensor(self, features: FacialFeatures, caregiver_inputs: CaregiverInputs) -> torch.Tensor:
        """Prepare input tensor for model inference."""
        # Combine facial features and caregiver inputs
        input_features = [
            features.mouthOpen,
            features.eyeClosureAvg,
            features.browFurrowAvg,
            features.headTiltVar,
            features.microMovementVar,
            caregiver_inputs.grimace / 5.0,
            caregiver_inputs.breathing / 5.0,
            caregiver_inputs.restlessness / 5.0
        ]
        
        # Convert to tensor and add batch dimension
        tensor = torch.tensor(input_features, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def _generate_ml_explanations(self, input_tensor: torch.Tensor, features: FacialFeatures, caregiver_inputs: CaregiverInputs) -> List[ExplanationItem]:
        """Generate explanations using gradient-based attribution."""
        if self.model is None:
            return []
        
        # Enable gradient computation for input
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            score = output[0]
        else:
            score = output
        
        # Backward pass to get gradients
        score.backward()
        
        # Get input gradients
        gradients = input_tensor.grad.squeeze().cpu().numpy()
        input_values = input_tensor.squeeze().detach().cpu().numpy()
        
        # Calculate attributions (gradient * input)
        attributions = gradients * input_values
        
        # Feature names corresponding to input order
        feature_names = [
            "mouthOpen", "eyeClosureAvg", "browFurrowAvg", 
            "headTiltVar", "microMovementVar",
            "grimace", "breathing", "restlessness"
        ]
        
        # Create explanations
        explanations = []
        total_abs_attribution = sum(abs(attr) for attr in attributions)
        
        if total_abs_attribution > 0:
            for i, (name, attribution) in enumerate(zip(feature_names, attributions)):
                importance = abs(attribution) / total_abs_attribution
                
                explanations.append(ExplanationItem(
                    signal=name,
                    contribution=float(attribution),
                    importance=float(importance)
                ))
        
        # Sort by importance
        explanations.sort(key=lambda x: x.importance, reverse=True)
        return explanations


class PainAssessmentPredictor:
    """Main predictor class that combines baseline and ML models."""
    
    def __init__(self, model_path: Optional[str] = None, baseline_weights_path: Optional[str] = None):
        """Initialize predictor with fallback to baseline."""
        self.feature_extractor = FacialFeatureExtractor()
        self.baseline_predictor = BaselinePredictor(baseline_weights_path)
        self.ml_predictor = None
        
        # Try to load ML model, fallback to baseline
        if model_path and Path(model_path).exists():
            self.ml_predictor = MLPredictor(model_path)
            if self.ml_predictor.model is None:
                self.ml_predictor = None
    
    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """
        Main prediction method that handles the full inference pipeline.
        
        Args:
            request: InferenceRequest containing landmarks or features and caregiver inputs
            
        Returns:
            InferenceResponse with prediction results
        """
        with Timer() as timer:
            # Extract or use provided features
            if request.landmarks:
                features = self.feature_extractor.extract_features_from_landmarks(request.landmarks)
            elif request.features:
                features = request.features
            else:
                raise ValueError("Either landmarks or features must be provided")
            
            # Choose predictor (ML with baseline fallback)
            predictor = self.ml_predictor if self.ml_predictor else self.baseline_predictor
            
            # Perform prediction
            pain_score, confidence, explanations = predictor.predict(features, request.caregiverInputs)
            
            # Generate recommended actions
            recommended_actions = self._generate_recommendations(pain_score, confidence, explanations)
            
            # Build response
            response = InferenceResponse(
                session_id=request.session_id or generate_session_id(),
                score=pain_score,
                confidence=confidence,
                explanation=explanations,
                recommendedActions=recommended_actions,
                model_version=predictor.model_version,
                processing_ms=timer.elapsed_ms
            )
            
            return response
    
    def _generate_recommendations(self, score: float, confidence: float, explanations: List[ExplanationItem]) -> List[str]:
        """Generate recommended actions based on assessment results."""
        recommendations = []
        
        # Score-based recommendations
        if score >= 7.0:
            recommendations.append("Consider immediate pain intervention")
            recommendations.append("Monitor vital signs closely")
        elif score >= 4.0:
            recommendations.append("Assess need for pain management")
            recommendations.append("Continue monitoring")
        else:
            recommendations.append("Current comfort level appears adequate")
        
        # Confidence-based recommendations
        if confidence < 0.5:
            recommendations.append("Consider additional assessment methods")
            recommendations.append("Verify with direct patient communication if possible")
        
        # Feature-specific recommendations
        top_contributors = explanations[:3]  # Top 3 contributing features
        for explanation in top_contributors:
            if explanation.signal == "grimace" and explanation.importance > 0.3:
                recommendations.append("Facial expression indicates significant discomfort")
            elif explanation.signal == "breathing" and explanation.importance > 0.3:
                recommendations.append("Breathing pattern suggests distress")
            elif explanation.signal == "restlessness" and explanation.importance > 0.3:
                recommendations.append("Physical restlessness observed")
        
        return recommendations
    
    def reload_model(self, model_path: Optional[str] = None) -> bool:
        """Reload the ML model."""
        if model_path:
            new_predictor = MLPredictor(model_path)
            if new_predictor.model is not None:
                self.ml_predictor = new_predictor
                return True
        return False
    
    @property
    def current_model_version(self) -> str:
        """Get current model version."""
        if self.ml_predictor:
            return self.ml_predictor.model_version
        return self.baseline_predictor.model_version
    
    @property
    def is_ml_model_loaded(self) -> bool:
        """Check if ML model is loaded."""
        return self.ml_predictor is not None