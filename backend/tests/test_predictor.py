"""
Unit tests for pain assessment predictor.
Tests use programmatically generated synthetic inputs.
"""
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.inference.predictor import BaselinePredictor, MLPredictor, PainAssessmentPredictor
from app.schemas import FacialFeatures, CaregiverInputs, InferenceRequest, InferenceResponse, Landmark


class TestBaselinePredictor:
    """Test cases for baseline predictor."""
    
    @pytest.fixture
    def predictor(self):
        """Create baseline predictor instance."""
        return BaselinePredictor()
    
    @pytest.fixture
    def sample_features(self):
        """Generate synthetic facial features."""
        return FacialFeatures(
            mouthOpen=0.3,
            eyeClosureAvg=0.2,
            browFurrowAvg=0.4,
            headTiltVar=0.1,
            microMovementVar=0.15
        )
    
    @pytest.fixture
    def sample_caregiver_inputs(self):
        """Generate synthetic caregiver inputs."""
        return CaregiverInputs(
            grimace=2,
            breathing=1,
            restlessness=3,
            gestures=["clench"]
        )
    
    def test_predictor_initialization(self, predictor):
        """Test predictor initializes correctly."""
        assert predictor is not None
        assert hasattr(predictor, 'weights')
        assert hasattr(predictor, 'model_version')
        assert predictor.model_version == "baseline-1.0"
        
        # Check that all expected weights are present
        expected_weights = [
            'mouthOpen', 'eyeClosureAvg', 'browFurrowAvg', 
            'headTiltVar', 'microMovementVar', 'grimace', 
            'breathing', 'restlessness'
        ]
        for weight in expected_weights:
            assert weight in predictor.weights
            assert isinstance(predictor.weights[weight], (int, float))
    
    def test_predict_valid_input(self, predictor, sample_features, sample_caregiver_inputs):
        """Test prediction with valid input."""
        score, confidence, explanations = predictor.predict(sample_features, sample_caregiver_inputs)
        
        # Verify output types and ranges
        assert isinstance(score, float)
        assert isinstance(confidence, float)
        assert isinstance(explanations, list)
        
        assert 0.0 <= score <= 10.0
        assert 0.0 <= confidence <= 1.0
        assert len(explanations) > 0
        
        # Check explanation structure
        for explanation in explanations:
            assert hasattr(explanation, 'signal')
            assert hasattr(explanation, 'contribution')
            assert hasattr(explanation, 'importance')
            assert isinstance(explanation.signal, str)
            assert isinstance(explanation.contribution, float)
            assert 0.0 <= explanation.importance <= 1.0
    
    def test_predict_extreme_values(self, predictor):
        """Test prediction with extreme input values."""
        # Maximum values
        max_features = FacialFeatures(
            mouthOpen=1.0,
            eyeClosureAvg=1.0,
            browFurrowAvg=1.0,
            headTiltVar=1.0,
            microMovementVar=1.0
        )
        max_caregiver = CaregiverInputs(
            grimace=5,
            breathing=5,
            restlessness=5,
            gestures=["clench", "point", "shake"]
        )
        
        max_score, max_confidence, max_explanations = predictor.predict(max_features, max_caregiver)
        
        # Minimum values
        min_features = FacialFeatures(
            mouthOpen=0.0,
            eyeClosureAvg=0.0,
            browFurrowAvg=0.0,
            headTiltVar=0.0,
            microMovementVar=0.0
        )
        min_caregiver = CaregiverInputs(
            grimace=0,
            breathing=0,
            restlessness=0,
            gestures=[]
        )
        
        min_score, min_confidence, min_explanations = predictor.predict(min_features, min_caregiver)
        
        # Maximum input should generally produce higher score
        assert max_score >= min_score
        assert 0.0 <= min_score <= 10.0
        assert 0.0 <= max_score <= 10.0
        assert 0.0 <= min_confidence <= 1.0
        assert 0.0 <= max_confidence <= 1.0
    
    def test_explanation_ordering(self, predictor, sample_features, sample_caregiver_inputs):
        """Test that explanations are ordered by importance."""
        _, _, explanations = predictor.predict(sample_features, sample_caregiver_inputs)
        
        # Explanations should be sorted by importance (descending)
        for i in range(len(explanations) - 1):
            assert explanations[i].importance >= explanations[i + 1].importance
    
    def test_contribution_sum(self, predictor, sample_features, sample_caregiver_inputs):
        """Test that explanation contributions are reasonable."""
        score, _, explanations = predictor.predict(sample_features, sample_caregiver_inputs)
        
        # Sum of absolute contributions should relate to score
        total_contribution = sum(abs(exp.contribution) for exp in explanations)
        assert total_contribution > 0
        
        # Importance values should sum to approximately 1.0
        total_importance = sum(exp.importance for exp in explanations)
        assert 0.9 <= total_importance <= 1.1  # Allow small floating point errors
    
    def test_confidence_calculation(self, predictor):
        """Test confidence calculation logic."""
        # Test with decisive values (far from 0.5)
        decisive_features = FacialFeatures(
            mouthOpen=1.0,
            eyeClosureAvg=0.0,
            browFurrowAvg=1.0,
            headTiltVar=0.0,
            microMovementVar=1.0
        )
        decisive_caregiver = CaregiverInputs(
            grimace=5,
            breathing=0,
            restlessness=5,
            gestures=[]
        )
        
        _, decisive_confidence, _ = predictor.predict(decisive_features, decisive_caregiver)
        
        # Test with ambiguous values (close to 0.5)
        ambiguous_features = FacialFeatures(
            mouthOpen=0.5,
            eyeClosureAvg=0.5,
            browFurrowAvg=0.5,
            headTiltVar=0.5,
            microMovementVar=0.5
        )
        ambiguous_caregiver = CaregiverInputs(
            grimace=2,
            breathing=2,
            restlessness=3,
            gestures=[]
        )
        
        _, ambiguous_confidence, _ = predictor.predict(ambiguous_features, ambiguous_caregiver)
        
        # Decisive inputs should have higher confidence
        assert decisive_confidence >= ambiguous_confidence
        assert 0.1 <= decisive_confidence <= 1.0  # Minimum confidence clamp
        assert 0.1 <= ambiguous_confidence <= 1.0


class TestMLPredictor:
    """Test cases for ML predictor."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock ML model."""
        model = MagicMock()
        model.eval.return_value = None
        model.return_value = (torch.tensor(5.0), torch.tensor(0.8))
        return model
    
    @pytest.fixture
    def predictor(self, mock_model):
        """Create ML predictor with mock model."""
        predictor = MLPredictor()
        predictor.model = mock_model
        predictor.device = torch.device('cpu')
        return predictor
    
    @pytest.fixture
    def sample_features(self):
        """Generate synthetic facial features."""
        return FacialFeatures(
            mouthOpen=0.3,
            eyeClosureAvg=0.2,
            browFurrowAvg=0.4,
            headTiltVar=0.1,
            microMovementVar=0.15
        )
    
    @pytest.fixture
    def sample_caregiver_inputs(self):
        """Generate synthetic caregiver inputs."""
        return CaregiverInputs(
            grimace=2,
            breathing=1,
            restlessness=3,
            gestures=["clench"]
        )
    
    def test_predictor_initialization(self):
        """Test ML predictor initialization."""
        predictor = MLPredictor()
        assert predictor.model is None
        assert predictor.model_version == "unknown"
        assert predictor.device is not None
    
    def test_predict_with_model(self, predictor, sample_features, sample_caregiver_inputs):
        """Test prediction with loaded model."""
        score, confidence, explanations = predictor.predict(sample_features, sample_caregiver_inputs)
        
        # Verify output types and ranges
        assert isinstance(score, float)
        assert isinstance(confidence, float)
        assert isinstance(explanations, list)
        
        assert 0.0 <= score <= 10.0
        assert 0.0 <= confidence <= 1.0
    
    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        predictor = MLPredictor()
        features = FacialFeatures(
            mouthOpen=0.3, eyeClosureAvg=0.2, browFurrowAvg=0.4,
            headTiltVar=0.1, microMovementVar=0.15
        )
        caregiver = CaregiverInputs(grimace=2, breathing=1, restlessness=3)
        
        with pytest.raises(ValueError, match="No model loaded"):
            predictor.predict(features, caregiver)
    
    def test_input_tensor_preparation(self, predictor, sample_features, sample_caregiver_inputs):
        """Test input tensor preparation."""
        tensor = predictor._prepare_input_tensor(sample_features, sample_caregiver_inputs)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 8)  # Batch size 1, 8 features
        assert tensor.device == predictor.device
        
        # Verify feature order and normalization
        expected_values = [
            sample_features.mouthOpen,
            sample_features.eyeClosureAvg,
            sample_features.browFurrowAvg,
            sample_features.headTiltVar,
            sample_features.microMovementVar,
            sample_caregiver_inputs.grimace / 5.0,
            sample_caregiver_inputs.breathing / 5.0,
            sample_caregiver_inputs.restlessness / 5.0
        ]
        
        for i, expected in enumerate(expected_values):
            assert abs(tensor[0, i].item() - expected) < 1e-6


class TestPainAssessmentPredictor:
    """Test cases for main predictor class."""
    
    @pytest.fixture
    def predictor(self):
        """Create main predictor instance."""
        # Use baseline only for testing
        return PainAssessmentPredictor(model_path=None)
    
    @pytest.fixture
    def sample_request_features(self):
        """Generate request with features."""
        return InferenceRequest(
            features=FacialFeatures(
                mouthOpen=0.3,
                eyeClosureAvg=0.2,
                browFurrowAvg=0.4,
                headTiltVar=0.1,
                microMovementVar=0.15
            ),
            caregiverInputs=CaregiverInputs(
                grimace=2,
                breathing=1,
                restlessness=3,
                gestures=["clench"]
            )
        )
    
    @pytest.fixture
    def sample_request_landmarks(self):
        """Generate request with landmarks."""
        # Create minimal landmark sequence
        landmarks = []
        for i in range(468):
            x = (i % 10) / 10.0 - 0.5
            y = ((i // 10) % 10) / 10.0 - 0.5
            z = ((i // 100) % 5) / 10.0 - 0.2
            landmarks.append(Landmark(x=x, y=y, z=z))
        
        return InferenceRequest(
            landmarks=[landmarks],  # Single frame
            caregiverInputs=CaregiverInputs(
                grimace=2,
                breathing=1,
                restlessness=3,
                gestures=["clench"]
            )
        )
    
    def test_predictor_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor is not None
        assert predictor.feature_extractor is not None
        assert predictor.baseline_predictor is not None
        assert predictor.ml_predictor is None  # No model path provided
    
    def test_predict_with_features(self, predictor, sample_request_features):
        """Test prediction with precomputed features."""
        response = predictor.predict(sample_request_features)
        
        assert isinstance(response, InferenceResponse)
        assert response.session_id is not None
        assert 0.0 <= response.score <= 10.0
        assert 0.0 <= response.confidence <= 1.0
        assert len(response.explanation) > 0
        assert isinstance(response.recommendedActions, list)
        assert response.model_version == "baseline-1.0"
        assert response.processing_ms > 0
    
    def test_predict_with_landmarks(self, predictor, sample_request_landmarks):
        """Test prediction with landmarks."""
        response = predictor.predict(sample_request_landmarks)
        
        assert isinstance(response, InferenceResponse)
        assert response.session_id is not None
        assert 0.0 <= response.score <= 10.0
        assert 0.0 <= response.confidence <= 1.0
        assert len(response.explanation) > 0
        assert isinstance(response.recommendedActions, list)
        assert response.processing_ms > 0
    
    def test_predict_invalid_request(self, predictor):
        """Test prediction with invalid request."""
        # Request with neither landmarks nor features
        invalid_request = InferenceRequest(
            caregiverInputs=CaregiverInputs(
                grimace=2,
                breathing=1,
                restlessness=3
            )
        )
        
        with pytest.raises(ValueError, match="Either landmarks or features must be provided"):
            predictor.predict(invalid_request)
    
    def test_recommendation_generation(self, predictor):
        """Test recommendation generation based on scores."""
        # High pain score
        high_pain_request = InferenceRequest(
            features=FacialFeatures(
                mouthOpen=1.0,
                eyeClosureAvg=1.0,
                browFurrowAvg=1.0,
                headTiltVar=1.0,
                microMovementVar=1.0
            ),
            caregiverInputs=CaregiverInputs(
                grimace=5,
                breathing=5,
                restlessness=5,
                gestures=["clench", "point", "shake"]
            )
        )
        
        high_response = predictor.predict(high_pain_request)
        
        # Low pain score
        low_pain_request = InferenceRequest(
            features=FacialFeatures(
                mouthOpen=0.0,
                eyeClosureAvg=0.0,
                browFurrowAvg=0.0,
                headTiltVar=0.0,
                microMovementVar=0.0
            ),
            caregiverInputs=CaregiverInputs(
                grimace=0,
                breathing=0,
                restlessness=0,
                gestures=[]
            )
        )
        
        low_response = predictor.predict(low_pain_request)
        
        # High pain should have more urgent recommendations
        assert len(high_response.recommendedActions) >= len(low_response.recommendedActions)
        
        # Check for specific recommendation patterns
        high_recommendations = ' '.join(high_response.recommendedActions).lower()
        low_recommendations = ' '.join(low_response.recommendedActions).lower()
        
        # High pain should mention intervention
        if high_response.score >= 7.0:
            assert 'intervention' in high_recommendations or 'immediate' in high_recommendations
        
        # Low pain should mention adequate comfort
        if low_response.score < 4.0:
            assert 'adequate' in low_recommendations or 'comfort' in low_recommendations
    
    def test_model_version_tracking(self, predictor):
        """Test model version tracking."""
        assert predictor.current_model_version == "baseline-1.0"
        assert predictor.is_ml_model_loaded is False
    
    def test_session_id_handling(self, predictor, sample_request_features):
        """Test session ID handling."""
        # Request without session ID
        request_no_id = sample_request_features
        request_no_id.session_id = None
        
        response = predictor.predict(request_no_id)
        assert response.session_id is not None
        assert len(response.session_id) > 0
        
        # Request with session ID
        request_with_id = sample_request_features
        test_session_id = "test-session-123"
        request_with_id.session_id = test_session_id
        
        response = predictor.predict(request_with_id)
        assert response.session_id == test_session_id
    
    def test_processing_time_measurement(self, predictor, sample_request_features):
        """Test that processing time is measured."""
        response = predictor.predict(sample_request_features)
        
        assert response.processing_ms > 0
        assert response.processing_ms < 10000  # Should be reasonable (< 10 seconds)
    
    def test_feature_extraction_from_landmarks(self, predictor, sample_request_landmarks):
        """Test feature extraction integration."""
        response = predictor.predict(sample_request_landmarks)
        
        # Should successfully extract features and predict
        assert isinstance(response, InferenceResponse)
        assert response.score >= 0.0
        
        # Check that explanations include facial features
        explanation_signals = [exp.signal for exp in response.explanation]
        facial_features = ['mouthOpen', 'eyeClosureAvg', 'browFurrowAvg', 'headTiltVar', 'microMovementVar']
        
        for feature in facial_features:
            assert feature in explanation_signals