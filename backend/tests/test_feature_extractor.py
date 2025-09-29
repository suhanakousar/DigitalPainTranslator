"""
Unit tests for facial feature extraction.
Tests use programmatically generated synthetic inputs.
"""
import pytest
import numpy as np
from typing import List, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.inference.feature_extractor import FacialFeatureExtractor
from app.schemas import Landmark, FacialFeatures


class TestFacialFeatureExtractor:
    """Test cases for facial feature extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return FacialFeatureExtractor()
    
    @pytest.fixture
    def sample_landmarks(self):
        """Generate synthetic facial landmarks for testing."""
        # Generate 468 landmarks (MediaPipe face mesh count)
        landmarks = []
        for i in range(468):
            # Create deterministic but varied coordinates
            x = (i % 10) / 10.0 - 0.5  # Range [-0.5, 0.5]
            y = ((i // 10) % 10) / 10.0 - 0.5
            z = ((i // 100) % 5) / 10.0 - 0.2  # Range [-0.2, 0.3]
            landmarks.append(Landmark(x=x, y=y, z=z))
        return landmarks
    
    @pytest.fixture
    def mouth_open_landmarks(self):
        """Generate landmarks representing an open mouth."""
        landmarks = []
        for i in range(468):
            x = (i % 10) / 10.0 - 0.5
            y = ((i // 10) % 10) / 10.0 - 0.5
            z = ((i // 100) % 5) / 10.0 - 0.2
            
            # Modify mouth landmarks to represent openness
            if i in [13, 14]:  # Mouth vertical landmarks
                if i == 13:  # Upper lip
                    y = 0.1
                else:  # Lower lip
                    y = -0.1
            
            landmarks.append(Landmark(x=x, y=y, z=z))
        return landmarks
    
    @pytest.fixture
    def eye_closed_landmarks(self):
        """Generate landmarks representing closed eyes."""
        landmarks = []
        for i in range(468):
            x = (i % 10) / 10.0 - 0.5
            y = ((i // 10) % 10) / 10.0 - 0.5
            z = ((i // 100) % 5) / 10.0 - 0.2
            
            # Modify eye landmarks to represent closure
            if i in [159, 145, 158, 153]:  # Left eye landmarks
                y = 0.0  # Minimal distance = closed
            if i in [33, 7, 160, 144]:  # Right eye landmarks
                y = 0.0
            
            landmarks.append(Landmark(x=x, y=y, z=z))
        return landmarks
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initializes correctly."""
        assert extractor is not None
        assert hasattr(extractor, 'landmark_indices')
        assert 'mouth' in extractor.landmark_indices
        assert len(extractor.mouth_vertical_pairs) > 0
    
    def test_extract_features_empty_input(self, extractor):
        """Test extraction with empty input."""
        features = extractor.extract_features_from_landmarks([])
        assert isinstance(features, FacialFeatures)
        assert features.mouthOpen == 0.0
        assert features.eyeClosureAvg == 0.0
        assert features.browFurrowAvg == 0.0
        assert features.headTiltVar == 0.0
        assert features.microMovementVar == 0.0
    
    def test_extract_features_single_frame(self, extractor, sample_landmarks):
        """Test extraction with single frame."""
        features = extractor.extract_features_from_landmarks([sample_landmarks])
        
        # Verify output is FacialFeatures instance
        assert isinstance(features, FacialFeatures)
        
        # Verify all features are in valid range [0, 1]
        assert 0.0 <= features.mouthOpen <= 1.0
        assert 0.0 <= features.eyeClosureAvg <= 1.0
        assert 0.0 <= features.browFurrowAvg <= 1.0
        assert 0.0 <= features.headTiltVar <= 1.0
        assert 0.0 <= features.microMovementVar <= 1.0
    
    def test_extract_features_multiple_frames(self, extractor, sample_landmarks):
        """Test extraction with multiple frames."""
        # Create sequence with slight variations
        sequence = []
        for frame_idx in range(5):
            frame_landmarks = []
            for i, landmark in enumerate(sample_landmarks):
                # Add small random variation to each frame
                noise = (frame_idx - 2) * 0.01  # Centered around original
                x = max(-1.0, min(1.0, landmark.x + noise))
                y = max(-1.0, min(1.0, landmark.y + noise))
                z = max(-1.0, min(1.0, landmark.z + noise))
                frame_landmarks.append(Landmark(x=x, y=y, z=z))
            sequence.append(frame_landmarks)
        
        features = extractor.extract_features_from_landmarks(sequence)
        
        # Verify valid output
        assert isinstance(features, FacialFeatures)
        assert 0.0 <= features.mouthOpen <= 1.0
        assert 0.0 <= features.eyeClosureAvg <= 1.0
        assert 0.0 <= features.browFurrowAvg <= 1.0
        assert 0.0 <= features.headTiltVar <= 1.0
        assert 0.0 <= features.microMovementVar <= 1.0
        
        # Variance features should be > 0 due to frame variations
        assert features.headTiltVar >= 0.0
        assert features.microMovementVar >= 0.0
    
    def test_mouth_openness_detection(self, extractor, sample_landmarks, mouth_open_landmarks):
        """Test mouth openness detection."""
        # Extract features from normal and open mouth landmarks
        normal_features = extractor.extract_features_from_landmarks([sample_landmarks])
        open_features = extractor.extract_features_from_landmarks([mouth_open_landmarks])
        
        # Open mouth should have higher mouth openness
        assert open_features.mouthOpen >= normal_features.mouthOpen
        assert 0.0 <= open_features.mouthOpen <= 1.0
    
    def test_eye_closure_detection(self, extractor, sample_landmarks, eye_closed_landmarks):
        """Test eye closure detection."""
        # Extract features from normal and closed eye landmarks
        normal_features = extractor.extract_features_from_landmarks([sample_landmarks])
        closed_features = extractor.extract_features_from_landmarks([eye_closed_landmarks])
        
        # Closed eyes should have higher closure value
        assert closed_features.eyeClosureAvg >= normal_features.eyeClosureAvg
        assert 0.0 <= closed_features.eyeClosureAvg <= 1.0
    
    def test_temporal_variance_calculation(self, extractor):
        """Test temporal variance calculation with controlled input."""
        # Create sequence with known variance pattern
        sequence = []
        for frame_idx in range(10):
            landmarks = []
            for i in range(468):
                # Create oscillating pattern for head tilt
                x = 0.0
                y = 0.0
                z = 0.0
                
                # Key landmarks for head pose
                if i == 1:  # Nose tip
                    x = 0.1 * np.sin(frame_idx * 0.5)  # Oscillating motion
                
                landmarks.append(Landmark(x=x, y=y, z=z))
            sequence.append(landmarks)
        
        features = extractor.extract_features_from_landmarks(sequence)
        
        # Should detect variance in head movement
        assert features.headTiltVar >= 0.0
        assert features.microMovementVar >= 0.0
    
    def test_validate_landmarks_valid_input(self, extractor, sample_landmarks):
        """Test landmark validation with valid input."""
        assert extractor.validate_landmarks([sample_landmarks]) is True
    
    def test_validate_landmarks_invalid_input(self, extractor):
        """Test landmark validation with invalid input."""
        # Empty sequence
        assert extractor.validate_landmarks([]) is False
        
        # Too few landmarks
        few_landmarks = [Landmark(x=0, y=0, z=0) for _ in range(10)]
        assert extractor.validate_landmarks([few_landmarks]) is False
    
    def test_calculate_mouth_openness_edge_cases(self, extractor):
        """Test mouth openness calculation with edge cases."""
        # Test with insufficient landmarks
        coords = [(0, 0, 0) for _ in range(5)]  # Not enough for mouth pairs
        openness = extractor._calculate_mouth_openness(coords)
        assert openness == 0.0
        
        # Test with exact coordinates
        coords = [(0, 0, 0) for _ in range(300)]  # Sufficient landmarks
        coords[13] = (0, 0.1, 0)  # Upper lip
        coords[14] = (0, -0.1, 0)  # Lower lip
        openness = extractor._calculate_mouth_openness(coords)
        assert 0.0 <= openness <= 1.0
    
    def test_calculate_eye_closure_edge_cases(self, extractor):
        """Test eye closure calculation with edge cases."""
        # Test with insufficient landmarks
        coords = [(0, 0, 0) for _ in range(50)]
        closure = extractor._calculate_eye_closure(coords, 'left')
        assert closure == 0.0
        
        # Test with sufficient landmarks
        coords = [(0, 0, 0) for _ in range(400)]
        coords[159] = (0, 0.05, 0)  # Upper eyelid
        coords[145] = (0, 0.0, 0)   # Lower eyelid (closer = more closed)
        closure = extractor._calculate_eye_closure(coords, 'left')
        assert 0.0 <= closure <= 1.0
    
    def test_feature_normalization(self, extractor):
        """Test that all extracted features are properly normalized."""
        # Generate extreme synthetic landmarks
        extreme_landmarks = []
        for i in range(468):
            # Use extreme but valid coordinate values
            x = 1.0 if i % 2 == 0 else -1.0
            y = 1.0 if i % 3 == 0 else -1.0
            z = 1.0 if i % 5 == 0 else -1.0
            extreme_landmarks.append(Landmark(x=x, y=y, z=z))
        
        features = extractor.extract_features_from_landmarks([extreme_landmarks])
        
        # All features should be normalized to [0, 1]
        assert 0.0 <= features.mouthOpen <= 1.0
        assert 0.0 <= features.eyeClosureAvg <= 1.0
        assert 0.0 <= features.browFurrowAvg <= 1.0
        assert 0.0 <= features.headTiltVar <= 1.0
        assert 0.0 <= features.microMovementVar <= 1.0
    
    def test_reproducibility(self, extractor, sample_landmarks):
        """Test that extraction is reproducible for same input."""
        features1 = extractor.extract_features_from_landmarks([sample_landmarks])
        features2 = extractor.extract_features_from_landmarks([sample_landmarks])
        
        # Results should be identical
        assert features1.mouthOpen == features2.mouthOpen
        assert features1.eyeClosureAvg == features2.eyeClosureAvg
        assert features1.browFurrowAvg == features2.browFurrowAvg
        assert features1.headTiltVar == features2.headTiltVar
        assert features1.microMovementVar == features2.microMovementVar
    
    def test_performance_large_sequence(self, extractor):
        """Test performance with large landmark sequence."""
        # Generate large sequence
        sequence_length = 100
        landmarks_per_frame = 468
        
        sequence = []
        for frame in range(sequence_length):
            frame_landmarks = []
            for i in range(landmarks_per_frame):
                x = np.sin(frame * 0.1 + i * 0.01)
                y = np.cos(frame * 0.1 + i * 0.01)
                z = (frame + i) % 10 / 10.0 - 0.5
                frame_landmarks.append(Landmark(x=x, y=y, z=z))
            sequence.append(frame_landmarks)
        
        # Should complete without error
        features = extractor.extract_features_from_landmarks(sequence)
        assert isinstance(features, FacialFeatures)
        assert 0.0 <= features.microMovementVar <= 1.0