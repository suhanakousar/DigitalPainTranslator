"""
Feature extraction from facial landmarks for pain assessment.
Implements the same deterministic formulas used in the frontend.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from ..schemas import Landmark, FacialFeatures
from ..utils import euclidean_distance, normalize_value, calculate_variance
from ..config import settings


class FacialFeatureExtractor:
    """Extracts pain-relevant features from facial landmarks."""
    
    def __init__(self):
        """Initialize feature extractor with landmark indices."""
        self.landmark_indices = settings.landmark_indices
        
        # Mouth landmark pairs for openness calculation
        self.mouth_vertical_pairs = [
            (13, 14),  # Upper and lower lip center
            (12, 15),  # Upper and lower lip corners
        ]
        
        # Eye landmark indices for closure calculation
        self.left_eye_vertical_pairs = [(159, 145), (158, 153)]
        self.right_eye_vertical_pairs = [(33, 7), (160, 144)]
        
        # Eyebrow landmarks for furrow detection
        self.left_brow_indices = [46, 53]
        self.right_brow_indices = [276, 283]
        
        # Reference points for head pose estimation
        self.head_pose_landmarks = [1, 5, 6, 8, 10, 151, 9, 10, 151, 152]
    
    def extract_features_from_landmarks(self, landmarks_sequence: List[List[Landmark]]) -> FacialFeatures:
        """
        Extract facial features from a sequence of facial landmarks.
        
        Args:
            landmarks_sequence: List of landmark frames, each containing facial landmarks
            
        Returns:
            FacialFeatures: Computed facial features for pain assessment
        """
        if not landmarks_sequence:
            return self._get_default_features()
        
        # Extract features from each frame
        frame_features = []
        for frame_landmarks in landmarks_sequence:
            if frame_landmarks:
                features = self._extract_frame_features(frame_landmarks)
                frame_features.append(features)
        
        if not frame_features:
            return self._get_default_features()
        
        # Aggregate features across frames
        return self._aggregate_temporal_features(frame_features)
    
    def _extract_frame_features(self, landmarks: List[Landmark]) -> Dict[str, float]:
        """Extract features from a single frame of landmarks."""
        features = {}
        
        # Convert landmarks to coordinate arrays for easier processing
        coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
        
        # Mouth openness
        features['mouth_open'] = self._calculate_mouth_openness(coords)
        
        # Eye closure
        features['left_eye_closure'] = self._calculate_eye_closure(coords, 'left')
        features['right_eye_closure'] = self._calculate_eye_closure(coords, 'right')
        
        # Brow furrow
        features['left_brow_furrow'] = self._calculate_brow_furrow(coords, 'left')
        features['right_brow_furrow'] = self._calculate_brow_furrow(coords, 'right')
        
        # Head tilt
        features['head_tilt'] = self._calculate_head_tilt(coords)
        
        # Micro movements (calculated later across frames)
        features['micro_movement'] = 0.0
        
        return features
    
    def _calculate_mouth_openness(self, coords: List[Tuple[float, float, float]]) -> float:
        """Calculate mouth openness from landmark coordinates."""
        if len(coords) < max(max(pair) for pair in self.mouth_vertical_pairs) + 1:
            return 0.0
        
        distances = []
        for upper_idx, lower_idx in self.mouth_vertical_pairs:
            if upper_idx < len(coords) and lower_idx < len(coords):
                upper_point = coords[upper_idx]
                lower_point = coords[lower_idx]
                distance = euclidean_distance(
                    (upper_point[0], upper_point[1]),
                    (lower_point[0], lower_point[1])
                )
                distances.append(distance)
        
        if not distances:
            return 0.0
        
        avg_distance = np.mean(distances)
        # Normalize to [0, 1] based on typical mouth opening ranges
        normalized = normalize_value(avg_distance, 0.0, 0.1)
        return float(normalized)
    
    def _calculate_eye_closure(self, coords: List[Tuple[float, float, float]], eye: str) -> float:
        """Calculate eye closure amount."""
        pairs = self.left_eye_vertical_pairs if eye == 'left' else self.right_eye_vertical_pairs
        
        distances = []
        for upper_idx, lower_idx in pairs:
            if upper_idx < len(coords) and lower_idx < len(coords):
                upper_point = coords[upper_idx]
                lower_point = coords[lower_idx]
                distance = euclidean_distance(
                    (upper_point[0], upper_point[1]),
                    (lower_point[0], lower_point[1])
                )
                distances.append(distance)
        
        if not distances:
            return 0.0
        
        avg_distance = np.mean(distances)
        # Invert: smaller distance = more closure
        # Normalize to [0, 1] where 1 is fully closed
        normalized = 1.0 - normalize_value(avg_distance, 0.0, 0.05)
        return float(max(0.0, normalized))
    
    def _calculate_brow_furrow(self, coords: List[Tuple[float, float, float]], side: str) -> float:
        """Calculate eyebrow furrow intensity."""
        indices = self.left_brow_indices if side == 'left' else self.right_brow_indices
        
        if len(coords) < max(indices) + 1:
            return 0.0
        
        # Calculate vertical displacement of eyebrow landmarks
        brow_points = [coords[idx] for idx in indices if idx < len(coords)]
        if len(brow_points) < 2:
            return 0.0
        
        # Calculate average y-coordinate (vertical position)
        avg_y = np.mean([point[1] for point in brow_points])
        
        # Normalize based on typical eyebrow movement range
        # Lower y values (eyebrows pulled down) indicate more furrow
        normalized = normalize_value(-avg_y, -0.1, 0.1)
        return float(normalized)
    
    def _calculate_head_tilt(self, coords: List[Tuple[float, float, float]]) -> float:
        """Calculate head tilt from facial landmarks."""
        if len(coords) < max(self.head_pose_landmarks) + 1:
            return 0.0
        
        # Use key facial landmarks to estimate head orientation
        try:
            nose_tip = coords[1]  # Nose tip
            left_mouth = coords[61]  # Left mouth corner
            right_mouth = coords[291]  # Right mouth corner
            
            # Calculate angle between mouth corners
            dx = right_mouth[0] - left_mouth[0]
            dy = right_mouth[1] - left_mouth[1]
            
            if dx == 0:
                return 0.0
            
            angle = np.arctan(dy / dx)
            # Normalize angle to [0, 1]
            normalized = normalize_value(abs(angle), 0.0, np.pi / 6)  # 30 degrees max
            return float(normalized)
            
        except (IndexError, ZeroDivisionError):
            return 0.0
    
    def _aggregate_temporal_features(self, frame_features: List[Dict[str, float]]) -> FacialFeatures:
        """Aggregate features across temporal frames."""
        if not frame_features:
            return self._get_default_features()
        
        # Calculate averages for most features
        mouth_open_avg = np.mean([f['mouth_open'] for f in frame_features])
        
        left_eye_avg = np.mean([f['left_eye_closure'] for f in frame_features])
        right_eye_avg = np.mean([f['right_eye_closure'] for f in frame_features])
        eye_closure_avg = (left_eye_avg + right_eye_avg) / 2.0
        
        left_brow_avg = np.mean([f['left_brow_furrow'] for f in frame_features])
        right_brow_avg = np.mean([f['right_brow_furrow'] for f in frame_features])
        brow_furrow_avg = (left_brow_avg + right_brow_avg) / 2.0
        
        # Calculate variance for motion-based features
        head_tilt_values = [f['head_tilt'] for f in frame_features]
        head_tilt_var = calculate_variance(head_tilt_values)
        
        # Calculate micro-movement variance from all features
        micro_movement_var = self._calculate_micro_movement_variance(frame_features)
        
        return FacialFeatures(
            mouthOpen=float(mouth_open_avg),
            eyeClosureAvg=float(eye_closure_avg),
            browFurrowAvg=float(brow_furrow_avg),
            headTiltVar=float(head_tilt_var),
            microMovementVar=float(micro_movement_var)
        )
    
    def _calculate_micro_movement_variance(self, frame_features: List[Dict[str, float]]) -> float:
        """Calculate micro-movement variance across frames."""
        if len(frame_features) < 2:
            return 0.0
        
        # Collect all feature values across frames
        all_values = []
        for features in frame_features:
            values = [
                features['mouth_open'],
                features['left_eye_closure'],
                features['right_eye_closure'],
                features['left_brow_furrow'],
                features['right_brow_furrow'],
                features['head_tilt']
            ]
            all_values.extend(values)
        
        # Calculate overall variance
        variance = calculate_variance(all_values)
        
        # Normalize to [0, 1]
        normalized = normalize_value(variance, 0.0, 0.1)
        return float(normalized)
    
    def _get_default_features(self) -> FacialFeatures:
        """Return default features when extraction fails."""
        return FacialFeatures(
            mouthOpen=0.0,
            eyeClosureAvg=0.0,
            browFurrowAvg=0.0,
            headTiltVar=0.0,
            microMovementVar=0.0
        )
    
    def validate_landmarks(self, landmarks_sequence: List[List[Landmark]]) -> bool:
        """Validate that landmarks sequence is properly formatted."""
        if not landmarks_sequence:
            return False
        
        for frame_landmarks in landmarks_sequence:
            if frame_landmarks and len(frame_landmarks) < 50:  # Minimum expected landmarks
                return False
        
        return True