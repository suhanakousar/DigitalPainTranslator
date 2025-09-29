"""
Utility functions for geometric calculations and data processing.
"""
import numpy as np
from typing import List, Tuple
import uuid
import json
import time
from pathlib import Path


def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to [0, 1] range."""
    if max_val <= min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def calculate_variance(values: List[float]) -> float:
    """Calculate variance of a list of values."""
    if len(values) < 2:
        return 0.0
    
    mean = np.mean(values)
    variance = np.mean([(x - mean) ** 2 for x in values])
    return float(variance)


def moving_average(values: List[float], window_size: int = 3) -> List[float]:
    """Calculate moving average with specified window size."""
    if len(values) < window_size:
        return values
    
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window = values[start_idx:i + 1]
        result.append(np.mean(window))
    
    return result


def generate_session_id() -> str:
    """Generate a unique session identifier."""
    return str(uuid.uuid4())


def load_json_file(file_path: str) -> dict:
    """Safely load JSON file."""
    path = Path(file_path)
    if not path.exists():
        return {}
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_json_file(data: dict, file_path: str) -> bool:
    """Safely save data to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except IOError:
        return False


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        if self.start_time is None or self.end_time is None:
            return 0
        return int((self.end_time - self.start_time) * 1000)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to specified range."""
    return max(min_val, min(max_val, value))


def interpolate_missing_landmarks(landmarks: List[List], max_interpolation_gap: int = 5) -> List[List]:
    """Interpolate missing landmarks in sequence."""
    if not landmarks:
        return landmarks
    
    result = landmarks.copy()
    n_frames = len(landmarks)
    
    for frame_idx in range(n_frames):
        if not landmarks[frame_idx]:  # Empty frame
            # Find nearest non-empty frames
            prev_frame_idx = frame_idx - 1
            next_frame_idx = frame_idx + 1
            
            # Find previous valid frame
            while prev_frame_idx >= 0 and not landmarks[prev_frame_idx]:
                prev_frame_idx -= 1
            
            # Find next valid frame
            while next_frame_idx < n_frames and not landmarks[next_frame_idx]:
                next_frame_idx += 1
            
            # Interpolate if gap is not too large
            if (prev_frame_idx >= 0 and next_frame_idx < n_frames and 
                next_frame_idx - prev_frame_idx <= max_interpolation_gap):
                
                prev_landmarks = landmarks[prev_frame_idx]
                next_landmarks = landmarks[next_frame_idx]
                
                # Linear interpolation
                alpha = (frame_idx - prev_frame_idx) / (next_frame_idx - prev_frame_idx)
                interpolated = []
                
                for i in range(len(prev_landmarks)):
                    interp_landmark = {
                        'x': prev_landmarks[i]['x'] * (1 - alpha) + next_landmarks[i]['x'] * alpha,
                        'y': prev_landmarks[i]['y'] * (1 - alpha) + next_landmarks[i]['y'] * alpha,
                        'z': prev_landmarks[i]['z'] * (1 - alpha) + next_landmarks[i]['z'] * alpha,
                    }
                    interpolated.append(interp_landmark)
                
                result[frame_idx] = interpolated
    
    return result