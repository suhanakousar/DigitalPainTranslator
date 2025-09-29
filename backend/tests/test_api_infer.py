"""
Unit tests for API inference endpoints.
Tests use programmatically generated synthetic inputs.
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.schemas import FacialFeatures, CaregiverInputs, InferenceRequest


class TestInferenceAPI:
    """Test cases for inference API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_inference_request(self):
        """Generate synthetic inference request."""
        return {
            "features": {
                "mouthOpen": 0.3,
                "eyeClosureAvg": 0.2,
                "browFurrowAvg": 0.4,
                "headTiltVar": 0.1,
                "microMovementVar": 0.15
            },
            "caregiverInputs": {
                "grimace": 2,
                "breathing": 1,
                "restlessness": 3,
                "gestures": ["clench"]
            },
            "session_id": "test-session-123"
        }
    
    @pytest.fixture
    def sample_landmarks_request(self):
        """Generate synthetic landmarks request."""
        # Create minimal landmarks
        landmarks = []
        for i in range(468):
            landmarks.append({
                "x": (i % 10) / 10.0 - 0.5,
                "y": ((i // 10) % 10) / 10.0 - 0.5,
                "z": ((i // 100) % 5) / 10.0 - 0.2
            })
        
        return {
            "landmarks": [landmarks],  # Single frame
            "caregiverInputs": {
                "grimace": 2,
                "breathing": 1,
                "restlessness": 3,
                "gestures": ["clench"]
            }
        }
    
    def test_inference_endpoint_features(self, client, sample_inference_request):
        """Test inference endpoint with features."""
        response = client.post("/api/infer", json=sample_inference_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "session_id" in data
        assert "score" in data
        assert "confidence" in data
        assert "explanation" in data
        assert "recommendedActions" in data
        assert "model_version" in data
        assert "processing_ms" in data
        
        # Verify data types and ranges
        assert isinstance(data["score"], (int, float))
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["explanation"], list)
        assert isinstance(data["recommendedActions"], list)
        assert isinstance(data["model_version"], str)
        assert isinstance(data["processing_ms"], int)
        
        assert 0.0 <= data["score"] <= 10.0
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["processing_ms"] > 0
        
        # Check explanation structure
        for explanation in data["explanation"]:
            assert "signal" in explanation
            assert "contribution" in explanation
            assert "importance" in explanation
            assert 0.0 <= explanation["importance"] <= 1.0
    
    def test_inference_endpoint_landmarks(self, client, sample_landmarks_request):
        """Test inference endpoint with landmarks."""
        response = client.post("/api/infer", json=sample_landmarks_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify basic response structure
        assert "session_id" in data
        assert "score" in data
        assert "confidence" in data
        assert 0.0 <= data["score"] <= 10.0
        assert 0.0 <= data["confidence"] <= 1.0
    
    def test_inference_endpoint_validation_errors(self, client):
        """Test inference endpoint validation."""
        # Missing caregiverInputs
        invalid_request = {
            "features": {
                "mouthOpen": 0.3,
                "eyeClosureAvg": 0.2,
                "browFurrowAvg": 0.4,
                "headTiltVar": 0.1,
                "microMovementVar": 0.15
            }
        }
        
        response = client.post("/api/infer", json=invalid_request)
        assert response.status_code == 422  # Validation error
        
        # Missing features and landmarks
        invalid_request = {
            "caregiverInputs": {
                "grimace": 2,
                "breathing": 1,
                "restlessness": 3
            }
        }
        
        response = client.post("/api/infer", json=invalid_request)
        assert response.status_code == 400  # Bad request
        
        # Invalid feature values
        invalid_request = {
            "features": {
                "mouthOpen": 1.5,  # Out of range
                "eyeClosureAvg": -0.1,  # Out of range
                "browFurrowAvg": 0.4,
                "headTiltVar": 0.1,
                "microMovementVar": 0.15
            },
            "caregiverInputs": {
                "grimace": 2,
                "breathing": 1,
                "restlessness": 3
            }
        }
        
        response = client.post("/api/infer", json=invalid_request)
        assert response.status_code == 422
        
        # Invalid caregiver inputs
        invalid_request = {
            "features": {
                "mouthOpen": 0.3,
                "eyeClosureAvg": 0.2,
                "browFurrowAvg": 0.4,
                "headTiltVar": 0.1,
                "microMovementVar": 0.15
            },
            "caregiverInputs": {
                "grimace": 6,  # Out of range
                "breathing": -1,  # Out of range
                "restlessness": 3
            }
        }
        
        response = client.post("/api/infer", json=invalid_request)
        assert response.status_code == 422
    
    def test_batch_inference_endpoint(self, client, sample_inference_request):
        """Test batch inference endpoint."""
        # Create batch request
        batch_request = [sample_inference_request.copy() for _ in range(3)]
        for i, req in enumerate(batch_request):
            req["session_id"] = f"test-session-{i}"
        
        response = client.post("/api/infer/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return array of responses
        assert isinstance(data, list)
        assert len(data) == 3
        
        for i, inference_response in enumerate(data):
            assert inference_response["session_id"] == f"test-session-{i}"
            assert 0.0 <= inference_response["score"] <= 10.0
            assert 0.0 <= inference_response["confidence"] <= 1.0
    
    def test_batch_inference_size_limit(self, client, sample_inference_request):
        """Test batch inference size limit."""
        # Create oversized batch
        batch_request = [sample_inference_request.copy() for _ in range(101)]
        
        response = client.post("/api/infer/batch", json=batch_request)
        assert response.status_code == 400
        assert "Batch size limited" in response.json()["detail"]
    
    def test_inference_status_endpoint(self, client):
        """Test inference status endpoint."""
        response = client.get("/api/infer/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify status structure
        assert "status" in data
        assert "model_version" in data
        assert "ml_model_loaded" in data
        assert "baseline_available" in data
        assert "active_websocket_connections" in data
        assert "supported_inputs" in data
        assert "max_inference_time_ms" in data
        
        assert data["status"] == "active"
        assert isinstance(data["ml_model_loaded"], bool)
        assert data["baseline_available"] is True
        assert isinstance(data["active_websocket_connections"], int)
        assert "landmarks" in data["supported_inputs"]
        assert "features" in data["supported_inputs"]
    
    def test_session_id_persistence(self, client, sample_inference_request):
        """Test that session ID is preserved in response."""
        test_session_id = "custom-session-id-123"
        sample_inference_request["session_id"] = test_session_id
        
        response = client.post("/api/infer", json=sample_inference_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == test_session_id
    
    def test_inference_reproducibility(self, client, sample_inference_request):
        """Test that identical requests produce consistent results."""
        response1 = client.post("/api/infer", json=sample_inference_request)
        response2 = client.post("/api/infer", json=sample_inference_request)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Scores should be identical for deterministic baseline model
        assert data1["score"] == data2["score"]
        assert data1["confidence"] == data2["confidence"]
        assert len(data1["explanation"]) == len(data2["explanation"])
    
    def test_extreme_input_values(self, client):
        """Test with extreme but valid input values."""
        # Maximum values
        max_request = {
            "features": {
                "mouthOpen": 1.0,
                "eyeClosureAvg": 1.0,
                "browFurrowAvg": 1.0,
                "headTiltVar": 1.0,
                "microMovementVar": 1.0
            },
            "caregiverInputs": {
                "grimace": 5,
                "breathing": 5,
                "restlessness": 5,
                "gestures": ["clench", "point", "shake"]
            }
        }
        
        response = client.post("/api/infer", json=max_request)
        assert response.status_code == 200
        
        max_data = response.json()
        assert 0.0 <= max_data["score"] <= 10.0
        
        # Minimum values
        min_request = {
            "features": {
                "mouthOpen": 0.0,
                "eyeClosureAvg": 0.0,
                "browFurrowAvg": 0.0,
                "headTiltVar": 0.0,
                "microMovementVar": 0.0
            },
            "caregiverInputs": {
                "grimace": 0,
                "breathing": 0,
                "restlessness": 0,
                "gestures": []
            }
        }
        
        response = client.post("/api/infer", json=min_request)
        assert response.status_code == 200
        
        min_data = response.json()
        assert 0.0 <= min_data["score"] <= 10.0
        
        # Maximum should generally produce higher score
        assert max_data["score"] >= min_data["score"]
    
    def test_gesture_validation(self, client, sample_inference_request):
        """Test caregiver gesture validation."""
        # Valid gestures
        valid_request = sample_inference_request.copy()
        valid_request["caregiverInputs"]["gestures"] = ["clench", "point", "shake"]
        
        response = client.post("/api/infer", json=valid_request)
        assert response.status_code == 200
        
        # Invalid gesture
        invalid_request = sample_inference_request.copy()
        invalid_request["caregiverInputs"]["gestures"] = ["invalid_gesture"]
        
        response = client.post("/api/infer", json=invalid_request)
        assert response.status_code == 422
    
    def test_both_landmarks_and_features_error(self, client):
        """Test error when both landmarks and features are provided."""
        invalid_request = {
            "features": {
                "mouthOpen": 0.3,
                "eyeClosureAvg": 0.2,
                "browFurrowAvg": 0.4,
                "headTiltVar": 0.1,
                "microMovementVar": 0.15
            },
            "landmarks": [
                [{"x": 0.0, "y": 0.0, "z": 0.0}]
            ],
            "caregiverInputs": {
                "grimace": 2,
                "breathing": 1,
                "restlessness": 3
            }
        }
        
        response = client.post("/api/infer", json=invalid_request)
        assert response.status_code == 422
    
    def test_landmark_structure_validation(self, client):
        """Test landmark structure validation."""
        # Valid landmarks
        valid_landmarks = []
        for i in range(10):  # Minimal set
            valid_landmarks.append({
                "x": i / 10.0,
                "y": i / 10.0,
                "z": i / 10.0
            })
        
        valid_request = {
            "landmarks": [valid_landmarks],
            "caregiverInputs": {
                "grimace": 2,
                "breathing": 1,
                "restlessness": 3
            }
        }
        
        response = client.post("/api/infer", json=valid_request)
        # May fail due to insufficient landmarks, but structure should be valid
        assert response.status_code in [200, 400]  # Either success or processing error
        
        # Invalid landmark structure
        invalid_request = {
            "landmarks": [
                [{"x": 0.0, "y": 0.0}]  # Missing z coordinate
            ],
            "caregiverInputs": {
                "grimace": 2,
                "breathing": 1,
                "restlessness": 3
            }
        }
        
        response = client.post("/api/infer", json=invalid_request)
        assert response.status_code == 422
    
    def test_error_response_format(self, client):
        """Test that error responses have correct format."""
        # Trigger validation error
        invalid_request = {
            "features": {
                "mouthOpen": "invalid"  # Should be number
            }
        }
        
        response = client.post("/api/infer", json=invalid_request)
        assert response.status_code == 422
        
        # Check error response structure
        error_data = response.json()
        assert "detail" in error_data
    
    def test_processing_time_reasonable(self, client, sample_inference_request):
        """Test that processing time is reasonable."""
        response = client.post("/api/infer", json=sample_inference_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Processing time should be reasonable (< 5 seconds for baseline)
        assert data["processing_ms"] < 5000
        assert data["processing_ms"] > 0


class TestWebSocketEndpoints:
    """Test cases for WebSocket endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection establishment."""
        with client.websocket_connect("/ws/infer") as websocket:
            # Should receive welcome message
            data = websocket.receive_text()
            message = json.loads(data)
            
            assert message["type"] == "connection"
            assert "message" in message
            assert "timestamp" in message
    
    def test_websocket_ping_pong(self, client):
        """Test WebSocket ping/pong mechanism."""
        with client.websocket_connect("/ws/infer") as websocket:
            # Skip welcome message
            websocket.receive_text()
            
            # Send ping
            ping_message = {
                "type": "ping",
                "session_id": "test-session",
                "timestamp": "2023-01-01T00:00:00Z"
            }
            websocket.send_text(json.dumps(ping_message))
            
            # Should receive pong
            data = websocket.receive_text()
            message = json.loads(data)
            
            assert message["type"] == "pong"
            assert message["session_id"] == "test-session"
    
    def test_websocket_inference(self, client):
        """Test WebSocket inference."""
        with client.websocket_connect("/ws/infer") as websocket:
            # Skip welcome message
            websocket.receive_text()
            
            # Send inference request
            inference_message = {
                "type": "inference",
                "session_id": "test-session",
                "data": {
                    "features": {
                        "mouthOpen": 0.3,
                        "eyeClosureAvg": 0.2,
                        "browFurrowAvg": 0.4,
                        "headTiltVar": 0.1,
                        "microMovementVar": 0.15
                    },
                    "caregiverInputs": {
                        "grimace": 2,
                        "breathing": 1,
                        "restlessness": 3,
                        "gestures": ["clench"]
                    }
                }
            }
            websocket.send_text(json.dumps(inference_message))
            
            # Should receive inference result
            data = websocket.receive_text()
            message = json.loads(data)
            
            assert message["type"] == "inference_result"
            assert message["session_id"] == "test-session"
            assert "data" in message
            
            # Check inference result structure
            result = message["data"]
            assert "score" in result
            assert "confidence" in result
            assert 0.0 <= result["score"] <= 10.0
            assert 0.0 <= result["confidence"] <= 1.0
    
    def test_websocket_invalid_message(self, client):
        """Test WebSocket with invalid message."""
        with client.websocket_connect("/ws/infer") as websocket:
            # Skip welcome message
            websocket.receive_text()
            
            # Send invalid JSON
            websocket.send_text("invalid json")
            
            # Should receive error message
            data = websocket.receive_text()
            message = json.loads(data)
            
            assert "error" in message
            assert "message" in message