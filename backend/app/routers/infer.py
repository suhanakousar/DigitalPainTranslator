"""
FastAPI router for inference endpoints (REST and WebSocket).
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import JSONResponse
import json
import asyncio
from typing import Dict, Any

from ..schemas import (
    InferenceRequest, InferenceResponse, ErrorResponse
)
from ..inference.predictor import PainAssessmentPredictor
from ..config import settings
from ..utils import Timer

# Initialize predictor instance
predictor = PainAssessmentPredictor(
    model_path=settings.model_path,
    baseline_weights_path=settings.baseline_weights_path
)

router = APIRouter()
ws_router = APIRouter()


def get_predictor() -> PainAssessmentPredictor:
    """Dependency to get predictor instance."""
    return predictor


@router.post("/infer", response_model=InferenceResponse)
async def infer_pain_score(
    request: InferenceRequest,
    predictor_instance: PainAssessmentPredictor = Depends(get_predictor)
) -> InferenceResponse:
    """
    Perform pain assessment inference on provided data.
    
    Accepts either facial landmarks or precomputed features along with caregiver inputs.
    Returns pain score, confidence, and explainability information.
    """
    try:
        with Timer() as timer:
            # Validate request
            if not request.landmarks and not request.features:
                raise HTTPException(
                    status_code=400,
                    detail="Either landmarks or features must be provided"
                )
            
            # Perform inference
            response = predictor_instance.predict(request)
            
            return response
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


class WebSocketManager:
    """Manage WebSocket connections for real-time inference."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"WebSocket disconnected: {session_id}")
    
    async def send_response(self, session_id: str, response: dict):
        """Send response to specific WebSocket."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(response))
            except Exception as e:
                print(f"Failed to send WebSocket message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def send_error(self, session_id: str, error: str, message: str):
        """Send error message to WebSocket."""
        error_response = {
            "error": error,
            "message": message
        }
        await self.send_response(session_id, error_response)


# WebSocket manager instance
ws_manager = WebSocketManager()


@ws_router.websocket("/ws/infer")
async def websocket_inference(websocket: WebSocket):
    """
    WebSocket endpoint for real-time pain assessment inference.
    
    Accepts the same JSON messages as the REST endpoint and returns
    equivalent response messages for low-latency inference.
    """
    session_id = None
    
    try:
        # Accept connection
        await websocket.accept()
        
        # Send welcome message
        welcome_msg = {
            "type": "connection",
            "message": "Connected to pain assessment inference WebSocket",
            "timestamp": str(asyncio.get_event_loop().time())
        }
        await websocket.send_text(json.dumps(welcome_msg))
        
        while True:
            # Receive message
            try:
                raw_message = await websocket.receive_text()
                message_data = json.loads(raw_message)
                
                # Extract session ID if provided
                session_id = message_data.get("session_id", "unknown")
                
                # Add to active connections
                ws_manager.active_connections[session_id] = websocket
                
                # Validate message type
                if message_data.get("type") == "inference":
                    await handle_inference_message(websocket, message_data, session_id)
                elif message_data.get("type") == "ping":
                    await handle_ping_message(websocket, session_id)
                else:
                    await ws_manager.send_error(
                        session_id,
                        "invalid_message_type",
                        f"Unknown message type: {message_data.get('type')}"
                    )
                    
            except json.JSONDecodeError:
                await ws_manager.send_error(
                    session_id or "unknown",
                    "invalid_json",
                    "Invalid JSON message format"
                )
            except Exception as e:
                await ws_manager.send_error(
                    session_id or "unknown",
                    "processing_error",
                    f"Error processing message: {str(e)}"
                )
                
    except WebSocketDisconnect:
        if session_id:
            ws_manager.disconnect(session_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if session_id:
            ws_manager.disconnect(session_id)


async def handle_inference_message(websocket: WebSocket, message_data: dict, session_id: str):
    """Handle inference request via WebSocket."""
    try:
        # Extract inference data
        inference_data = message_data.get("data", {})
        
        # Create inference request
        request = InferenceRequest(**inference_data)
        request.session_id = session_id
        
        # Perform inference
        with Timer() as timer:
            response = predictor.predict(request)
        
        # Send response
        response_data = {
            "type": "inference_result",
            "session_id": session_id,
            "data": response.dict(),
            "timestamp": str(asyncio.get_event_loop().time())
        }
        
        await websocket.send_text(json.dumps(response_data))
        
    except Exception as e:
        await ws_manager.send_error(
            session_id,
            "inference_error",
            f"Inference failed: {str(e)}"
        )


async def handle_ping_message(websocket: WebSocket, session_id: str):
    """Handle ping message for connection health check."""
    pong_response = {
        "type": "pong",
        "session_id": session_id,
        "timestamp": str(asyncio.get_event_loop().time())
    }
    
    await websocket.send_text(json.dumps(pong_response))


@router.get("/infer/status")
async def get_inference_status(
    predictor_instance: PainAssessmentPredictor = Depends(get_predictor)
) -> Dict[str, Any]:
    """
    Get current inference service status.
    
    Returns information about loaded models and service health.
    """
    return {
        "status": "active",
        "model_version": predictor_instance.current_model_version,
        "ml_model_loaded": predictor_instance.is_ml_model_loaded,
        "baseline_available": True,
        "active_websocket_connections": len(ws_manager.active_connections),
        "supported_inputs": ["landmarks", "features"],
        "max_inference_time_ms": settings.max_inference_time_ms
    }


@router.post("/infer/batch")
async def batch_inference(
    requests: list[InferenceRequest],
    predictor_instance: PainAssessmentPredictor = Depends(get_predictor)
) -> list[InferenceResponse]:
    """
    Perform batch inference on multiple requests.
    
    Useful for processing multiple assessments efficiently.
    """
    if len(requests) > 100:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 requests"
        )
    
    responses = []
    
    try:
        for request in requests:
            response = predictor_instance.predict(request)
            responses.append(response)
        
        return responses
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch inference failed: {str(e)}"
        )