"""
Pydantic schemas for request and response validation.
"""
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator
import uuid


class Landmark(BaseModel):
    """Single facial landmark coordinates."""
    x: float = Field(..., ge=-1.0, le=1.0, description="Normalized X coordinate")
    y: float = Field(..., ge=-1.0, le=1.0, description="Normalized Y coordinate")
    z: float = Field(..., ge=-1.0, le=1.0, description="Normalized Z coordinate")


class FacialFeatures(BaseModel):
    """Computed facial features for pain assessment."""
    mouthOpen: float = Field(..., ge=0.0, le=1.0, description="Mouth openness measure")
    eyeClosureAvg: float = Field(..., ge=0.0, le=1.0, description="Average eye closure")
    browFurrowAvg: float = Field(..., ge=0.0, le=1.0, description="Average brow furrow intensity")
    headTiltVar: float = Field(..., ge=0.0, le=1.0, description="Head tilt variation")
    microMovementVar: float = Field(..., ge=0.0, le=1.0, description="Micro-movement variation")


class CaregiverInputs(BaseModel):
    """Caregiver assessment inputs."""
    grimace: int = Field(..., ge=0, le=5, description="Grimace score")
    breathing: int = Field(..., ge=0, le=5, description="Breathing pattern score")
    restlessness: int = Field(..., ge=0, le=5, description="Restlessness score")
    gestures: List[str] = Field(default=[], description="Observed gestures")
    
    @validator("gestures")
    def validate_gestures(cls, v):
        allowed_gestures = {"clench", "point", "shake"}
        for gesture in v:
            if gesture not in allowed_gestures:
                raise ValueError(f"Invalid gesture: {gesture}")
        return v


class InferenceRequest(BaseModel):
    """Request for pain assessment inference."""
    # Either landmarks OR features must be provided
    landmarks: Optional[List[List[Landmark]]] = Field(None, description="Sequence of facial landmarks")
    features: Optional[FacialFeatures] = Field(None, description="Precomputed facial features")
    
    # Required caregiver inputs
    caregiverInputs: CaregiverInputs = Field(..., description="Caregiver assessment")
    
    # Optional metadata
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: Optional[str] = Field(None, description="ISO timestamp")
    
    @validator("features", always=True)
    def validate_input_data(cls, v, values):
        landmarks = values.get("landmarks")
        if not landmarks and not v:
            raise ValueError("Either landmarks or features must be provided")
        if landmarks and v:
            raise ValueError("Provide either landmarks OR features, not both")
        return v


class ExplanationItem(BaseModel):
    """Individual explanation component."""
    signal: str = Field(..., description="Feature or signal name")
    contribution: float = Field(..., description="Contribution to final score")
    importance: float = Field(..., ge=0.0, le=1.0, description="Normalized importance")


class InferenceResponse(BaseModel):
    """Response from pain assessment inference."""
    session_id: str = Field(..., description="Session identifier")
    score: float = Field(..., ge=0.0, le=10.0, description="Pain assessment score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    explanation: List[ExplanationItem] = Field(..., description="Feature contributions")
    recommendedActions: List[str] = Field(default=[], description="Recommended actions")
    model_version: str = Field(..., description="Model version used")
    processing_ms: int = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Current model version")
    uptime_seconds: float = Field(..., description="Service uptime")


class RecordCreateRequest(BaseModel):
    """Request to create a new assessment record."""
    inference_request: InferenceRequest = Field(..., description="Original inference request")
    inference_response: InferenceResponse = Field(..., description="Inference result")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")


class RecordResponse(BaseModel):
    """Assessment record response."""
    id: str = Field(..., description="Record identifier")
    created_at: str = Field(..., description="Creation timestamp")
    inference_request: InferenceRequest = Field(..., description="Original request")
    inference_response: InferenceResponse = Field(..., description="Inference result")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class RecordsListResponse(BaseModel):
    """List of assessment records."""
    records: List[RecordResponse] = Field(..., description="Assessment records")
    total: int = Field(..., description="Total number of records")
    page: int = Field(default=1, description="Current page")
    limit: int = Field(default=100, description="Records per page")


class ModelReloadRequest(BaseModel):
    """Request to reload the model."""
    model_path: Optional[str] = Field(None, description="Path to new model file")
    force: bool = Field(default=False, description="Force reload even if path unchanged")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")