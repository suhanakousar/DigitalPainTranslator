"""
FastAPI router for admin endpoints (health, model management).
"""
from fastapi import APIRouter, HTTPException, Depends
import time
import psutil
import torch
from pathlib import Path
from typing import Dict, Any

from ..schemas import HealthResponse, ModelReloadRequest
from ..inference.predictor import PainAssessmentPredictor
from ..config import settings

router = APIRouter()

# Service start time for uptime calculation
service_start_time = time.time()


def get_predictor() -> PainAssessmentPredictor:
    """Dependency to get predictor instance."""
    # Import here to avoid circular import
    from .infer import predictor
    return predictor


@router.get("/health", response_model=HealthResponse)
async def health_check(
    predictor_instance: PainAssessmentPredictor = Depends(get_predictor)
) -> HealthResponse:
    """
    Comprehensive health check for the pain assessment service.
    
    Returns service status, model information, and uptime.
    """
    uptime_seconds = time.time() - service_start_time
    
    return HealthResponse(
        status="healthy",
        model_loaded=predictor_instance.is_ml_model_loaded,
        model_version=predictor_instance.current_model_version,
        uptime_seconds=uptime_seconds
    )


@router.get("/health/detailed")
async def detailed_health_check(
    predictor_instance: PainAssessmentPredictor = Depends(get_predictor)
) -> Dict[str, Any]:
    """
    Detailed health check with system resource information.
    
    Includes memory usage, CPU usage, and disk space information.
    """
    try:
        # System resource information
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('/')
        
        # GPU information if available
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_cached": torch.cuda.memory_reserved()
            }
        else:
            gpu_info = {"available": False}
        
        # Model file information
        model_files = {}
        if Path(settings.model_path).exists():
            model_stat = Path(settings.model_path).stat()
            model_files["ml_model"] = {
                "path": settings.model_path,
                "exists": True,
                "size_mb": model_stat.st_size / (1024 * 1024),
                "modified": model_stat.st_mtime
            }
        else:
            model_files["ml_model"] = {
                "path": settings.model_path,
                "exists": False
            }
        
        if Path(settings.baseline_weights_path).exists():
            baseline_stat = Path(settings.baseline_weights_path).stat()
            model_files["baseline_weights"] = {
                "path": settings.baseline_weights_path,
                "exists": True,
                "size_kb": baseline_stat.st_size / 1024,
                "modified": baseline_stat.st_mtime
            }
        else:
            model_files["baseline_weights"] = {
                "path": settings.baseline_weights_path,
                "exists": False
            }
        
        return {
            "service": {
                "status": "healthy",
                "uptime_seconds": time.time() - service_start_time,
                "version": "1.0.0"
            },
            "models": {
                "ml_model_loaded": predictor_instance.is_ml_model_loaded,
                "current_version": predictor_instance.current_model_version,
                "baseline_available": True,
                "files": model_files
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": memory_info.total / (1024**3),
                    "available_gb": memory_info.available / (1024**3),
                    "used_percent": memory_info.percent
                },
                "disk": {
                    "total_gb": disk_usage.total / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "used_percent": (disk_usage.used / disk_usage.total) * 100
                },
                "gpu": gpu_info
            },
            "configuration": {
                "max_inference_time_ms": settings.max_inference_time_ms,
                "confidence_threshold": settings.confidence_threshold,
                "allowed_origins": settings.allowed_origins
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get detailed health information: {str(e)}"
        )


@router.post("/model/reload")
async def reload_model(
    request: ModelReloadRequest,
    predictor_instance: PainAssessmentPredictor = Depends(get_predictor)
) -> Dict[str, Any]:
    """
    Reload the pain assessment model.
    
    Useful for updating the model without restarting the service.
    """
    try:
        model_path = request.model_path or settings.model_path
        
        # Check if model file exists
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {model_path}"
            )
        
        # Attempt to reload
        success = predictor_instance.reload_model(model_path)
        
        if success:
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "model_path": model_path,
                "model_version": predictor_instance.current_model_version,
                "ml_model_loaded": predictor_instance.is_ml_model_loaded
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to reload model - model file may be corrupted"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {str(e)}"
        )


@router.get("/model/info")
async def get_model_info(
    predictor_instance: PainAssessmentPredictor = Depends(get_predictor)
) -> Dict[str, Any]:
    """
    Get detailed information about the currently loaded model.
    """
    try:
        model_info = {
            "current_version": predictor_instance.current_model_version,
            "ml_model_loaded": predictor_instance.is_ml_model_loaded,
            "baseline_available": True,
            "model_type": "ml" if predictor_instance.is_ml_model_loaded else "baseline"
        }
        
        # Add file information
        if Path(settings.model_path).exists():
            model_stat = Path(settings.model_path).stat()
            model_info["ml_model_file"] = {
                "path": settings.model_path,
                "size_mb": model_stat.st_size / (1024 * 1024),
                "modified": model_stat.st_mtime
            }
        
        if Path(settings.baseline_weights_path).exists():
            baseline_stat = Path(settings.baseline_weights_path).stat()
            model_info["baseline_file"] = {
                "path": settings.baseline_weights_path,
                "size_kb": baseline_stat.st_size / 1024,
                "modified": baseline_stat.st_mtime
            }
        
        # Add model configuration
        model_info["configuration"] = {
            "input_features": settings.input_features,
            "hidden_size": settings.hidden_size,
            "num_layers": settings.num_layers,
            "dropout": settings.dropout,
            "sequence_length": settings.sequence_length
        }
        
        return model_info
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model information: {str(e)}"
        )


@router.post("/system/cleanup")
async def cleanup_system() -> Dict[str, Any]:
    """
    Perform system cleanup operations.
    
    Clears temporary files and optimizes memory usage.
    """
    try:
        cleanup_results = {
            "cache_cleared": False,
            "memory_optimized": False,
            "temp_files_removed": 0
        }
        
        # Clear PyTorch cache if GPU is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cleanup_results["cache_cleared"] = True
        
        # Force garbage collection
        import gc
        collected = gc.collect()
        cleanup_results["memory_optimized"] = True
        cleanup_results["objects_collected"] = collected
        
        # Clean up temporary files (implement based on your needs)
        temp_dir = Path("/tmp")
        if temp_dir.exists():
            temp_files = list(temp_dir.glob("pain_assessment_*"))
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    cleanup_results["temp_files_removed"] += 1
                except:
                    pass  # Ignore files that can't be removed
        
        return {
            "status": "success",
            "message": "System cleanup completed",
            "results": cleanup_results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"System cleanup failed: {str(e)}"
        )


@router.get("/metrics")
async def get_service_metrics() -> Dict[str, Any]:
    """
    Get performance metrics for the service.
    
    Includes request counts, response times, and error rates.
    """
    # This would typically integrate with a metrics collection system
    # For now, return basic placeholder metrics
    
    return {
        "requests": {
            "total": 0,  # Would be tracked in middleware
            "successful": 0,
            "failed": 0,
            "rate_per_minute": 0.0
        },
        "response_times": {
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0
        },
        "models": {
            "inference_count": 0,  # Would be tracked in predictor
            "avg_inference_time_ms": 0.0,
            "baseline_usage_percent": 100.0,
            "ml_usage_percent": 0.0
        },
        "errors": {
            "validation_errors": 0,
            "processing_errors": 0,
            "model_errors": 0
        }
    }