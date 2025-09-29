"""
FastAPI router for assessment record storage and retrieval.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import json
import uuid
from datetime import datetime
from pathlib import Path

from ..schemas import (
    RecordCreateRequest, RecordResponse, RecordsListResponse,
    InferenceRequest, InferenceResponse
)
from ..config import settings
from ..utils import load_json_file, save_json_file

router = APIRouter()


class RecordsManager:
    """Manage assessment records storage and retrieval."""
    
    def __init__(self, db_path: str):
        """Initialize records manager with database path."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty database if it doesn't exist
        if not self.db_path.exists():
            self._initialize_database()
    
    def _initialize_database(self):
        """Initialize empty records database."""
        empty_db = {
            "records": [],
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "total_records": 0
            }
        }
        save_json_file(empty_db, str(self.db_path))
    
    def _load_database(self) -> dict:
        """Load records database from file."""
        data = load_json_file(str(self.db_path))
        if not data or "records" not in data:
            self._initialize_database()
            return load_json_file(str(self.db_path))
        return data
    
    def _save_database(self, data: dict) -> bool:
        """Save records database to file."""
        data["metadata"]["last_updated"] = datetime.utcnow().isoformat()
        return save_json_file(data, str(self.db_path))
    
    def create_record(self, request: RecordCreateRequest) -> RecordResponse:
        """Create a new assessment record."""
        # Load current database
        db_data = self._load_database()
        
        # Create new record
        record_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        record_data = {
            "id": record_id,
            "created_at": created_at,
            "inference_request": request.inference_request.dict(),
            "inference_response": request.inference_response.dict(),
            "metadata": request.metadata or {}
        }
        
        # Add to database
        db_data["records"].append(record_data)
        db_data["metadata"]["total_records"] = len(db_data["records"])
        
        # Save database
        success = self._save_database(db_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save record")
        
        # Return record response
        return RecordResponse(**record_data)
    
    def get_records(
        self,
        page: int = 1,
        limit: int = 100,
        session_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> RecordsListResponse:
        """Retrieve assessment records with filtering and pagination."""
        # Load database
        db_data = self._load_database()
        records = db_data["records"]
        
        # Apply filters
        filtered_records = []
        for record in records:
            # Session ID filter
            if session_id:
                if record["inference_response"].get("session_id") != session_id:
                    continue
            
            # Date range filter
            if start_date or end_date:
                record_date = record["created_at"]
                if start_date and record_date < start_date:
                    continue
                if end_date and record_date > end_date:
                    continue
            
            filtered_records.append(record)
        
        # Apply pagination
        total_records = len(filtered_records)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_records = filtered_records[start_idx:end_idx]
        
        # Convert to response objects
        record_responses = [RecordResponse(**record) for record in paginated_records]
        
        return RecordsListResponse(
            records=record_responses,
            total=total_records,
            page=page,
            limit=limit
        )
    
    def get_record_by_id(self, record_id: str) -> Optional[RecordResponse]:
        """Get a specific record by ID."""
        db_data = self._load_database()
        
        for record in db_data["records"]:
            if record["id"] == record_id:
                return RecordResponse(**record)
        
        return None
    
    def delete_record(self, record_id: str) -> bool:
        """Delete a record by ID."""
        db_data = self._load_database()
        
        # Find and remove record
        original_count = len(db_data["records"])
        db_data["records"] = [r for r in db_data["records"] if r["id"] != record_id]
        
        if len(db_data["records"]) == original_count:
            return False  # Record not found
        
        # Update metadata
        db_data["metadata"]["total_records"] = len(db_data["records"])
        
        # Save database
        return self._save_database(db_data)
    
    def get_statistics(self) -> dict:
        """Get database statistics."""
        db_data = self._load_database()
        records = db_data["records"]
        
        if not records:
            return {
                "total_records": 0,
                "date_range": None,
                "avg_pain_score": 0.0,
                "score_distribution": {}
            }
        
        # Calculate statistics
        pain_scores = []
        dates = []
        
        for record in records:
            pain_score = record["inference_response"].get("score", 0)
            pain_scores.append(pain_score)
            dates.append(record["created_at"])
        
        # Score distribution
        score_bins = {
            "0-2": 0,  # Minimal pain
            "3-4": 0,  # Mild pain
            "5-6": 0,  # Moderate pain
            "7-8": 0,  # Severe pain
            "9-10": 0  # Extreme pain
        }
        
        for score in pain_scores:
            if score <= 2:
                score_bins["0-2"] += 1
            elif score <= 4:
                score_bins["3-4"] += 1
            elif score <= 6:
                score_bins["5-6"] += 1
            elif score <= 8:
                score_bins["7-8"] += 1
            else:
                score_bins["9-10"] += 1
        
        return {
            "total_records": len(records),
            "date_range": {
                "earliest": min(dates) if dates else None,
                "latest": max(dates) if dates else None
            },
            "avg_pain_score": sum(pain_scores) / len(pain_scores) if pain_scores else 0.0,
            "score_distribution": score_bins
        }


# Initialize records manager
records_manager = RecordsManager(settings.records_db_path)


@router.post("/records", response_model=RecordResponse)
async def create_assessment_record(request: RecordCreateRequest) -> RecordResponse:
    """
    Store a new pain assessment record.
    
    Records include the original inference request, response, and optional metadata.
    """
    try:
        return records_manager.create_record(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create record: {str(e)}")


@router.get("/records", response_model=RecordsListResponse)
async def get_assessment_records(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(100, ge=1, le=1000, description="Records per page"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    start_date: Optional[str] = Query(None, description="Filter by start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="Filter by end date (ISO format)")
) -> RecordsListResponse:
    """
    Retrieve assessment records with optional filtering and pagination.
    
    Supports filtering by session ID and date range.
    """
    try:
        return records_manager.get_records(
            page=page,
            limit=limit,
            session_id=session_id,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve records: {str(e)}")


@router.get("/records/{record_id}", response_model=RecordResponse)
async def get_assessment_record(record_id: str) -> RecordResponse:
    """Get a specific assessment record by ID."""
    record = records_manager.get_record_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@router.delete("/records/{record_id}")
async def delete_assessment_record(record_id: str) -> dict:
    """Delete a specific assessment record by ID."""
    success = records_manager.delete_record(record_id)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    
    return {"message": "Record deleted successfully", "record_id": record_id}


@router.get("/records/stats/summary")
async def get_records_statistics() -> dict:
    """
    Get summary statistics for assessment records.
    
    Includes total count, date range, average pain scores, and score distribution.
    """
    try:
        return records_manager.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post("/records/export")
async def export_records(
    session_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> dict:
    """
    Export assessment records as JSON.
    
    Returns all records matching the specified filters.
    """
    try:
        # Get all matching records without pagination
        records_response = records_manager.get_records(
            page=1,
            limit=10000,  # Large limit to get all records
            session_id=session_id,
            start_date=start_date,
            end_date=end_date
        )
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "filters": {
                "session_id": session_id,
                "start_date": start_date,
                "end_date": end_date
            },
            "total_records": records_response.total,
            "records": [record.dict() for record in records_response.records]
        }
        
        return export_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export records: {str(e)}")