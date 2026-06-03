"""Pydantic models for API requests/responses."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response for file upload."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StatusResponse(BaseModel):
    """Response for job status check."""

    job_id: str
    status: str
    progress: int = Field(default=0, ge=0, le=100)
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DownloadResponse(BaseModel):
    """Response for download preparation."""

    job_id: str
    status: str
    clip_path: Optional[str] = None
    verdict_path: Optional[str] = None
    diagram_path: Optional[str] = None
    file_size_mb: Optional[float] = None


class VerdictData(BaseModel):
    """Verdict data from processing."""

    verdict: str = Field(..., description="OFFSIDE/ONSIDE/GOAL/NO-GOAL")
    confidence: float = Field(..., ge=0.0, le=1.0)
    analysis_data: Optional[Dict[str, Any]] = None


class JobRecord(BaseModel):
    """Job state record."""

    job_id: str
    video_filename: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    kaggle_kernel_id: Optional[str] = None
    kaggle_dataset_path: Optional[str] = None
    progress: int = Field(default=0, ge=0, le=100)
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    result_files: Optional[Dict[str, str]] = None
