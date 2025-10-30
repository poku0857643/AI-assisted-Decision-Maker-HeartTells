from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class ECGRecordCreate(BaseModel):
    recorded_at: datetime
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    apple_watch_version: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class ECGRecordUpdate(BaseModel):
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    apple_watch_version: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    processed_data: Optional[Dict[str, Any]] = None
    classification: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    notes: Optional[str] = None
    is_processed: Optional[bool] = None

class ECGRecordResponse(BaseModel):
    id: int
    user_id: int
    recorded_at: datetime
    duration_seconds: Optional[float]
    sample_rate: Optional[int]
    apple_watch_version: Optional[str]
    raw_data: Optional[Dict[str, Any]]
    processed_data: Optional[Dict[str, Any]]
    classification: Optional[str]
    confidence_score: Optional[float]
    notes: Optional[str]
    is_processed: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ECGRecordList(BaseModel):
    total: int
    records: list[ECGRecordResponse]

class AppleWatchECGData(BaseModel):
    """Schema for Apple Watch HKElectrocardiogram data"""
    voltage_measurements: list[float]
    sampling_frequency: float
    classification: Optional[str] = None
    symptoms: Optional[list[str]] = None
    start_date: datetime
    end_date: datetime