from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from app.database import get_db
from app.database.models import ECGRecord, User
from app.schemas.ecg import ECGRecordCreate, ECGRecordUpdate, ECGRecordResponse, ECGRecordList
from app.auth.dependencies import get_current_user

router = APIRouter(prefix="/ecg", tags=["ECG Records"])

@router.post("/records", response_model=ECGRecordResponse)
async def create_ecg_record(
    ecg_data: ECGRecordCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new ECG record for the current user"""
    db_ecg = ECGRecord(
        user_id=current_user.id,
        recorded_at=ecg_data.recorded_at,
        duration_seconds=ecg_data.duration_seconds,
        sample_rate=ecg_data.sample_rate,
        apple_watch_version=ecg_data.apple_watch_version,
        raw_data=ecg_data.raw_data,
        notes=ecg_data.notes
    )
    
    db.add(db_ecg)
    db.commit()
    db.refresh(db_ecg)
    
    return db_ecg

@router.get("/records", response_model=ECGRecordList)
async def get_user_ecg_records(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get ECG records for the current user with pagination"""
    total = db.query(ECGRecord).filter(ECGRecord.user_id == current_user.id).count()
    records = (
        db.query(ECGRecord)
        .filter(ECGRecord.user_id == current_user.id)
        .order_by(ECGRecord.recorded_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    return ECGRecordList(total=total, records=records)

@router.get("/records/{record_id}", response_model=ECGRecordResponse)
async def get_ecg_record(
    record_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific ECG record by ID"""
    ecg_record = (
        db.query(ECGRecord)
        .filter(ECGRecord.id == record_id, ECGRecord.user_id == current_user.id)
        .first()
    )
    
    if not ecg_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ECG record not found"
        )
    
    return ecg_record

@router.put("/records/{record_id}", response_model=ECGRecordResponse)
async def update_ecg_record(
    record_id: int,
    ecg_update: ECGRecordUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update an ECG record"""
    ecg_record = (
        db.query(ECGRecord)
        .filter(ECGRecord.id == record_id, ECGRecord.user_id == current_user.id)
        .first()
    )
    
    if not ecg_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ECG record not found"
        )
    
    update_data = ecg_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(ecg_record, field, value)
    
    db.commit()
    db.refresh(ecg_record)
    
    return ecg_record

@router.delete("/records/{record_id}")
async def delete_ecg_record(
    record_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an ECG record"""
    ecg_record = (
        db.query(ECGRecord)
        .filter(ECGRecord.id == record_id, ECGRecord.user_id == current_user.id)
        .first()
    )
    
    if not ecg_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ECG record not found"
        )
    
    db.delete(ecg_record)
    db.commit()
    
    return {"message": "ECG record deleted successfully"}

@router.get("/records/{record_id}/raw-data")
async def get_ecg_raw_data(
    record_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get raw ECG data for processing"""
    ecg_record = (
        db.query(ECGRecord)
        .filter(ECGRecord.id == record_id, ECGRecord.user_id == current_user.id)
        .first()
    )
    
    if not ecg_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ECG record not found"
        )
    
    if not ecg_record.raw_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No raw data available for this record"
        )
    
    return ecg_record.raw_data

@router.get("/stats")
async def get_ecg_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get ECG statistics for the current user"""
    total_records = db.query(ECGRecord).filter(ECGRecord.user_id == current_user.id).count()
    processed_records = (
        db.query(ECGRecord)
        .filter(ECGRecord.user_id == current_user.id, ECGRecord.is_processed == True)
        .count()
    )
    
    # Get classification breakdown
    classifications = (
        db.query(ECGRecord.classification, db.func.count(ECGRecord.classification))
        .filter(ECGRecord.user_id == current_user.id, ECGRecord.classification.isnot(None))
        .group_by(ECGRecord.classification)
        .all()
    )
    
    return {
        "total_records": total_records,
        "processed_records": processed_records,
        "unprocessed_records": total_records - processed_records,
        "classifications": dict(classifications)
    }