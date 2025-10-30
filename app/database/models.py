from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Numeric, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    date_of_birth = Column(DateTime, nullable=True)
    gender = Column(String(10), nullable=True)
    google_id = Column(String(255), unique=True, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship to ECG records
    ecg_records = relationship("ECGRecord", back_populates="user")

class ECGRecord(Base):
    __tablename__ = "ecg_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False)
    duration_seconds = Column(Numeric(5, 2), nullable=True)
    sample_rate = Column(Integer, nullable=True)  # Hz
    apple_watch_version = Column(String(50), nullable=True)
    
    # Raw HKElectrocardiogram data from Apple Watch
    raw_data = Column(JSONB, nullable=True)
    
    # Processed analysis results
    processed_data = Column(JSONB, nullable=True)
    
    # Analysis results
    classification = Column(String(50), nullable=True)  # 'normal', 'atrial_fibrillation', etc.
    confidence_score = Column(Numeric(3, 2), nullable=True)  # 0.00 to 1.00
    
    # Additional metadata
    notes = Column(Text, nullable=True)
    is_processed = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship to user
    user = relationship("User", back_populates="ecg_records")