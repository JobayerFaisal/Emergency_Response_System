# backend/app/agents/agent_2_responder_chat/repository.py

import json
from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.sql import func
from app.core.db import Base

class ChatMessage(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    responder_id = Column(String, index=True)
    role = Column(String)
    message = Column(Text)

    latitude = Column(Float, nullable=True)   # NEW
    longitude = Column(Float, nullable=True)  # NEW

    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class EmergencyReport(Base):
    __tablename__ = "emergency_reports"

    id = Column(Integer, primary_key=True, index=True)
    responder_id = Column(String, index=True)
    raw_message = Column(Text)

    people = Column(Text)     # JSON string
    needs = Column(Text)      # JSON string
    hazards = Column(Text)    # JSON string
    urgency = Column(String)
    confidence = Column(Float)  # ← FIXED (no JSON dump)

    timestamp = Column(DateTime(timezone=True), server_default=func.now())


def save_emergency_report(db, report):
    db_obj = EmergencyReport(
        responder_id=report.responder_id,
        raw_message=report.raw_message,

        people=json.dumps(report.people),
        needs=json.dumps(report.needs),
        hazards=json.dumps(report.hazards),

        urgency=report.urgency,
        confidence=report.confidence  # ← FLOAT stored directly
    )

    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

