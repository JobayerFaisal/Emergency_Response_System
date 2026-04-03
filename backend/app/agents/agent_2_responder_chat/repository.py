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

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class EmergencyReport(Base):
    __tablename__ = "emergency_reports"

    id = Column(Integer, primary_key=True, index=True)
    responder_id = Column(String, index=True)
    raw_message = Column(Text)

    # Existing fields
    people = Column(Text)        # JSON string
    needs = Column(Text)         # JSON string
    hazards = Column(Text)       # JSON string
    urgency = Column(String)
    confidence = Column(Float)

    # NEW FIELDS FOR AGENT-2
    team_status = Column(Text, nullable=True)         # string
    supply_request = Column(Text, nullable=True)       # JSON list
    mobility_issues = Column(Text, nullable=True)      # JSON list
    rescue_progress = Column(Text, nullable=True)      # string
    medical_needs = Column(Text, nullable=True)        # JSON list

    timestamp = Column(DateTime(timezone=True), server_default=func.now())


def save_emergency_report(db, report):

    db_obj = EmergencyReport(
        responder_id=report.responder_id,
        raw_message=report.raw_message,

        # Existing fields
        people=json.dumps(report.people),
        needs=json.dumps(report.needs),
        hazards=json.dumps(report.hazards),
        urgency=report.urgency,
        confidence=report.confidence,

        # NEW fields
        team_status=report.team_status,
        supply_request=json.dumps(report.supply_request) if report.supply_request else None,
        mobility_issues=json.dumps(report.mobility_issues) if report.mobility_issues else None,
        rescue_progress=report.rescue_progress,
        medical_needs=json.dumps(report.medical_needs) if report.medical_needs else None,
    )

    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj
