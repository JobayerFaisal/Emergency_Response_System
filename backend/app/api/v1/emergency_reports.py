# backend/app/api/v1/emergency_reports.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.db import SessionLocal
from app.agents.agent_2_responder_chat.repository import EmergencyReport
import json

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def list_reports(db: Session = Depends(get_db)):
    reports = db.query(EmergencyReport).order_by(EmergencyReport.timestamp.desc()).all()

    output = []
    for r in reports:

        timestamp_value = r.timestamp
        output.append({
            "id": r.id,
            "responder_id": r.responder_id,
            "raw_message": r.raw_message,

            "people": json.loads(r.people) if isinstance(r.people, str) else None,
            "needs": json.loads(r.needs) if isinstance(r.needs, str) else None,
            "hazards": json.loads(r.hazards) if isinstance(r.hazards, str) else [],

            # "people": json.loads(r.people) if r.people else None,
            # "needs": json.loads(r.needs) if r.needs else None,
            # "hazards": json.loads(r.hazards) if r.hazards else [],

            "urgency": r.urgency,
            "confidence": r.confidence,  # ← FLOAT directly returned


            "timestamp": timestamp_value.isoformat() if timestamp_value is not None else None

            # "timestamp": r.timestamp.isoformat() if r.timestamp else None
        })

    return output
