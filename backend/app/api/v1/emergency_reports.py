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
    """Return all extracted emergency reports."""
    reports = db.query(EmergencyReport).order_by(EmergencyReport.timestamp.desc()).all()

    output = []
    for r in reports:
        output.append({
            "id": r.id,
            "responder_id": r.responder_id,
            "raw_message": r.raw_message,
            "people": json.loads(r.people) if r.people else None,
            "needs": json.loads(r.needs) if r.needs else None,
            "hazards": json.loads(r.hazards) if r.hazards else [],
            "urgency": r.urgency,
            "confidence": json.loads(r.confidence) if r.confidence else None,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None
        })

    return output
