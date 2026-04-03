from pydantic import BaseModel
from typing import Optional, List, Any

class DistressUpdate(BaseModel):
    responder_id: str
    raw_message: str

    people: Optional[Any] = None
    needs: Optional[Any] = None
    hazards: Optional[List[str]] = None
    urgency: Optional[str] = None
    confidence: Optional[Any] = None

    # -----------------------
    # NEW FIELDS FOR AGENT-2
    # -----------------------
    team_status: Optional[str] = None
    supply_request: Optional[List[str]] = None
    mobility_issues: Optional[List[str]] = None
    rescue_progress: Optional[str] = None
    medical_needs: Optional[List[str]] = None
