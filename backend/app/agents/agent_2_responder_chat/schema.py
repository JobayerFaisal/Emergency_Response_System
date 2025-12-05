# path: backend/app/agents/agent_2_responder_chat/schema.py

from pydantic import BaseModel
from typing import Optional, Dict, List

class DistressUpdate(BaseModel):
    responder_id: str
    raw_message: str
    
    people: Optional[Dict] = None
    needs: Optional[Dict] = None
    hazards: Optional[List[str]] = []
    urgency: Optional[str] = None
    confidence: Optional[float] = None
