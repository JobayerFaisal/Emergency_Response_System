# backend/app/agents/agent_2_responder_chat/schema.py

from pydantic import BaseModel
from typing import Optional, List

class DistressUpdate(BaseModel):
    responder_id: str
    raw_message: str

    people: Optional[object] = None
    needs: Optional[object] = None
    hazards: Optional[List[str]] = []
    urgency: Optional[str] = None
    confidence: Optional[object] = None
