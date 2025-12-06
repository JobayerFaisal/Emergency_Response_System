# backend/app/agents/agent_2_responder_chat/schema.py

from pydantic import BaseModel
from typing import Optional, List, Any

class DistressUpdate(BaseModel):
    responder_id: str
    raw_message: str

    people: Optional[Any] = None
    needs: Optional[Any] = None
    hazards: Optional[List[str]] = None  # ← FIXED (no mutable default)
    urgency: Optional[str] = None
    confidence: Optional[Any] = None
