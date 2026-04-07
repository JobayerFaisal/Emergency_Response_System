# backend/shared/message_protocol.py

"""
shared/message_protocol.py
Standard Redis message envelope used by ALL agents.
Agent 1 & 2 must use this same format when publishing.
"""

from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime, timezone


class AgentMessage(BaseModel):
    """Standard envelope for ALL Redis pub/sub messages"""
    message_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sender_agent: str          # "agent_1_environmental", "agent_2_distress", etc.
    receiver_agent: str        # "agent_3_resource", "agent_4_dispatch", "all"
    message_type: str          # "flood_alert", "distress_queue", "dispatch_order", etc.
    zone_id: Optional[str] = None
    priority: int = Field(3, ge=1, le=5)   # 1=lowest, 5=highest
    payload: dict              # Actual data — varies per message type
