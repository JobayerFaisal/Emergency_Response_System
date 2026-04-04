"""
src/agents/agent_2_distress/models.py
All Pydantic models for Agent 2 — Distress Intelligence.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime, timezone
from enum import Enum


class UrgencyLevel(str, Enum):
    LIFE_THREATENING = "LIFE_THREATENING"   # People trapped, immediate danger
    URGENT_MEDICAL   = "URGENT_MEDICAL"     # Medical emergency
    URGENT           = "URGENT"             # Significant flooding, needs help soon
    MODERATE         = "MODERATE"           # Flooding present, not immediate danger
    LOW              = "LOW"                # Minor flooding, monitoring needed


class IncidentSource(str, Enum):
    FLOOD_ALERT          = "flood_alert"           # From Agent 1 prediction
    SOCIAL_MEDIA_BANGLA  = "social_media_bangla"   # Bengali text
    SOCIAL_MEDIA_BANGLISH = "social_media_banglish" # Romanized Bengali
    SOCIAL_MEDIA_ENGLISH = "social_media_english"  # English text
    OFFICIAL_REPORT      = "official_report"       # Govt/NGO reports


class FloodAlert(BaseModel):
    """What Agent 1 sends on the flood_alert channel."""
    zone_id: str
    zone_name: str
    risk_score: float           # 0.0 – 1.0
    severity_level: str         # minimal / moderate / high / critical
    confidence: float           # 0.0 – 1.0
    risk_factors: dict
    timestamp: str


class DistressIncident(BaseModel):
    """
    A single actionable distress incident.
    This is what Agent 2 publishes to distress_queue for Agent 3 to consume.
    Matches exactly what Agent 3's allocator.py expects.
    """
    incident_id: str = Field(default_factory=lambda: f"INC-{str(uuid4())[:8].upper()}")
    zone_id: str
    zone_name: str
    raw_message: str = ""
    raw_location: str = ""
    latitude: float
    longitude: float
    urgency: UrgencyLevel
    num_people: int = 0
    medical_need: bool = False
    priority: int = Field(3, ge=1, le=5)    # 1=lowest 5=highest
    source: IncidentSource = IncidentSource.FLOOD_ALERT
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProcessedAlert(BaseModel):
    """Internal model — Agent 2's enriched version of a flood alert."""
    alert: FloodAlert
    incidents: List[DistressIncident]
    processing_notes: str = ""
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Urgency mapping from Agent 1 severity levels ─────────────────────────────
SEVERITY_TO_URGENCY = {
    "critical": UrgencyLevel.LIFE_THREATENING,
    "high":     UrgencyLevel.URGENT,
    "moderate": UrgencyLevel.MODERATE,
    "low":      UrgencyLevel.LOW,
    "minimal":  UrgencyLevel.LOW,
}

SEVERITY_TO_PRIORITY = {
    "critical": 5,
    "high":     4,
    "moderate": 3,
    "low":      2,
    "minimal":  1,
}
