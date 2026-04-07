# backend/shared/severity.py

"""
shared/severity.py
Re-exports of core enums and models originally defined in Agent 1.
Agent 3 and Agent 4 import from here — never modify Agent 1 code.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID


class SeverityLevel(str, Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class GeoPoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class SentinelZone(BaseModel):
    id: UUID
    name: str
    center: GeoPoint
    radius_km: float
    risk_level: SeverityLevel
    population_density: Optional[int] = None
    elevation: Optional[float] = None
    drainage_capacity: Optional[str] = None
