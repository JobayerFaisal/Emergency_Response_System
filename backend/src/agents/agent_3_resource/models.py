# backend/src/agents/agent_3_resource/models.py

"""
src/agents/agent_3_resource/models.py
All Pydantic models for Agent 3 Resource Management.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from uuid import UUID, uuid4
from datetime import datetime, timezone
from enum import Enum

from shared.severity import GeoPoint


class ResourceType(str, Enum):
    RESCUE_BOAT   = "rescue_boat"
    MEDICAL_TEAM  = "medical_team"
    MEDICAL_KIT   = "medical_kit"
    FOOD_SUPPLY   = "food_supply"
    WATER_SUPPLY  = "water_supply"


class ResourceStatus(str, Enum):
    AVAILABLE    = "available"
    DEPLOYED     = "deployed"
    RETURNING    = "returning"
    MAINTENANCE  = "maintenance"


class ResourceUnit(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    resource_type: ResourceType
    name: str
    status: ResourceStatus = ResourceStatus.AVAILABLE
    capacity: int
    current_location: GeoPoint
    base_location: GeoPoint
    assigned_zone: Optional[str] = None
    assigned_incident_id: Optional[str] = None
    deployed_at: Optional[datetime] = None


class InventorySnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resources: Dict[str, dict]   # {resource_type: {total, available, deployed}}


class ResourceAllocation(BaseModel):
    """What Agent 3 sends to Agent 4 via dispatch_order channel."""
    allocation_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    incident_id: str
    zone_id: str
    zone_name: str
    destination: GeoPoint
    priority: int
    urgency: str
    num_people_affected: int
    allocated_resources: List[dict]   # [{unit_id, type, name, current_location}]
    requires_medical: bool
    partial_allocation: bool = False
    notes: str = ""


class RestockRequest(BaseModel):
    resource_type: ResourceType
    quantity: int = Field(..., gt=0)
    location: GeoPoint
    notes: Optional[str] = None


# ── Allocation rules per urgency level ────────────────────────────────────────
ALLOCATION_RULES = {
    "LIFE_THREATENING": {
        ResourceType.RESCUE_BOAT:  2,
        ResourceType.MEDICAL_TEAM: 1,
        ResourceType.MEDICAL_KIT:  2,
        ResourceType.FOOD_SUPPLY:  0,
        ResourceType.WATER_SUPPLY: 0,
    },
    "URGENT_MEDICAL": {
        ResourceType.RESCUE_BOAT:  1,
        ResourceType.MEDICAL_TEAM: 1,
        ResourceType.MEDICAL_KIT:  1,
        ResourceType.FOOD_SUPPLY:  0,
        ResourceType.WATER_SUPPLY: 0,
    },
    "URGENT": {
        ResourceType.RESCUE_BOAT:  1,
        ResourceType.MEDICAL_TEAM: 0,
        ResourceType.MEDICAL_KIT:  0,
        ResourceType.FOOD_SUPPLY:  1,
        ResourceType.WATER_SUPPLY: 1,
    },
    "MODERATE": {
        ResourceType.RESCUE_BOAT:  0,
        ResourceType.MEDICAL_TEAM: 0,
        ResourceType.MEDICAL_KIT:  0,
        ResourceType.FOOD_SUPPLY:  1,
        ResourceType.WATER_SUPPLY: 1,
    },
}
