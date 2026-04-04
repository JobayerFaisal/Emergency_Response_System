"""
src/agents/agent_4_dispatch/models.py
All Pydantic models for Agent 4 Dispatch Optimization.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime, timezone
from enum import Enum

from shared.severity import GeoPoint


class TransportMode(str, Enum):
    ROAD      = "road"       # Medical teams, supply trucks → OSRM routing
    WATERWAY  = "waterway"   # Rescue boats → straight-line + speed estimate


class TeamStatus(str, Enum):
    DISPATCHED  = "dispatched"
    EN_ROUTE    = "en_route"
    ARRIVED     = "arrived"
    RETURNING   = "returning"


class TeamAssignment(BaseModel):
    """A single team/resource assigned to a route."""
    unit_id: UUID
    unit_name: str
    resource_type: str
    transport_mode: TransportMode
    origin: GeoPoint
    destination: GeoPoint
    route_geometry: Optional[dict] = None   # GeoJSON LineString
    distance_km: float
    eta_minutes: float
    status: TeamStatus = TeamStatus.DISPATCHED
    route_safety_score: float = 1.0         # 0.0 (unsafe) – 1.0 (safe)


class RouteAssignment(BaseModel):
    """Complete route plan for one incident — published to route_assignment channel."""
    assignment_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    incident_id: str
    allocation_id: str
    zone_id: str
    zone_name: str
    destination: GeoPoint
    priority: int
    urgency: str
    teams: List[TeamAssignment]
    total_eta_minutes: float        # Longest ETA among all teams
    route_safety_score: float       # Minimum safety score across all team routes
    partial_allocation: bool = False
    notes: str = ""


class TeamStatusUpdate(BaseModel):
    status: TeamStatus
    notes: Optional[str] = None


# Boat speed table (km/h) — Bangladesh river/flood conditions
BOAT_SPEEDS = {
    "normal_river":  15.0,
    "flood_shallow": 10.0,   # depth < 1m
    "flood_deep":     8.0,   # depth 1–2m
    "flood_severe":   5.0,   # depth > 2m or strong current
}

# Default road speed fallback if OSRM unavailable (km/h)
ROAD_SPEED_FALLBACK_KMH = 30.0
