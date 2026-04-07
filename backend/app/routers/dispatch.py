# backend/app/routers/dispatch.py
"""
backend/app/routers/dispatch.py
==================================
GET  /api/dispatch              → active dispatch_routes + team_routes
GET  /api/dispatch/{route_id}   → single route with full team list
GET  /api/dispatch/geojson      → GeoJSON FeatureCollection of all active routes
POST /api/dispatch              → manually trigger a dispatch order (→ Agent 4)
POST /api/dispatch/{route_id}/complete → mark a dispatch as completed
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import asyncpg
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.db import get_db
from app.services.redis_bridge import publish

logger = logging.getLogger("dashboard.routers.dispatch")

router = APIRouter(prefix="/api/dispatch", tags=["dispatch"])

AGENT4_URL = os.getenv("AGENT4_URL", "http://localhost:8004")


# ── GET /api/dispatch ─────────────────────────────────────────────────────────

@router.get("")
async def list_dispatches(
    status: str = Query("active", description="active | completed | all"),
    limit: int = Query(20, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns dispatch_routes with their aggregated team summary.
    """

    where = "" if status == "all" else f"WHERE dr.status = '{status}'"

    rows = await conn.fetch(f"""
        SELECT
            dr.id::text         AS id,
            dr.timestamp,
            dr.allocation_id::text,
            dr.incident_id,
            dr.zone_id,
            dr.zone_name,
            ST_Y(dr.destination::geometry) AS dest_lat,
            ST_X(dr.destination::geometry) AS dest_lon,
            dr.priority,
            dr.total_eta_minutes,
            dr.route_safety_score,
            dr.status,
            dr.completed_at,
            COUNT(tr.id)           AS team_count,
            ARRAY_AGG(tr.unit_name ORDER BY tr.eta_minutes) AS team_names,
            MIN(tr.eta_minutes)    AS fastest_eta,
            MAX(tr.eta_minutes)    AS slowest_eta
        FROM dispatch_routes dr
        LEFT JOIN team_routes tr ON tr.dispatch_id = dr.id
        {where}
        GROUP BY dr.id
        ORDER BY dr.timestamp DESC
        LIMIT $1
    """, limit)

    dispatches: List[Dict[str, Any]] = []
    for r in rows:
        dispatches.append({
            "id":                 r["id"],
            "timestamp":          r["timestamp"].isoformat(),
            "allocation_id":      r["allocation_id"],
            "incident_id":        r["incident_id"],
            "zone_id":            r["zone_id"],
            "zone_name":          r["zone_name"],
            "destination": {
                "latitude":  float(r["dest_lat"] or 0),
                "longitude": float(r["dest_lon"] or 0),
            },
            "priority":           r["priority"],
            "total_eta_minutes":  float(r["total_eta_minutes"] or 0),
            "route_safety_score": float(r["route_safety_score"] or 0),
            "status":             r["status"],
            "completed_at":       r["completed_at"].isoformat() if r["completed_at"] else None,
            "team_count":         int(r["team_count"] or 0),
            "team_names":         [n for n in (r["team_names"] or []) if n],
            "fastest_eta":        float(r["fastest_eta"] or 0),
            "slowest_eta":        float(r["slowest_eta"] or 0),
        })

    return {"dispatches": dispatches, "count": len(dispatches)}


# ── GET /api/dispatch/geojson ─────────────────────────────────────────────────

@router.get("/geojson")
async def dispatch_geojson(
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns a GeoJSON FeatureCollection of all ACTIVE team routes.
    Each Feature is a LineString (the OSRM route geometry) with team metadata.
    Consumed directly by MapLibre addSource("dispatch-routes", ...).
    """

    rows = await conn.fetch("""
        SELECT
            tr.id::text          AS id,
            tr.unit_name,
            tr.resource_type,
            tr.transport_mode,
            tr.distance_km,
            tr.eta_minutes,
            tr.status,
            tr.route_geometry,
            ST_Y(tr.origin::geometry)      AS origin_lat,
            ST_X(tr.origin::geometry)      AS origin_lon,
            ST_Y(tr.destination::geometry) AS dest_lat,
            ST_X(tr.destination::geometry) AS dest_lon,
            dr.zone_name,
            dr.priority,
            dr.route_safety_score
        FROM team_routes tr
        JOIN dispatch_routes dr ON dr.id = tr.dispatch_id
        WHERE dr.status = 'active'
        ORDER BY tr.eta_minutes ASC
    """)

    features: List[Dict[str, Any]] = []
    for r in rows:
        geom = r["route_geometry"]  # already JSONB dict or None

        # Fall back to straight line if OSRM geometry is missing
        if geom is None or not isinstance(geom, dict):
            geom = {
                "type": "LineString",
                "coordinates": [
                    [float(r["origin_lon"]), float(r["origin_lat"])],
                    [float(r["dest_lon"]),   float(r["dest_lat"])],
                ],
            }

        color = {
            "rescue_boat":  "#4da9ff",
            "medical_team": "#ff4444",
            "medical_kit":  "#ffaa00",
            "food_supply":  "#44cc88",
            "water_supply": "#79c0ff",
        }.get(r["resource_type"], "#4da9ff")

        features.append({
            "type":     "Feature",
            "geometry": geom,
            "properties": {
                "id":            r["id"],
                "unit_name":     r["unit_name"],
                "resource_type": r["resource_type"],
                "transport_mode": r["transport_mode"],
                "distance_km":   float(r["distance_km"] or 0),
                "eta_minutes":   float(r["eta_minutes"] or 0),
                "status":        r["status"],
                "zone_name":     r["zone_name"],
                "priority":      r["priority"],
                "safety_score":  float(r["route_safety_score"] or 0),
                "color":         color,
                "origin":    [float(r["origin_lon"]), float(r["origin_lat"])],
                "destination": [float(r["dest_lon"]),   float(r["dest_lat"])],
            },
        })

    return {
        "type":     "FeatureCollection",
        "features": features,
        "count":    len(features),
    }


# ── GET /api/dispatch/{route_id} ──────────────────────────────────────────────

@router.get("/{route_id}")
async def get_dispatch(
    route_id: str,
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """Single dispatch route with all team details."""

    route_row = await conn.fetchrow("""
        SELECT
            id::text, timestamp, allocation_id::text,
            incident_id, zone_id, zone_name,
            ST_Y(destination::geometry) AS dest_lat,
            ST_X(destination::geometry) AS dest_lon,
            priority, total_eta_minutes,
            route_safety_score, status, completed_at
        FROM dispatch_routes
        WHERE id::text = $1
    """, route_id)

    if not route_row:
        raise HTTPException(status_code=404, detail="Dispatch route not found")

    team_rows = await conn.fetch("""
        SELECT
            id::text, unit_id::text, unit_name, resource_type,
            transport_mode, distance_km, eta_minutes, status,
            route_geometry,
            ST_Y(origin::geometry)      AS origin_lat,
            ST_X(origin::geometry)      AS origin_lon,
            ST_Y(destination::geometry) AS dest_lat,
            ST_X(destination::geometry) AS dest_lon,
            departed_at, arrived_at
        FROM team_routes
        WHERE dispatch_id::text = $1
        ORDER BY eta_minutes ASC
    """, route_id)

    teams = []
    for t in team_rows:
        teams.append({
            "id":            t["id"],
            "unit_id":       t["unit_id"],
            "unit_name":     t["unit_name"],
            "resource_type": t["resource_type"],
            "transport_mode": t["transport_mode"],
            "distance_km":   float(t["distance_km"] or 0),
            "eta_minutes":   float(t["eta_minutes"] or 0),
            "status":        t["status"],
            "route_geometry": t["route_geometry"],
            "origin":    {"latitude": float(t["origin_lat"] or 0), "longitude": float(t["origin_lon"] or 0)},
            "destination": {"latitude": float(t["dest_lat"] or 0), "longitude": float(t["dest_lon"] or 0)},
            "departed_at": t["departed_at"].isoformat() if t["departed_at"] else None,
            "arrived_at":  t["arrived_at"].isoformat() if t["arrived_at"] else None,
        })

    route = dict(route_row)
    route["timestamp"] = route["timestamp"].isoformat()
    route["completed_at"] = route["completed_at"].isoformat() if route["completed_at"] else None
    route["destination"] = {
        "latitude":  float(route.pop("dest_lat") or 0),
        "longitude": float(route.pop("dest_lon") or 0),
    }
    route["teams"] = teams

    return route


# ── POST /api/dispatch ────────────────────────────────────────────────────────

class ManualDispatchBody(BaseModel):
    """Minimal body to manually trigger Agent 4 via the dashboard."""

    incident_id: str
    zone_id: str
    zone_name: str
    destination_lat: float
    destination_lon: float
    priority: int = Field(3, ge=1, le=5)
    urgency: str = "URGENT"
    num_people: int = 0
    resource_units: List[Dict[str, Any]] = []
    notes: str = ""


@router.post("")
async def manual_dispatch(body: ManualDispatchBody) -> Dict[str, Any]:
    """
    Manually trigger a dispatch order.
    Calls Agent 4 /trigger directly AND publishes to dispatch_order Redis channel.
    """
    import uuid

    payload = {
        "incident_id":       body.incident_id,
        "allocation_id":     str(uuid.uuid4()),
        "zone_id":           body.zone_id,
        "zone_name":         body.zone_name,
        "destination": {
            "latitude":  body.destination_lat,
            "longitude": body.destination_lon,
        },
        "priority":          body.priority,
        "urgency":           body.urgency,
        "num_people":        body.num_people,
        "allocated_resources": body.resource_units,
        "requires_medical":  False,
        "partial_allocation": False,
        "notes":             body.notes,
    }

    # Forward to Agent 4
    agent4_status: Optional[int] = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{AGENT4_URL}/trigger", json=payload)
            agent4_status = resp.status_code
    except Exception as exc:
        logger.warning("Agent 4 unreachable: %s", exc)

    # Publish to Redis channel
    published = await publish("dispatch_order", {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "sender_agent":   "dashboard_manual",
        "receiver_agent": "agent_4_dispatch",
        "message_type":   "dispatch_order",
        "payload":        payload,
    })

    return {
        "status":          "dispatched",
        "incident_id":     body.incident_id,
        "zone":            body.zone_name,
        "agent4_http":     agent4_status,
        "redis_published": published,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


# ── POST /api/dispatch/{route_id}/complete ────────────────────────────────────

@router.post("/{route_id}/complete")
async def complete_dispatch(
    route_id: str,
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """Mark a dispatch route as completed."""

    result = await conn.execute("""
        UPDATE dispatch_routes
        SET status = 'completed',
            completed_at = NOW()
        WHERE id::text = $1
          AND status = 'active'
    """, route_id)

    if result == "UPDATE 0":
        raise HTTPException(404, "Route not found or already completed")

    # Also update all team_routes under this dispatch
    await conn.execute("""
        UPDATE team_routes
        SET status = 'arrived', arrived_at = NOW()
        WHERE dispatch_id::text = $1
          AND status != 'arrived'
    """, route_id)

    return {"status": "completed", "route_id": route_id}
