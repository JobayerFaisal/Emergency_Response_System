# backend/app/routers/resources.py
"""
backend/app/routers/resources.py
===================================
GET  /api/resources                  → full inventory (resource_units view)
GET  /api/resources/summary          → aggregated counts per type
GET  /api/resources/allocations      → recent resource_allocations
GET  /api/resources/allocations/{id} → single allocation detail
POST /api/resources/restock          → forward restock request to Agent 3
"""

import logging
import os
from typing import Any, Dict, List, Optional
from uuid import UUID

import asyncpg
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.db import get_db

logger = logging.getLogger("dashboard.routers.resources")

router = APIRouter(prefix="/api/resources", tags=["resources"])

AGENT3_URL = os.getenv("AGENT3_URL", "http://localhost:8003")


# ── GET /api/resources ────────────────────────────────────────────────────────

@router.get("")
async def list_resource_units(
    resource_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="available|deployed|returning|maintenance"),
    limit: int = Query(100, ge=1, le=500),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns raw resource_units rows — one row per vehicle/team.
    Useful for the resource map layer.
    """

    rows = await conn.fetch("""
        SELECT
            id::text, resource_type, name, status, capacity,
            ST_Y(current_location::geometry) AS lat,
            ST_X(current_location::geometry) AS lon,
            ST_Y(base_location::geometry)    AS base_lat,
            ST_X(base_location::geometry)    AS base_lon,
            assigned_zone_id, assigned_incident_id,
            deployed_at, created_at
        FROM resource_units
        WHERE ($1::text IS NULL OR resource_type = $1)
          AND ($2::text IS NULL OR status = $2)
        ORDER BY resource_type, name
        LIMIT $3
    """, resource_type, status, limit)

    units: List[Dict[str, Any]] = []
    for r in rows:
        units.append({
            "id":                   r["id"],
            "resource_type":        r["resource_type"],
            "name":                 r["name"],
            "status":               r["status"],
            "capacity":             r["capacity"],
            "current_location": {
                "latitude":  float(r["lat"] or 0),
                "longitude": float(r["lon"] or 0),
            },
            "base_location": {
                "latitude":  float(r["base_lat"] or 0),
                "longitude": float(r["base_lon"] or 0),
            },
            "assigned_zone_id":      r["assigned_zone_id"],
            "assigned_incident_id":  r["assigned_incident_id"],
            "deployed_at":           r["deployed_at"].isoformat() if r["deployed_at"] else None,
            "created_at":            r["created_at"].isoformat() if r["created_at"] else None,
        })

    return {"units": units, "count": len(units)}


# ── GET /api/resources/summary ────────────────────────────────────────────────

@router.get("/summary")
async def resource_summary(
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns inventory_summary view — aggregated counts per resource type.
    Used by the React inventory bars.
    """

    rows = await conn.fetch("""
        SELECT resource_type,
               total, available, deployed, returning, maintenance
        FROM inventory_summary
        ORDER BY resource_type
    """)

    summary: List[Dict[str, Any]] = []
    for r in rows:
        total = int(r["total"] or 0)
        avail = int(r["available"] or 0)
        pct = avail / total if total > 0 else 0
        summary.append({
            "resource_type": r["resource_type"],
            "total":         total,
            "available":     avail,
            "deployed":      int(r["deployed"] or 0),
            "returning":     int(r["returning"] or 0),
            "maintenance":   int(r["maintenance"] or 0),
            "availability_pct": round(pct, 3),
            "status_color": (
                "#44cc88" if pct > 0.5 else
                "#ffaa00" if pct > 0.2 else
                "#ff4444"
            ),
        })

    # Overall totals
    totals = {
        "total":     sum(s["total"] for s in summary),
        "available": sum(s["available"] for s in summary),
        "deployed":  sum(s["deployed"] for s in summary),
    }

    return {"summary": summary, "totals": totals}


# ── GET /api/resources/allocations ───────────────────────────────────────────

@router.get("/allocations")
async def list_allocations(
    zone_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns recent resource_allocations — what Agent 3 decided to deploy.
    """

    rows = await conn.fetch("""
        SELECT
            id::text, timestamp, incident_id,
            zone_id, zone_name,
            ST_Y(destination::geometry) AS dest_lat,
            ST_X(destination::geometry) AS dest_lon,
            priority, urgency, num_people_affected,
            allocated_units, partial_allocation, requires_medical, status
        FROM resource_allocations
        WHERE ($1::text IS NULL OR zone_id = $1)
        ORDER BY timestamp DESC
        LIMIT $2
    """, zone_id, limit)

    allocations: List[Dict[str, Any]] = []
    for r in rows:
        allocations.append({
            "id":                  r["id"],
            "incident_id":         r["incident_id"],
            "timestamp":           r["timestamp"].isoformat(),
            "zone_id":             r["zone_id"],
            "zone_name":           r["zone_name"],
            "destination": {
                "latitude":  float(r["dest_lat"] or 0),
                "longitude": float(r["dest_lon"] or 0),
            },
            "priority":            r["priority"],
            "urgency":             r["urgency"],
            "num_people_affected": r["num_people_affected"],
            "allocated_units":     r["allocated_units"],
            "partial_allocation":  r["partial_allocation"],
            "requires_medical":    r["requires_medical"],
            "status":              r["status"],
        })

    return {"allocations": allocations, "count": len(allocations)}


# ── GET /api/resources/allocations/{allocation_id} ───────────────────────────

@router.get("/allocations/{allocation_id}")
async def get_allocation(
    allocation_id: str,
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:

    row = await conn.fetchrow("""
        SELECT
            id::text, timestamp, incident_id, zone_id, zone_name,
            ST_Y(destination::geometry) AS dest_lat,
            ST_X(destination::geometry) AS dest_lon,
            priority, urgency, num_people_affected,
            allocated_units, partial_allocation, requires_medical, status
        FROM resource_allocations
        WHERE id::text = $1
    """, allocation_id)

    if not row:
        raise HTTPException(status_code=404, detail="Allocation not found")

    data = dict(row)
    data["timestamp"] = data["timestamp"].isoformat()
    data["destination"] = {
        "latitude":  float(data.pop("dest_lat") or 0),
        "longitude": float(data.pop("dest_lon") or 0),
    }
    return data


# ── POST /api/resources/restock ──────────────────────────────────────────────

class RestockBody(BaseModel):
    resource_type: str
    quantity: int = Field(..., gt=0)
    latitude: float
    longitude: float
    notes: Optional[str] = None


@router.post("/restock")
async def restock_resources(body: RestockBody) -> Dict[str, Any]:
    """
    Forwards a restock request to Agent 3's /inventory/restock endpoint.
    Agent 3 seeds the new units directly in PostgreSQL.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{AGENT3_URL}/inventory/restock",
                json={
                    "resource_type": body.resource_type,
                    "quantity":      body.quantity,
                    "location": {
                        "latitude":  body.latitude,
                        "longitude": body.longitude,
                    },
                    "notes": body.notes,
                },
            )
            if resp.status_code >= 400:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Agent 3 returned {resp.status_code}: {resp.text}",
                )
            return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Agent 3 unreachable: {exc}")