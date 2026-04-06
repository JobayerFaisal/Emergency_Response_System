"""
backend/app/routers/distress.py
=================================
GET  /api/distress          → recent distress incidents (from resource_allocations)
GET  /api/distress/messages → agent_messages of type distress_queue
POST /api/distress/trigger  → inject a manual flood alert into Agent 2
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.services.db import get_db
from app.services.redis_bridge import publish

logger = logging.getLogger("dashboard.routers.distress")

router = APIRouter(prefix="/api/distress", tags=["distress"])


# ── GET /api/distress ─────────────────────────────────────────────────────────

@router.get("")
async def list_incidents(
    urgency: Optional[str] = Query(None, description="LIFE_THREATENING | URGENT | MODERATE"),
    limit: int = Query(20, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns recent distress incidents sourced from resource_allocations.
    These are the incidents Agent 3 has processed and stored.
    """

    rows = await conn.fetch("""
        SELECT
            id::text, timestamp, incident_id,
            zone_id, zone_name,
            ST_Y(destination::geometry) AS dest_lat,
            ST_X(destination::geometry) AS dest_lon,
            priority, urgency, num_people_affected,
            allocated_units, partial_allocation,
            requires_medical, status
        FROM resource_allocations
        WHERE ($1::text IS NULL OR urgency = $1)
        ORDER BY timestamp DESC
        LIMIT $2
    """, urgency, limit)

    incidents: List[Dict[str, Any]] = []
    for r in rows:
        incidents.append({
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

    return {"incidents": incidents, "count": len(incidents)}


# ── GET /api/distress/messages ────────────────────────────────────────────────

@router.get("/messages")
async def get_distress_messages(
    limit: int = Query(30, ge=1, le=200),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns agent_messages of type distress_queue (published by Agent 2).
    Useful for the live feed panel.
    """

    rows = await conn.fetch("""
        SELECT
            message_id::text, timestamp,
            sender_agent, receiver_agent,
            zone_id, priority, payload
        FROM agent_messages
        WHERE message_type = 'distress_queue'
        ORDER BY timestamp DESC
        LIMIT $1
    """, limit)

    messages = []
    for r in rows:
        messages.append({
            "message_id":    r["message_id"],
            "timestamp":     r["timestamp"].isoformat(),
            "sender_agent":  r["sender_agent"],
            "receiver_agent": r["receiver_agent"],
            "zone_id":       r["zone_id"],
            "priority":      r["priority"],
            "payload":       r["payload"],
        })

    return {"messages": messages, "count": len(messages)}


# ── POST /api/distress/trigger ────────────────────────────────────────────────

class FloodAlertTrigger(BaseModel):
    zone_id: str
    zone_name: str
    risk_score: float
    severity_level: str = "high"
    confidence: float = 0.85
    risk_factors: dict = {}


import os

AGENT2_URL = os.getenv("AGENT2_URL", "http://localhost:8002")


@router.post("/trigger")
async def trigger_flood_alert(
    body: FloodAlertTrigger,
) -> Dict[str, Any]:
    """
    Manual flood alert trigger.
    POSTs to Agent 2 /trigger/flood-alert and simultaneously publishes
    to the flood_alert Redis channel so the WS bridge picks it up.
    """

    payload = body.model_dump()

    # 1. Forward to Agent 2 REST API
    agent2_status: Optional[int] = None
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(f"{AGENT2_URL}/trigger/flood-alert", json=payload)
            agent2_status = resp.status_code
    except Exception as exc:
        logger.warning("Agent 2 unreachable: %s", exc)

    # 2. Also publish direct to Redis so WS clients see it immediately
    published = await publish("flood_alert", {
        "message_id":    "manual",
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "sender_agent":  "dashboard_manual",
        "receiver_agent": "agent_2_distress",
        "message_type":  "flood_alert",
        "zone_id":       body.zone_id,
        "priority":      5 if body.severity_level in ("critical", "high") else 3,
        "payload":       payload,
    })

    return {
        "status":          "triggered",
        "zone":            body.zone_name,
        "agent2_http":     agent2_status,
        "redis_published": published,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }