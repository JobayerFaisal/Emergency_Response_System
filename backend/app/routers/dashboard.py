"""
backend/app/routers/dashboard.py
==================================
GET /api/dashboard

Returns a single JSON snapshot used by the React KPI bar:
  - zone counts + highest risk
  - active alert counts by severity
  - teams deployed (active dispatch routes)
  - people at risk (sum of recent allocations)
  - per-agent online status (checked via HTTP health calls)
  - recent agent_messages (for the live feed)
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import asyncpg
from fastapi import APIRouter, Depends, Query

from app.services.db import get_db

logger = logging.getLogger("dashboard.routers.dashboard")

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Agent base URLs
_AGENT_URLS: Dict[str, str] = {
    "agent_1_environmental": os.getenv("AGENT1_URL", "http://localhost:8001"),
    "agent_2_distress":      os.getenv("AGENT2_URL", "http://localhost:8002"),
    "agent_3_resource":      os.getenv("AGENT3_URL", "http://localhost:8003"),
    "agent_4_dispatch":      os.getenv("AGENT4_URL", "http://localhost:8004"),
}


# ── helpers ──────────────────────────────────────────────────────────────────

async def _check_agent_health(name: str, url: str) -> Dict[str, Any]:
    """Call /health on a single agent; returns status dict."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{url}/health")
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            online = resp.status_code < 400 and body.get("status") in ("ok", "healthy", "operational")
            return {"agent": name, "url": url, "online": online, "detail": body}
    except Exception as exc:
        return {"agent": name, "url": url, "online": False, "detail": str(exc)}


async def _check_all_agents() -> List[Dict[str, Any]]:
    tasks = [_check_agent_health(n, u) for n, u in _AGENT_URLS.items()]
    return await asyncio.gather(*tasks)


# ── main endpoint ─────────────────────────────────────────────────────────────

@router.get("")
async def get_dashboard_summary(
    feed_limit: int = Query(30, ge=1, le=100, description="Number of feed messages to return"),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Single-call dashboard snapshot.  The React frontend polls this every 30 s.

    Returns:
      kpis          – headline numbers for the KPI bar
      agents        – online/offline status for all 4 agents
      feed          – latest agent_messages for the live feed panel
      last_updated  – ISO timestamp
    """

    # ── 1. Zone counts ────────────────────────────────────────────────────
    zone_rows = await conn.fetch("""
        SELECT
            COUNT(*)                                         AS total,
            COUNT(*) FILTER (WHERE risk_level = 'critical') AS critical,
            COUNT(*) FILTER (WHERE risk_level = 'high')     AS high,
            COUNT(*) FILTER (WHERE risk_level = 'moderate') AS moderate,
            COUNT(*) FILTER (WHERE risk_level = 'low')      AS low,
            COUNT(*) FILTER (WHERE risk_level = 'minimal')  AS minimal
        FROM sentinel_zones
    """)
    zone_stats = dict(zone_rows[0]) if zone_rows else {}

    # ── 2. Latest prediction stats ────────────────────────────────────────
    pred_rows = await conn.fetch("""
        WITH latest AS (
            SELECT DISTINCT ON (zone_id)
                zone_id, risk_score, severity_level, confidence, timestamp
            FROM flood_predictions
            ORDER BY zone_id, timestamp DESC
        )
        SELECT
            COUNT(*)                                              AS total_predictions,
            MAX(risk_score)                                       AS max_risk_score,
            AVG(risk_score)                                       AS avg_risk_score,
            COUNT(*) FILTER (WHERE severity_level = 'critical')  AS critical_predictions,
            COUNT(*) FILTER (WHERE severity_level = 'high')      AS high_predictions,
            COUNT(*) FILTER (WHERE severity_level NOT IN ('minimal','low')) AS active_alerts
        FROM latest
    """)
    pred_stats = dict(pred_rows[0]) if pred_rows else {}

    # ── 3. Teams deployed (active dispatch routes) ────────────────────────
    dispatch_rows = await conn.fetch("""
        SELECT
            COUNT(DISTINCT dr.id)     AS active_dispatches,
            COUNT(tr.id)              AS teams_deployed,
            AVG(dr.total_eta_minutes) AS avg_eta_minutes,
            MIN(dr.route_safety_score)AS min_safety_score
        FROM dispatch_routes dr
        LEFT JOIN team_routes tr ON tr.dispatch_id = dr.id
        WHERE dr.status = 'active'
    """)
    dispatch_stats = dict(dispatch_rows[0]) if dispatch_rows else {}

    # ── 4. People at risk (sum of recent allocations) ────────────────────
    people_rows = await conn.fetch("""
        SELECT COALESCE(SUM(num_people_affected), 0) AS people_at_risk
        FROM resource_allocations
        WHERE timestamp > NOW() - INTERVAL '6 hours'
          AND status NOT IN ('completed', 'cancelled')
    """)
    people_at_risk: int = people_rows[0]["people_at_risk"] if people_rows else 0

    # ── 5. Inventory summary ──────────────────────────────────────────────
    inv_rows = await conn.fetch("""
        SELECT resource_type,
               COUNT(*)                                      AS total,
               COUNT(*) FILTER (WHERE status='available')   AS available,
               COUNT(*) FILTER (WHERE status='deployed')    AS deployed,
               COUNT(*) FILTER (WHERE status='returning')   AS returning
        FROM resource_units
        GROUP BY resource_type
        ORDER BY resource_type
    """)
    inventory = [dict(r) for r in inv_rows]

    # ── 6. Agent feed (latest messages) ──────────────────────────────────
    feed_rows = await conn.fetch("""
        SELECT
            message_id::text, timestamp,
            sender_agent, receiver_agent, message_type,
            zone_id, priority,
            payload
        FROM agent_messages
        ORDER BY timestamp DESC
        LIMIT $1
    """, feed_limit)

    feed = []
    for r in feed_rows:
        item = dict(r)
        item["timestamp"] = item["timestamp"].isoformat() if item["timestamp"] else None
        feed.append(item)

    # ── 7. Agent health check (parallel HTTP) ────────────────────────────
    agents = await _check_all_agents()

    # ── Assemble KPI block ────────────────────────────────────────────────
    def _f(v: Any) -> Any:
        """Convert Decimal/None to float/None for JSON serialisation."""
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    kpis = {
        # Zones
        "zones_monitored":    int(zone_stats.get("total", 0)),
        "zones_critical":     int(zone_stats.get("critical", 0)),
        "zones_high":         int(zone_stats.get("high", 0)),
        "zones_moderate":     int(zone_stats.get("moderate", 0)),
        # Predictions
        "active_alerts":      int(pred_stats.get("active_alerts", 0)),
        "max_risk_score":     _f(pred_stats.get("max_risk_score", 0)),
        "avg_risk_score":     _f(pred_stats.get("avg_risk_score", 0)),
        "critical_predictions": int(pred_stats.get("critical_predictions", 0)),
        "high_predictions":   int(pred_stats.get("high_predictions", 0)),
        # Dispatch
        "active_dispatches":  int(dispatch_stats.get("active_dispatches", 0)),
        "teams_deployed":     int(dispatch_stats.get("teams_deployed", 0)),
        "avg_eta_minutes":    _f(dispatch_stats.get("avg_eta_minutes")),
        "min_safety_score":   _f(dispatch_stats.get("min_safety_score")),
        # People
        "people_at_risk":     int(people_at_risk),
        # System
        "agents_online":      sum(1 for a in agents if a["online"]),
        "agents_total":       len(agents),
    }

    return {
        "kpis":         kpis,
        "inventory":    inventory,
        "agents":       agents,
        "feed":         feed,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }