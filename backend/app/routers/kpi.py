"""
backend/app/routers/kpi.py
===========================
GET /api/kpi          → KPI summary for the top bar (matches useDashboard.js)
GET /api/agents/agent1 → Agent 1 data shape (matches Agent1Panel.jsx)
GET /api/agents/agent2 → Agent 2 data shape (matches Agent2Panel.jsx)
GET /api/agents/agent3 → Agent 3 data shape (matches Agent3Panel.jsx)
GET /api/agents/agent4 → Agent 4 data shape (matches Agent4Panel.jsx)

These are the routes the React frontend actually calls.
Shape contracts match exactly what each panel component expects.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List
import json

import asyncpg
from fastapi import APIRouter, Depends

from app.services.db import get_db
from app.config.scenario import is_replay
from app.services.redis_bridge import _redis as _get_redis

logger = logging.getLogger("dashboard.routers.kpi")

router = APIRouter(tags=["kpi"])


def _as_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}

def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value]
    return [value]

# ── GET /api/kpi ──────────────────────────────────────────────────────────────

@router.get("/api/kpi")
async def get_kpi(conn: asyncpg.Connection = Depends(get_db)) -> Dict[str, Any]:
    """
    KPI shape expected by KPIBar.jsx / useDashboard.js:
    {
      incidentStatus, activeZones, criticalAlerts,
      deployedTeams, affectedPeople, severity, confidence
    }

    FIX — LIVE MODE STALE DATA:
    All queries are now filtered to the last 6 hours so that old replay-session
    rows (timestamped 2022-06-17) never bleed into the live dashboard.
    """

    # Zones counts
    zone_row = await conn.fetchrow("""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE risk_level IN ('critical','high')) AS active
        FROM sentinel_zones
    """)

    # Latest predictions summary — LIVE only: last 6 hours
    # FIX: Without this WHERE clause, a previous replay run's 2022-dated rows
    # with high risk_scores always win ORDER BY risk_score DESC and appear as
    # "live" flood detections even when Agent 1 has since written lower-risk
    # real readings. Adding the 6-hour window ensures only fresh data is shown.
    pred_row = await conn.fetchrow("""
        WITH latest AS (
            SELECT DISTINCT ON (zone_id)
                risk_score, severity_level, confidence
            FROM flood_predictions
            WHERE timestamp > NOW() - INTERVAL '6 hours'
            ORDER BY zone_id, timestamp DESC
        )
        SELECT
            COUNT(*) FILTER (WHERE severity_level IN ('critical','high')) AS critical_alerts,
            MAX(risk_score)   AS max_risk,
            AVG(confidence)   AS avg_conf,
            CASE
              WHEN MAX(risk_score) >= 0.8 THEN 5
              WHEN MAX(risk_score) >= 0.6 THEN 4
              WHEN MAX(risk_score) >= 0.4 THEN 3
              WHEN MAX(risk_score) >= 0.2 THEN 2
              ELSE 1
            END AS severity_int
        FROM latest
    """)

    # Teams deployed
    team_row = await conn.fetchrow("""
        SELECT
            COUNT(*) FILTER (WHERE status = 'deployed') AS deployed,
            COUNT(*) AS total
        FROM resource_units
    """)

    # People at risk — last 6 hours only
    people_row = await conn.fetchrow("""
        SELECT COALESCE(SUM(num_people_affected), 0) AS total
        FROM resource_allocations
        WHERE timestamp > NOW() - INTERVAL '6 hours'
          AND status NOT IN ('completed','cancelled')
    """)

    max_risk = float(pred_row["max_risk"] or 0)
    incident_status = (
        "FLOOD_RESPONSE_ACTIVE" if max_risk >= 0.6 else
        "ACTIVE"                if max_risk >= 0.3 else
        "IDLE"
    )

    return {
        "incidentStatus":  incident_status,
        "activeZones":     int(zone_row["active"] or 0),
        "criticalAlerts":  int(pred_row["critical_alerts"] or 0),
        "deployedTeams":   int(team_row["deployed"] or 0),
        "affectedPeople":  int(people_row["total"] or 0),
        "severity":        int(pred_row["severity_int"] or 1),
        "confidence":      round(float(pred_row["avg_conf"] or 0), 3),
    }


# ── GET /api/agents/agent1 ────────────────────────────────────────────────────

@router.get("/api/agents/agent1")
async def get_agent1(conn: asyncpg.Connection = Depends(get_db)) -> Dict[str, Any]:
    """
    Shape expected by Agent1Panel.jsx:
    {
      detected, severity, confidence, trend, risk_factors[],
      evidence_refs[], weather{}, scores{},
      data_age_minutes, all_zones[]
    }

    FIX — AGENT 1 FALSE OUTPUT IN LIVE MODE:
    Root cause: The query had no timestamp filter, so rows inserted during a
    2022 historical replay session (with high risk_scores like 0.92) were
    always returned as the "latest" prediction because ORDER BY risk_score DESC
    put them first. Those rows have timestamp='2022-06-17 ...' but were written
    to the DB during the replay and stay there forever.

    Fix 1 — Recency filter: Only read predictions written in the last 6 hours.
             If Agent 1 has run recently, this returns its actual current output.
             If Agent 1 has NOT run yet today, this returns empty → panel shows
             "No recent data" instead of stale 2022 values.

    Fix 2 — ORDER BY timestamp DESC first, THEN risk_score: We want the most
             RECENT cycle's data, not the highest-ever historical score.

    Fix 3 — Per-zone summary: Return all monitored zones so the panel can show
             a zone table alongside the top-risk zone headline.

    Fix 4 — data_age_minutes: Tell the frontend how old the data is so it can
             show a staleness indicator.
    """

    # FIX: Only read predictions from the last 6 hours, ordered by recency first.
    pred = await conn.fetchrow("""
        SELECT p.risk_score, p.severity_level, p.confidence,
               p.risk_factors, p.recommended_actions, p.timestamp,
               z.name AS zone_name
        FROM flood_predictions p
        JOIN sentinel_zones z ON z.id = p.zone_id
        WHERE p.timestamp > NOW() - INTERVAL '6 hours'
        ORDER BY p.timestamp DESC, p.risk_score DESC
        LIMIT 1
    """)

    # FIX: Same recency filter for weather — old weather from a replay session
    # should not appear on the live panel.
    weather = await conn.fetchrow("""
        SELECT temperature, humidity, wind_speed,
               precipitation_1h, precipitation_24h, condition
        FROM weather_data
        WHERE timestamp > NOW() - INTERVAL '6 hours'
        ORDER BY timestamp DESC
        LIMIT 1
    """)

    # Per-zone risk summary (last 6 hours, one row per zone)
    zone_rows = await conn.fetch("""
        SELECT DISTINCT ON (p.zone_id)
               z.name AS zone_name,
               p.risk_score, p.severity_level, p.confidence, p.timestamp
        FROM flood_predictions p
        JOIN sentinel_zones z ON z.id = p.zone_id
        WHERE p.timestamp > NOW() - INTERVAL '6 hours'
        ORDER BY p.zone_id, p.timestamp DESC
    """)

    all_zones = [
        {
            "zone_name":     r["zone_name"],
            "risk_score":    float(r["risk_score"] or 0),
            "severity":      r["severity_level"],
            "confidence":    float(r["confidence"] or 0),
            "timestamp":     r["timestamp"].isoformat() if r["timestamp"] else None,
        }
        for r in zone_rows
    ]

    # No recent data from Agent 1 — return a safe empty state
    if not pred:
        return {
            "detected":        False,
            "severity":        0,
            "confidence":      0,
            "trend":           "stable",
            "risk_factors":    [],
            "evidence_refs":   [],
            "weather":         {},
            "scores":          {},
            "zone_name":       None,
            "timestamp":       None,
            "data_age_minutes": None,
            "all_zones":       all_zones,
            "no_recent_data":  True,
        }

    rf                  = _as_dict(pred["risk_factors"])
    recommended_actions = _as_list(pred["recommended_actions"])
    risk                = float(pred["risk_score"] or 0)

    # Data age — how many minutes ago did Agent 1 last write?
    pred_ts = pred["timestamp"]
    data_age_minutes = None
    if pred_ts:
        # pred_ts may be tz-naive (stored as UTC without tz info in some setups)
        if pred_ts.tzinfo is None:
            pred_ts = pred_ts.replace(tzinfo=timezone.utc)
        data_age_minutes = round(
            (datetime.now(timezone.utc) - pred_ts).total_seconds() / 60, 1
        )

    # Map risk_factors dict keys → 9-bar display scores.
    # Key names come directly from FloodRiskFactors.model_dump() in predictor.py.
    scores = {
        "rainfall_score":     float(rf.get("rainfall_intensity",         0)),
        "river_level_score":  float(rf.get("river_level_factor",         0)),
        "water_extent_score": float(rf.get("satellite_flood_detection",  0)),
        "social_score":       float(rf.get("social_reports_density",     0)),
        "depth_score":        float(rf.get("flood_depth_estimate",       0)),
        "trend_score":        float(rf.get("accumulated_rainfall",       0)),
        "satellite_conf":     float(rf.get("satellite_flood_detection",  0)),
        "credibility_score":  float(rf.get("historical_risk",            0)),
        "cluster_score":      float(rf.get("drainage_factor",            0)),
    }

    w = dict(weather) if weather else {}

    return {
        "detected":      risk >= 0.4,
        "severity":      (
            5 if pred["severity_level"] == "critical" else
            4 if pred["severity_level"] == "high"     else
            3 if pred["severity_level"] == "moderate" else
            2 if pred["severity_level"] == "low"      else 1
        ),
        "confidence":    float(pred["confidence"] or 0),
        "trend":         "rising" if risk >= 0.5 else "stable",
        "zone_name":     pred["zone_name"],
        "risk_factors":  recommended_actions,
        "evidence_refs": [
            f"Zone: {pred['zone_name']}",
            f"Risk score: {risk:.2f}",
            f"Data age: {data_age_minutes:.0f} min" if data_age_minutes is not None else "Timestamp: unknown",
        ],
        "weather": {
            "rainfall_mm":   float(w.get("precipitation_1h") or w.get("precipitation_24h") or 0),
            "river_level_m": None,
            "danger_level_m": None,
            "condition":     w.get("condition", "unknown"),
            "temperature_c": float(w.get("temperature") or 0),
            "humidity_pct":  float(w.get("humidity") or 0),
            "wind_ms":       float(w.get("wind_speed") or 0),
        },
        "scores":            scores,
        "timestamp":         pred["timestamp"].isoformat() if pred["timestamp"] else None,
        "data_age_minutes":  data_age_minutes,
        "all_zones":         all_zones,
        "no_recent_data":    False,
    }


# ── GET /api/agents/agent2 ────────────────────────────────────────────────────

@router.get("/api/agents/agent2")
async def get_agent2(conn: asyncpg.Connection = Depends(get_db)) -> Dict[str, Any]:
    """
    Shape expected by Agent2Panel.jsx:
    { reports[], clusters{}, total_reports, critical_reports }

    FIX: In replay mode, return empty reports so the DB data from previous
    runs does not bleed into the replay view. Distress pins are driven
    exclusively by WS events (distress_queue channel) during replay.

    FIX: In LIVE mode, filter resource_allocations to the last 6 hours so
    old replay-session allocations (timestamped 2022-06-17) don't appear
    as active distress pins on a live dashboard.
    """

    # In replay mode suppress DB rows — WS events from the simulator
    # drive distress pins and the Agent2 panel during replay.
    if is_replay():
        return {
            "reports":          [],
            "clusters":         {},
            "total_reports":    0,
            "critical_reports": 0,
        }

    # FIX: Added WHERE timestamp > NOW() - INTERVAL '6 hours'
    # Previously this had no time filter, so resource_allocations rows inserted
    # during a 2022 replay session appeared here as "live" distress reports.
    rows = await conn.fetch("""
        SELECT
            id::text, incident_id, zone_id, zone_name,
            ST_Y(destination::geometry) AS lat,
            ST_X(destination::geometry) AS lon,
            urgency, priority, num_people_affected, status, timestamp
        FROM resource_allocations
        WHERE timestamp > NOW() - INTERVAL '6 hours'
        ORDER BY timestamp DESC
        LIMIT 50
    """)

    reports = []
    urgency_map = {"LIFE_THREATENING": 5, "URGENT": 4, "MODERATE": 3, "LOW": 2}

    for r in rows:
        urgency_int = urgency_map.get(r["urgency"], 3)
        reports.append({
            "report_id":   r["id"],
            "text":        f"{r['urgency']} situation at {r['zone_name']} — {r['num_people_affected'] or '?'} people",
            "district":    r["zone_name"],
            "urgency":     urgency_int,
            "credibility": 0.85,
            "lat":         float(r["lat"] or 0),
            "lon":         float(r["lon"] or 0),
            "status":      r["status"],
        })

    clusters = {}
    for rep in reports:
        key = rep["district"] or "unknown"
        clusters.setdefault(key, []).append(rep["report_id"])

    critical = sum(1 for r in reports if r["urgency"] >= 5)

    return {
        "reports":          reports,
        "clusters":         clusters,
        "total_reports":    len(reports),
        "critical_reports": critical,
    }


# ── GET /api/agents/agent3 ────────────────────────────────────────────────────

@router.get("/api/agents/agent3")
async def get_agent3(conn: asyncpg.Connection = Depends(get_db)) -> Dict[str, Any]:
    """
    Shape expected by Agent3Panel.jsx:
    { inventory[], volunteers[], total_volunteers, available_volunteers, deployed_volunteers }

    FIX: In replay mode, return zero inventory so old deployed counts from
    the DB don't show. The replay simulator publishes inventory_update events
    which App.jsx merges into agents.agent3 in real-time.
    In LIVE mode, read from the DB as normal.
    """

    if is_replay():
        # Return zeroed inventory in replay — WS events drive the panel
        return {
            "inventory":             [],
            "volunteers":            [],
            "total_volunteers":      0,
            "available_volunteers":  0,
            "deployed_volunteers":   0,
        }

    # Resource summary
    inv_rows = await conn.fetch("""
        SELECT resource_type,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE status='available') AS available,
               COUNT(*) FILTER (WHERE status='deployed')  AS deployed
        FROM resource_units
        GROUP BY resource_type
        ORDER BY resource_type
    """)

    inventory = [
        {
            "type":      r["resource_type"],
            "available": int(r["available"] or 0),
            "total":     int(r["total"] or 0),
            "unit":      "",
        }
        for r in inv_rows
    ]

    # Units as "volunteers"
    unit_rows = await conn.fetch("""
        SELECT
            id::text AS volunteer_id,
            name,
            status,
            resource_type,
            ST_Y(current_location::geometry) AS lat,
            ST_X(current_location::geometry) AS lon,
            assigned_zone_id
        FROM resource_units
        ORDER BY status, name
        LIMIT 30
    """)

    volunteers = [
        {
            "volunteer_id": u["volunteer_id"][:8],
            "available":    u["status"] == "available",
            "district":     u["assigned_zone_id"] or "Base",
            "skills":       [u["resource_type"]],
            "equipment":    [u["name"]],
            "lat":          float(u["lat"] or 0),
            "lon":          float(u["lon"] or 0),
        }
        for u in unit_rows
    ]

    total     = len(volunteers)
    available = sum(1 for v in volunteers if v["available"])

    return {
        "inventory":             inventory,
        "volunteers":            volunteers,
        "total_volunteers":      total,
        "available_volunteers":  available,
        "deployed_volunteers":   total - available,
    }


# ── GET /api/agents/agent4 ────────────────────────────────────────────────────

@router.get("/api/agents/agent4")
async def get_agent4(conn: asyncpg.Connection = Depends(get_db)) -> Dict[str, Any]:
    """
    Shape expected by Agent4Panel.jsx:
    { missions[], total_missions, active_missions, completed_missions, failed_missions }
    """

    rows = await conn.fetch("""
        SELECT
            tr.id::text        AS mission_id,
            tr.unit_name       AS type,
            tr.status,
            tr.eta_minutes,
            tr.distance_km,
            dr.zone_name       AS district,
            dr.priority,
            tr.unit_name       AS assigned_volunteer,
            ST_Y(tr.destination::geometry) AS lat,
            ST_X(tr.destination::geometry) AS lon
        FROM team_routes tr
        JOIN dispatch_routes dr ON dr.id = tr.dispatch_id
        ORDER BY tr.eta_minutes ASC
        LIMIT 50
    """)

    status_map = {
        "pending":   "CREATED",
        "assigned":  "ASSIGNED",
        "en_route":  "EN_ROUTE",
        "active":    "ACTIVE",
        "arrived":   "COMPLETED",
        "failed":    "FAILED",
    }

    priority_map = {5: "CRITICAL", 4: "HIGH", 3: "MEDIUM", 2: "LOW", 1: "LOW"}

    missions = [
        {
            "mission_id":         r["mission_id"],
            "type":               r["type"],
            "status":             status_map.get(r["status"], "ASSIGNED"),
            "district":           r["district"],
            "priority":           priority_map.get(r["priority"], "MEDIUM"),
            "assigned_volunteer": r["assigned_volunteer"],
            "eta_minutes":        int(r["eta_minutes"] or 0),
            "distance_km":        float(r["distance_km"] or 0),
            "lat":                float(r["lat"] or 0),
            "lon":                float(r["lon"] or 0),
        }
        for r in rows
    ]

    total     = len(missions)
    active    = sum(1 for m in missions if m["status"] in ("EN_ROUTE", "ACTIVE", "ASSIGNED"))
    completed = sum(1 for m in missions if m["status"] == "COMPLETED")
    failed    = sum(1 for m in missions if m["status"] == "FAILED")

    return {
        "missions":           missions,
        "total_missions":     total,
        "active_missions":    active,
        "completed_missions": completed,
        "failed_missions":    failed,
    }
