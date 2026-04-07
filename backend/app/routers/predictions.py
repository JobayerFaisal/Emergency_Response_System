# backend/app/routers/predictions.py

"""
backend/app/routers/predictions.py
=====================================
GET /api/predictions          → latest prediction per zone
GET /api/predictions/history  → time-series for charts
GET /api/predictions/{zone_id}→ latest prediction for one zone
"""

import logging
from typing import Any, Dict, List, Optional

import asyncpg
from fastapi import APIRouter, Depends, Query

from app.services.db import get_db

logger = logging.getLogger("dashboard.routers.predictions")

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


# ── GET /api/predictions ──────────────────────────────────────────────────────

@router.get("")
async def list_predictions(
    severity: Optional[str] = Query(None, description="Filter: critical|high|moderate|low|minimal"),
    limit: int = Query(50, ge=1, le=500),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns the LATEST flood prediction for each zone.
    Ordered by risk_score descending — the highest-risk zones appear first.
    """

    rows = await conn.fetch("""
        WITH ranked AS (
            SELECT
                p.*,
                z.name        AS zone_name,
                ST_Y(z.center::geometry) AS lat,
                ST_X(z.center::geometry) AS lon,
                ROW_NUMBER() OVER (PARTITION BY p.zone_id ORDER BY p.timestamp DESC) AS rn
            FROM flood_predictions p
            JOIN sentinel_zones z ON z.id = p.zone_id
        )
        SELECT
            id::text, zone_id::text, zone_name, lat, lon,
            risk_score, severity_level, confidence,
            time_to_impact_hours, affected_area_km2,
            risk_factors, recommended_actions, timestamp
        FROM ranked
        WHERE rn = 1
          AND ($1::text IS NULL OR severity_level = $1)
        ORDER BY risk_score DESC
        LIMIT $2
    """, severity, limit)

    items: List[Dict[str, Any]] = []
    for r in rows:
        items.append({
            "id":                   r["id"],
            "zone_id":              r["zone_id"],
            "zone_name":            r["zone_name"],
            "latitude":             float(r["lat"]),
            "longitude":            float(r["lon"]),
            "risk_score":           float(r["risk_score"] or 0),
            "severity_level":       r["severity_level"],
            "confidence":           float(r["confidence"] or 0),
            "time_to_impact_hours": float(r["time_to_impact_hours"] or 0),
            "affected_area_km2":    float(r["affected_area_km2"] or 0),
            "risk_factors":         r["risk_factors"],
            "recommended_actions":  r["recommended_actions"],
            "timestamp":            r["timestamp"].isoformat(),
        })

    return {"predictions": items, "count": len(items)}


# ── GET /api/predictions/history ─────────────────────────────────────────────

@router.get("/history")
async def prediction_history(
    zone_id: Optional[str] = Query(None, description="Filter to a single zone"),
    hours: int = Query(24, ge=1, le=168, description="How many hours back"),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Time-series data for the React risk-score chart.
    Returns up to `hours` worth of predictions, grouped by zone.
    """

    rows = await conn.fetch("""
        SELECT
            p.id::text, p.zone_id::text, z.name AS zone_name,
            p.risk_score, p.severity_level, p.confidence,
            p.affected_area_km2, p.timestamp
        FROM flood_predictions p
        JOIN sentinel_zones z ON z.id = p.zone_id
        WHERE p.timestamp > NOW() - ($1 || ' hours')::INTERVAL
          AND ($2::text IS NULL OR p.zone_id::text = $2)
        ORDER BY p.timestamp ASC
    """, str(hours), zone_id)

    # Group by zone for the frontend chart library
    by_zone: Dict[str, List] = {}
    for r in rows:
        zid = r["zone_id"]
        if zid not in by_zone:
            by_zone[zid] = []
        by_zone[zid].append({
            "risk_score":      float(r["risk_score"] or 0),
            "severity_level":  r["severity_level"],
            "confidence":      float(r["confidence"] or 0),
            "affected_km2":    float(r["affected_area_km2"] or 0),
            "timestamp":       r["timestamp"].isoformat(),
        })

    # Build series list
    series = []
    for zid, points in by_zone.items():
        if points:
            series.append({
                "zone_id":   zid,
                "zone_name": rows[[r["zone_id"] for r in rows].index(zid)]["zone_name"],
                "points":    points,
            })

    return {
        "series":    series,
        "hours":     hours,
        "zone_id":   zone_id,
        "count":     sum(len(s["points"]) for s in series),
    }


# ── GET /api/predictions/{zone_id} ───────────────────────────────────────────

@router.get("/{zone_id}")
async def get_zone_prediction(
    zone_id: str,
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """Latest prediction for a single zone."""

    row = await conn.fetchrow("""
        SELECT
            p.id::text, p.zone_id::text, z.name AS zone_name,
            p.risk_score, p.severity_level, p.confidence,
            p.time_to_impact_hours, p.affected_area_km2,
            p.risk_factors, p.recommended_actions, p.timestamp
        FROM flood_predictions p
        JOIN sentinel_zones z ON z.id = p.zone_id
        WHERE p.zone_id::text = $1
        ORDER BY p.timestamp DESC
        LIMIT 1
    """, zone_id)

    if not row:
        return {"prediction": None, "zone_id": zone_id}

    return {
        "prediction": {
            "id":                   row["id"],
            "zone_id":              row["zone_id"],
            "zone_name":            row["zone_name"],
            "risk_score":           float(row["risk_score"] or 0),
            "severity_level":       row["severity_level"],
            "confidence":           float(row["confidence"] or 0),
            "time_to_impact_hours": float(row["time_to_impact_hours"] or 0),
            "affected_area_km2":    float(row["affected_area_km2"] or 0),
            "risk_factors":         row["risk_factors"],
            "recommended_actions":  row["recommended_actions"],
            "timestamp":            row["timestamp"].isoformat(),
        }
    }