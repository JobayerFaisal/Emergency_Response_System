"""
backend/app/routers/zones.py
=============================
GET /api/zones            → GeoJSON FeatureCollection of all sentinel zones
GET /api/zones/{zone_id}  → Single zone Feature with latest prediction attached
"""

import logging
from typing import Any, Dict, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query

from app.services.db import get_db

logger = logging.getLogger("dashboard.routers.zones")

router = APIRouter(prefix="/api/zones", tags=["zones"])


def _risk_to_color(risk_level: str) -> str:
    return {
        "critical": "#ff4444",
        "high":     "#ffaa00",
        "moderate": "#4da9ff",
        "low":      "#66cc88",
        "minimal":  "#44aa66",
    }.get(risk_level, "#4da9ff")


# ── GET /api/zones ────────────────────────────────────────────────────────────

@router.get("")
async def list_zones(
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns all sentinel zones as a GeoJSON FeatureCollection.
    Each Feature's properties include the latest flood_prediction for that zone.
    Consumed directly by MapLibre GL addSource().
    """

    where_clause = ""
    params: list = []
    if risk_level:
        where_clause = "WHERE z.risk_level = $1"
        params.append(risk_level)

    rows = await conn.fetch(f"""
        SELECT
            z.id::text            AS id,
            z.name,
            z.risk_level,
            z.radius_km,
            z.population_density,
            z.elevation,
            z.drainage_capacity,
            z.last_monitored,
            ST_Y(z.center::geometry) AS lat,
            ST_X(z.center::geometry) AS lon,
            -- Latest prediction (subquery)
            p.risk_score,
            p.severity_level,
            p.confidence,
            p.affected_area_km2,
            p.risk_factors,
            p.timestamp           AS prediction_ts
        FROM sentinel_zones z
        LEFT JOIN LATERAL (
            SELECT risk_score, severity_level, confidence,
                   affected_area_km2, risk_factors, timestamp
            FROM flood_predictions
            WHERE zone_id = z.id
            ORDER BY timestamp DESC
            LIMIT 1
        ) p ON TRUE
        {where_clause}
        ORDER BY COALESCE(p.risk_score, 0) DESC
    """, *params)

    features = []
    for r in rows:
        risk = r["risk_level"] or "minimal"
        pred_risk = r["severity_level"] or risk

        props = {
            "id":                 r["id"],
            "name":               r["name"],
            "risk_level":         risk,
            "radius_km":          float(r["radius_km"] or 5),
            "population_density": r["population_density"],
            "elevation":          float(r["elevation"] or 0),
            "drainage_capacity":  r["drainage_capacity"],
            "last_monitored":     r["last_monitored"].isoformat() if r["last_monitored"] else None,
            # prediction
            "risk_score":         float(r["risk_score"] or 0),
            "severity_level":     pred_risk,
            "confidence":         float(r["confidence"] or 0),
            "affected_area_km2":  float(r["affected_area_km2"] or 0),
            "risk_factors":       r["risk_factors"],
            "prediction_ts":      r["prediction_ts"].isoformat() if r["prediction_ts"] else None,
            # map styling helpers
            "color":              _risk_to_color(pred_risk),
            "fill_opacity":       min(0.75, 0.08 + float(r["risk_score"] or 0) * 0.85),
        }

        features.append({
            "type":       "Feature",
            "geometry": {
                "type":        "Point",
                "coordinates": [float(r["lon"]), float(r["lat"])],
            },
            "properties": props,
        })

    return {
        "type":     "FeatureCollection",
        "features": features,
        "count":    len(features),
    }


# ── GET /api/zones/{zone_id} ──────────────────────────────────────────────────

@router.get("/{zone_id}")
async def get_zone(
    zone_id: str,
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns a single zone with its full prediction history (last 24 h).
    """

    zone_row = await conn.fetchrow("""
        SELECT
            id::text, name, risk_level, radius_km,
            population_density, elevation, drainage_capacity, last_monitored,
            ST_Y(center::geometry) AS lat,
            ST_X(center::geometry) AS lon
        FROM sentinel_zones
        WHERE id::text = $1
    """, zone_id)

    if not zone_row:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id!r} not found")

    # Last 24 h of predictions for this zone
    pred_rows = await conn.fetch("""
        SELECT risk_score, severity_level, confidence,
               affected_area_km2, risk_factors, recommended_actions, timestamp
        FROM flood_predictions
        WHERE zone_id::text = $1
          AND timestamp > NOW() - INTERVAL '24 hours'
        ORDER BY timestamp DESC
        LIMIT 48
    """, zone_id)

    predictions = []
    for p in pred_rows:
        predictions.append({
            "risk_score":         float(p["risk_score"] or 0),
            "severity_level":     p["severity_level"],
            "confidence":         float(p["confidence"] or 0),
            "affected_area_km2":  float(p["affected_area_km2"] or 0),
            "risk_factors":       p["risk_factors"],
            "recommended_actions": p["recommended_actions"],
            "timestamp":          p["timestamp"].isoformat(),
        })

    # Latest weather for this zone
    weather_row = await conn.fetchrow("""
        SELECT temperature, humidity, wind_speed,
               precipitation_1h, precipitation_24h, condition, timestamp
        FROM weather_data
        WHERE zone_id::text = $1
        ORDER BY timestamp DESC
        LIMIT 1
    """, zone_id)

    weather = None
    if weather_row:
        weather = dict(weather_row)
        weather["timestamp"] = weather["timestamp"].isoformat() if weather["timestamp"] else None

    zone_data = dict(zone_row)
    zone_data["last_monitored"] = (
        zone_data["last_monitored"].isoformat() if zone_data["last_monitored"] else None
    )

    return {
        "zone":        zone_data,
        "predictions": predictions,
        "weather":     weather,
    }