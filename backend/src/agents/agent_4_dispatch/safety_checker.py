"""
src/agents/agent_4_dispatch/safety_checker.py
Route safety scoring: does a route pass through active flood zones?

Uses PostGIS ST_Intersects to check a road route geometry against
the flood zones detected by Agent 1 (stored in satellite_imagery_detections).

Safety score:
  1.0  → Route is completely clear of flood zones
  0.6  → Route passes through MODERATE flood zone
  0.3  → Route passes through HIGH flood zone
  0.1  → Route passes through CRITICAL flood zone
"""

import json
import logging
from typing import Optional

import asyncpg

logger = logging.getLogger("agent4.safety")

SEVERITY_SCORES = {
    "minimal":  1.0,
    "low":      0.9,
    "moderate": 0.6,
    "high":     0.3,
    "critical": 0.1,
}


class RouteSafetyChecker:

    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool

    async def check_route_safety(
        self,
        route_geojson: dict,
        transport_mode: str = "road",
    ) -> float:
        """
        Returns safety score 0.0–1.0 for a given GeoJSON LineString route.
        Boats are always 1.0 (they navigate water intentionally).
        """
        if transport_mode == "waterway":
            return 1.0   # Boats go through water — that's the point

        if not route_geojson:
            return 0.8   # Unknown route — moderately safe assumption

        try:
            geojson_str = json.dumps(route_geojson)
            async with self.pool.acquire() as conn:
                # Check if route intersects any active flood detection zones
                # Agent 1 stores detections in satellite_imagery_detections table
                rows = await conn.fetch("""
                    SELECT risk_level
                    FROM satellite_imagery_detections
                    WHERE is_active = TRUE
                      AND ST_Intersects(
                            flood_geometry,
                            ST_GeomFromGeoJSON($1)::geography::geometry
                          )
                    ORDER BY 
                        CASE risk_level
                            WHEN 'critical' THEN 1
                            WHEN 'high'     THEN 2
                            WHEN 'moderate' THEN 3
                            WHEN 'low'      THEN 4
                            ELSE 5
                        END
                    LIMIT 5
                """, geojson_str)

            if not rows:
                return 1.0   # No flood zones intersected — safe

            # Return the worst (minimum) safety score found
            scores = [
                SEVERITY_SCORES.get(row["risk_level"], 0.5)
                for row in rows
            ]
            min_score = min(scores)
            logger.info(
                "Route passes through %d flood zone(s), min safety=%.1f",
                len(rows), min_score
            )
            return min_score

        except Exception as e:
            # If Agent 1's table doesn't exist yet (pre-integration), return safe default
            logger.warning(
                "Safety check skipped (table unavailable: %s) — defaulting to 0.8", e
            )
            return 0.8

    async def get_flood_condition_at_point(
        self, lat: float, lon: float
    ) -> str:
        """
        Determine flood condition at a specific point — used to
        select the correct boat speed from BOAT_SPEEDS table.
        Returns: 'normal_river' | 'flood_shallow' | 'flood_deep' | 'flood_severe'
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT risk_level, flood_depth_m
                    FROM satellite_imagery_detections
                    WHERE is_active = TRUE
                      AND ST_DWithin(
                            flood_geometry,
                            ST_MakePoint($1, $2)::geography,
                            5000   -- within 5km
                          )
                    ORDER BY 
                        ST_Distance(
                            flood_geometry,
                            ST_MakePoint($1, $2)::geography
                        )
                    LIMIT 1
                """, lon, lat)

            if not row:
                return "normal_river"

            depth = row.get("flood_depth_m") or 0
            risk  = row.get("risk_level", "low")

            if risk == "critical" or depth > 2.0:
                return "flood_severe"
            elif risk == "high" or depth > 1.0:
                return "flood_deep"
            elif risk in ("moderate", "low") or depth > 0:
                return "flood_shallow"
            return "normal_river"

        except Exception as e:
            logger.warning("Flood condition check failed: %s", e)
            return "flood_shallow"   # Conservative default
