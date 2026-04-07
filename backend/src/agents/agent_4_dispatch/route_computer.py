# backend/src/agents/agent_4_dispatch/route_computer.py

"""
src/agents/agent_4_dispatch/route_computer.py
OSRM integration + boat ETA estimation for Agent 4.

Road vehicles  → OSRM /route/v1/driving (real Bangladesh roads)
Rescue boats   → Haversine straight-line + flood-adjusted speed
"""

import logging
import math
from typing import Optional

import httpx

from shared.severity import GeoPoint
from shared.geo_utils import haversine_km, straight_line_geojson
from .models import TransportMode, BOAT_SPEEDS, ROAD_SPEED_FALLBACK_KMH

logger = logging.getLogger("agent4.route_computer")


class RouteComputer:

    def __init__(self, osrm_url: str = "http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"):
        self.osrm_url = osrm_url.rstrip("/")

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    async def compute_route(
        self,
        origin: GeoPoint,
        destination: GeoPoint,
        transport_mode: TransportMode,
        flood_condition: str = "flood_shallow",
    ) -> dict:
        """
        Returns a dict with keys:
            distance_km, eta_minutes, route_geometry (GeoJSON LineString)
        """
        if transport_mode == TransportMode.WATERWAY:
            return self._boat_route(origin, destination, flood_condition)
        else:
            return await self._road_route(origin, destination)

    async def compute_route_matrix(
        self,
        origins: list[GeoPoint],
        destinations: list[GeoPoint],
    ) -> Optional[dict]:
        """
        OSRM Table API — get duration/distance matrix for multiple origins→destinations.
        Returns None if OSRM unavailable.
        """
        all_points = origins + destinations
        coords_str = ";".join(
            f"{p.longitude},{p.latitude}" for p in all_points
        )
        src_idx  = ";".join(str(i) for i in range(len(origins)))
        dest_idx = ";".join(
            str(i) for i in range(len(origins), len(origins) + len(destinations))
        )
        url = (
            f"{self.osrm_url}/table/v1/driving/{coords_str}"
            f"?sources={src_idx}&destinations={dest_idx}"
            f"&annotations=duration,distance"
        )
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                data = resp.json()
            if data.get("code") == "Ok":
                return data
        except Exception as e:
            logger.warning("OSRM table API unavailable: %s", e)
        return None

    # ─────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────

    async def _road_route(self, origin: GeoPoint, destination: GeoPoint) -> dict:
        """
        Query OSRM for road route.
        Falls back to straight-line × 1.3 (road detour factor) if OSRM down.
        """
        url = (
            f"{self.osrm_url}/route/v1/driving/"
            f"{origin.longitude},{origin.latitude};"
            f"{destination.longitude},{destination.latitude}"
            f"?overview=full&geometries=geojson&steps=false"
        )
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                data = resp.json()

            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                return {
                    "distance_km":    round(route["distance"] / 1000, 2),
                    "eta_minutes":    round(route["duration"] / 60, 1),
                    "route_geometry": route["geometry"],   # GeoJSON LineString
                    "source":         "osrm",
                }
        except Exception as e:
            logger.warning("OSRM unavailable (%s) — using fallback", e)

        # Fallback: straight-line × detour factor
        dist_km  = haversine_km(
            origin.latitude, origin.longitude,
            destination.latitude, destination.longitude
        ) * 1.3   # 30% road detour factor
        eta_min  = (dist_km / ROAD_SPEED_FALLBACK_KMH) * 60
        return {
            "distance_km":    round(dist_km, 2),
            "eta_minutes":    round(eta_min, 1),
            "route_geometry": straight_line_geojson(
                origin.latitude, origin.longitude,
                destination.latitude, destination.longitude,
            ),
            "source": "fallback_haversine",
        }

    def _boat_route(
        self,
        origin: GeoPoint,
        destination: GeoPoint,
        flood_condition: str,
    ) -> dict:
        """
        Straight-line route at flood-adjusted speed.
        Boats don't follow roads — they navigate floodwater directly.
        """
        dist_km = haversine_km(
            origin.latitude, origin.longitude,
            destination.latitude, destination.longitude,
        )
        speed   = BOAT_SPEEDS.get(flood_condition, BOAT_SPEEDS["flood_shallow"])
        eta_min = (dist_km / speed) * 60

        return {
            "distance_km":    round(dist_km, 2),
            "eta_minutes":    round(eta_min, 1),
            "route_geometry": straight_line_geojson(
                origin.latitude, origin.longitude,
                destination.latitude, destination.longitude,
            ),
            "source":         "boat_haversine",
            "flood_condition": flood_condition,
            "speed_kmh":      speed,
        }
