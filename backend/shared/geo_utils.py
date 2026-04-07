# backend/shared/geo_utils.py

"""
shared/geo_utils.py
Geographic helper utilities shared across all agents.
"""

import math
from typing import Tuple


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points in kilometres.
    Used for: finding closest resource, estimating boat travel distance.
    """
    R = 6371.0  # Earth radius km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def straight_line_geojson(
    origin_lat: float, origin_lon: float,
    dest_lat: float, dest_lon: float
) -> dict:
    """
    Build a GeoJSON LineString for a straight-line (waterway) route.
    Note: GeoJSON uses [longitude, latitude] order.
    """
    return {
        "type": "LineString",
        "coordinates": [
            [origin_lon, origin_lat],
            [dest_lon, dest_lat],
        ]
    }


def point_geojson(lat: float, lon: float) -> dict:
    """GeoJSON Point. Note: [longitude, latitude] order."""
    return {"type": "Point", "coordinates": [lon, lat]}


def postgis_point_wkt(lat: float, lon: float) -> str:
    """WKT string for inserting a point into PostGIS GEOGRAPHY column."""
    return f"SRID=4326;POINT({lon} {lat})"
