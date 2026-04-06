# dashboard/api_client.py
"""
API Client
==========
All communication with the backend lives here.
Two data sources:
  1. HTTP  — Agent 1 FastAPI  GET /output   (full AgentOutput Pydantic model)
  2. Redis — pub/sub channel  flood_alert   (real-time agent coordination feed)

Both are kept intentionally thin so callbacks stay clean.
"""

from __future__ import annotations

import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

AGENT1_BASE = os.getenv("AGENT1_URL", "http://backend:8000")
REDIS_URL    = os.getenv("REDIS_URL",  "redis://redis:6379/0")
REQUEST_TIMEOUT = 5   # seconds

# ── HTTP session with retries ─────────────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://",  adapter)
    session.mount("https://", adapter)
    return session

_session = _make_session()


# ── AgentOutput fetcher ───────────────────────────────────────────────────────

def fetch_agent_output() -> Optional[dict]:
    """
    Fetch the latest AgentOutput from Agent 1's FastAPI /output endpoint.

    Returns the raw dict (already JSON-decoded) or None on any error.
    The dict structure mirrors the AgentOutput Pydantic model:

        {
          "agent_id": "agent_1_environmental",
          "timestamp": "...",
          "predictions": [ { FloodPrediction }, ... ],
          "alerts":      [ { EnvironmentalAlert }, ... ],
          "monitored_zones": [ { SentinelZone }, ... ],
          "data_sources_status": { ... },
          "processing_time_seconds": 21.4,
          "next_update_in_seconds": 180,
        }
    """
    try:
        resp = _session.get(
            f"{AGENT1_BASE}/output",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        logger.warning("Agent 1 unreachable — running in offline/demo mode")
        return None
    except Exception as e:
        logger.error(f"fetch_agent_output error: {e}")
        return None


def fetch_agent_status() -> Optional[dict]:
    """Fetch /status (lightweight heartbeat — used for the KPI bar)."""
    try:
        resp = _session.get(
            f"{AGENT1_BASE}/status",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ── Demo / fallback data ──────────────────────────────────────────────────────
# Used when Agent 1 is not reachable (local dev, unit tests, demo mode).

DEMO_OUTPUT: dict = {
    "agent_id": "agent_1_environmental",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "predictions": [
        {
            "zone": {
                "id": "00000000-0000-0000-0000-000000000001",
                "name": "Sunamganj Sadar",
                "center": {"latitude": 24.866, "longitude": 91.399},
                "radius_km": 6.0,
                "risk_level": "critical",
                "population_density": 28000,
                "elevation": 2.0,
                "drainage_capacity": "poor",
            },
            "risk_score": 0.82,
            "severity_level": "high",
            "confidence": 0.85,
            "risk_factors": {
                "rainfall_intensity": 0.65,
                "accumulated_rainfall": 0.58,
                "weather_severity": 0.55,
                "satellite_flood_detection": 0.897,
                "flood_depth_estimate": 0.72,
                "drainage_factor": 0.80,
                "elevation_factor": 0.90,
                "social_reports_density": 0.40,
                "historical_risk": 0.70,
                "river_level_factor": 0.74,
                "has_satellite_data": True,
                "has_social_data": False,
                "has_river_data": True,
                "satellite_confirmed_flooding": True,
            },
            "affected_area_km2": 18.4,
            "time_to_impact_hours": None,
            "recommended_actions": [
                "SATELLITE CONFIRMED: Active flooding detected",
                "Alert emergency response teams to standby",
                "Issue public warnings via all channels",
            ],
            "alert_level": "flood_risk",
        },
        {
            "zone": {
                "id": "00000000-0000-0000-0000-000000000002",
                "name": "Sylhet City",
                "center": {"latitude": 24.8975, "longitude": 91.872},
                "radius_km": 5.0,
                "risk_level": "high",
                "population_density": 35000,
                "elevation": 3.5,
                "drainage_capacity": "poor",
            },
            "risk_score": 0.68,
            "severity_level": "high",
            "confidence": 0.78,
            "risk_factors": {
                "rainfall_intensity": 0.55,
                "accumulated_rainfall": 0.48,
                "weather_severity": 0.50,
                "satellite_flood_detection": 0.60,
                "flood_depth_estimate": 0.50,
                "drainage_factor": 0.80,
                "elevation_factor": 0.70,
                "social_reports_density": 0.30,
                "historical_risk": 0.65,
                "river_level_factor": 0.55,
                "has_satellite_data": True,
                "has_social_data": False,
                "has_river_data": True,
                "satellite_confirmed_flooding": False,
            },
            "affected_area_km2": 8.2,
            "time_to_impact_hours": 3.0,
            "recommended_actions": [
                "Alert emergency response teams",
                "Prepare evacuation routes",
            ],
            "alert_level": "flood_risk",
        },
        {
            "zone": {
                "id": "00000000-0000-0000-0000-000000000003",
                "name": "Netrokona Sadar",
                "center": {"latitude": 24.8703, "longitude": 90.7279},
                "radius_km": 5.0,
                "risk_level": "high",
                "population_density": 22000,
                "elevation": 3.0,
                "drainage_capacity": "poor",
            },
            "risk_score": 0.45,
            "severity_level": "moderate",
            "confidence": 0.70,
            "risk_factors": {
                "rainfall_intensity": 0.30,
                "accumulated_rainfall": 0.28,
                "weather_severity": 0.25,
                "satellite_flood_detection": 0.20,
                "flood_depth_estimate": 0.10,
                "drainage_factor": 0.80,
                "elevation_factor": 0.70,
                "social_reports_density": 0.0,
                "historical_risk": 0.50,
                "river_level_factor": 0.30,
                "has_satellite_data": True,
                "has_social_data": False,
                "has_river_data": True,
                "satellite_confirmed_flooding": False,
            },
            "affected_area_km2": 2.1,
            "time_to_impact_hours": 6.0,
            "recommended_actions": ["Issue flood watch advisory"],
            "alert_level": "weather_warning",
        },
        {
            "zone": {
                "id": "00000000-0000-0000-0000-000000000004",
                "name": "Sirajganj Sadar",
                "center": {"latitude": 24.449, "longitude": 89.700},
                "radius_km": 5.0,
                "risk_level": "moderate",
                "population_density": 31000,
                "elevation": 4.0,
                "drainage_capacity": "poor",
            },
            "risk_score": 0.16,
            "severity_level": "minimal",
            "confidence": 0.68,
            "risk_factors": {
                "rainfall_intensity": 0.05,
                "accumulated_rainfall": 0.04,
                "weather_severity": 0.03,
                "satellite_flood_detection": 0.0,
                "flood_depth_estimate": 0.0,
                "drainage_factor": 0.80,
                "elevation_factor": 0.60,
                "social_reports_density": 0.0,
                "historical_risk": 0.20,
                "river_level_factor": 0.063,
                "has_satellite_data": True,
                "has_social_data": False,
                "has_river_data": True,
                "satellite_confirmed_flooding": False,
            },
            "affected_area_km2": 0.0,
            "time_to_impact_hours": None,
            "recommended_actions": ["Monitor weather conditions"],
            "alert_level": "all_clear",
        },
        {
            "zone": {
                "id": "00000000-0000-0000-0000-000000000005",
                "name": "Jamalpur Sadar",
                "center": {"latitude": 24.900, "longitude": 89.9333},
                "radius_km": 5.5,
                "risk_level": "high",
                "population_density": 29000,
                "elevation": 3.5,
                "drainage_capacity": "poor",
            },
            "risk_score": 0.16,
            "severity_level": "minimal",
            "confidence": 0.68,
            "risk_factors": {
                "rainfall_intensity": 0.04,
                "accumulated_rainfall": 0.03,
                "weather_severity": 0.02,
                "satellite_flood_detection": 0.0,
                "flood_depth_estimate": 0.0,
                "drainage_factor": 0.80,
                "elevation_factor": 0.70,
                "social_reports_density": 0.0,
                "historical_risk": 0.18,
                "river_level_factor": 0.009,
                "has_satellite_data": True,
                "has_social_data": False,
                "has_river_data": True,
                "satellite_confirmed_flooding": False,
            },
            "affected_area_km2": 0.0,
            "time_to_impact_hours": None,
            "recommended_actions": ["Monitor weather conditions"],
            "alert_level": "all_clear",
        },
    ],
    "alerts": [],
    "monitored_zones": [],
    "data_sources_status": {
        "weather_api": "operational",
        "social_media": "disabled",
        "spatial_db": "operational",
        "satellite_gee": "operational",
        "river_glofas": "operational (5/5 zones)",
    },
    "processing_time_seconds": 21.4,
    "next_update_in_seconds": 180,
}


def get_output() -> dict:
    """
    Public function used by all callbacks.
    Returns live data if Agent 1 is reachable, otherwise DEMO_OUTPUT.
    """
    data = fetch_agent_output()
    if data is None:
        logger.debug("Using demo data")
        return DEMO_OUTPUT
    return data
