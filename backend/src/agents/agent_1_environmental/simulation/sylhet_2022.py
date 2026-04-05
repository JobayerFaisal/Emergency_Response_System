"""
src/agents/agent_1_environmental/simulation/sylhet_2022.py

Sylhet 2022 Flood Simulation — Based on real historical data.

Real event facts (June 17-19, 2022):
  - Rainfall: 303.6 mm in a single day (June 18) — highest in 122 years
  - Total June rainfall: 2,456 mm vs normal 818 mm (3x above average)
  - 84% of Sylhet division submerged
  - 7.2 million people affected across Sylhet & Sunamganj
  - Surma river: 100cm+ above danger level
  - Kushiyara river: 74cm above danger level
  - 500,000 people evacuated to 1,432 shelters
  - Army, Navy, Air Force deployed for rescue
  - Airport and railway suspended
  - 53,000 hectares of agricultural land destroyed

Sources:
  - OCHA Bangladesh Flash Flood Situation Report (June 2022)
  - Bangladesh Red Crescent Society Situation Reports
  - Flood Forecasting & Warning Centre (FFWC) Bangladesh
  - NASA Earth Observatory
  - ReliefWeb MDRBD028 Final Report
"""

from datetime import datetime, timezone
from typing import List, Dict
from uuid import uuid4


# ── Simulation Zones (Sylhet Division) ───────────────────────────────────────
# Real coordinates of affected areas during the 2022 flood

SYLHET_2022_ZONES = [
    {
        "zone_id":            "sylhet-city-2022",
        "zone_name":          "Sylhet City (Surma River Basin)",
        "latitude":           24.8975,
        "longitude":          91.8720,
        "risk_score":         0.95,
        "severity_level":     "critical",
        "confidence":         0.97,
        "flood_area_pct":     84.0,        # 84% of Sylhet submerged
        "affected_people":    2_000_000,
        "rainfall_mm":        303.6,       # Single day June 18
        "river_above_danger": 100,         # cm above danger level (Surma)
        "medical_need":       True,
        "urgency":            "LIFE_THREATENING",
        "priority":           5,
        "risk_factors": {
            "rainfall_intensity":        1.0,
            "accumulated_rainfall":      1.0,
            "weather_severity":          0.98,
            "elevation_factor":          0.9,
            "drainage_factor":           1.0,
            "satellite_flood_detection": 0.95,
            "satellite_confirmed_flooding": True,
            "flood_depth_estimate":      2.5,
            "has_satellite_data":        True,
            "has_social_data":           True,
            "social_reports_density":    0.95,
            "historical_risk":           0.85,
        },
        "notes": "Surma river 100cm above danger level. Airport & railway suspended. "
                 "Army/Navy/Air Force deployed. 84% of city submerged.",
    },
    {
        "zone_id":            "sunamganj-2022",
        "zone_name":          "Sunamganj (Haor Region)",
        "latitude":           24.8660,
        "longitude":          91.3990,
        "risk_score":         0.98,
        "severity_level":     "critical",
        "confidence":         0.99,
        "flood_area_pct":     94.0,        # 94% of Sunamganj submerged
        "affected_people":    2_500_000,
        "rainfall_mm":        280.0,
        "river_above_danger": 120,
        "medical_need":       True,
        "urgency":            "LIFE_THREATENING",
        "priority":           5,
        "risk_factors": {
            "rainfall_intensity":        1.0,
            "accumulated_rainfall":      1.0,
            "weather_severity":          1.0,
            "elevation_factor":          1.0,
            "drainage_factor":           1.0,
            "satellite_flood_detection": 0.98,
            "satellite_confirmed_flooding": True,
            "flood_depth_estimate":      3.2,
            "has_satellite_data":        True,
            "has_social_data":           True,
            "social_reports_density":    0.98,
            "historical_risk":           0.90,
        },
        "notes": "94% submerged — worst affected district. Disconnected from rest of "
                 "country. Severe shortage of boats for rescue.",
    },
    {
        "zone_id":            "companiganj-2022",
        "zone_name":          "Companiganj Upazila, Sylhet",
        "latitude":           25.0333,
        "longitude":          91.6333,
        "risk_score":         0.91,
        "severity_level":     "critical",
        "confidence":         0.94,
        "flood_area_pct":     78.0,
        "affected_people":    46_500,
        "rainfall_mm":        250.0,
        "river_above_danger": 80,
        "medical_need":       True,
        "urgency":            "LIFE_THREATENING",
        "priority":           5,
        "risk_factors": {
            "rainfall_intensity":        0.95,
            "accumulated_rainfall":      0.95,
            "weather_severity":          0.93,
            "elevation_factor":          0.88,
            "drainage_factor":           0.95,
            "satellite_flood_detection": 0.90,
            "satellite_confirmed_flooding": True,
            "flood_depth_estimate":      2.1,
            "has_satellite_data":        True,
            "has_social_data":           True,
            "social_reports_density":    0.88,
            "historical_risk":           0.80,
        },
        "notes": "133 villages affected. 46,500 people displaced. "
                 "All road communications disrupted.",
    },
    {
        "zone_id":            "gowainghat-2022",
        "zone_name":          "Gowainghat Upazila, Sylhet",
        "latitude":           25.1000,
        "longitude":          92.0167,
        "risk_score":         0.88,
        "severity_level":     "critical",
        "confidence":         0.92,
        "flood_area_pct":     72.0,
        "affected_people":    69_165,
        "rainfall_mm":        220.0,
        "river_above_danger": 65,
        "medical_need":       True,
        "urgency":            "LIFE_THREATENING",
        "priority":           5,
        "risk_factors": {
            "rainfall_intensity":        0.92,
            "accumulated_rainfall":      0.90,
            "weather_severity":          0.89,
            "elevation_factor":          0.85,
            "drainage_factor":           0.92,
            "satellite_flood_detection": 0.87,
            "satellite_confirmed_flooding": True,
            "flood_depth_estimate":      1.9,
            "has_satellite_data":        True,
            "has_social_data":           True,
            "social_reports_density":    0.82,
            "historical_risk":           0.75,
        },
        "notes": "69,165 people affected across 10 unions. 42 shelters opened. "
                 "396 houses damaged.",
    },
    {
        "zone_id":            "kanaighat-2022",
        "zone_name":          "Kanaighat Upazila, Sylhet",
        "latitude":           24.9667,
        "longitude":          92.2667,
        "risk_score":         0.85,
        "severity_level":     "high",
        "confidence":         0.90,
        "flood_area_pct":     65.0,
        "affected_people":    72_300,
        "rainfall_mm":        195.0,
        "river_above_danger": 55,
        "medical_need":       True,
        "urgency":            "URGENT",
        "priority":           4,
        "risk_factors": {
            "rainfall_intensity":        0.88,
            "accumulated_rainfall":      0.87,
            "weather_severity":          0.85,
            "elevation_factor":          0.82,
            "drainage_factor":           0.88,
            "satellite_flood_detection": 0.83,
            "satellite_confirmed_flooding": True,
            "flood_depth_estimate":      1.6,
            "has_satellite_data":        True,
            "has_social_data":           True,
            "social_reports_density":    0.78,
            "historical_risk":           0.70,
        },
        "notes": "72,300 people affected. 329 houses damaged. "
                 "Kushiyara river 74cm above danger level nearby.",
    },
]


# ── Weather Data (June 17-18, 2022) ──────────────────────────────────────────

SYLHET_2022_WEATHER = {
    "condition":         "heavy rain",
    "temperature":       27.2,          # Celsius — typical June monsoon
    "humidity":          98,            # % — near saturation
    "pressure":          996,           # hPa — low pressure system
    "wind_speed":        12.5,          # m/s — strong monsoon winds
    "precipitation_1h":  38.0,          # mm/h — extreme rainfall rate
    "precipitation_3h":  95.0,          # mm
    "precipitation_24h": 303.6,         # mm — record single-day rainfall
    "visibility":        800,           # meters — heavy rain/mist
    "description":       "Extreme monsoon rainfall — worst in 122 years. "
                         "Cherrapunji upstream recorded 3rd highest rainfall in 122 years.",
}


# ── Simulation Scenarios ──────────────────────────────────────────────────────

SCENARIOS = {
    "peak": {
        "name":        "Peak Flood — June 18, 2022",
        "description": "Worst point of the flood. 84% of Sylhet submerged. "
                       "7.2 million affected. Army deployed.",
        "zones":       SYLHET_2022_ZONES,              # All 5 zones
        "weather":     SYLHET_2022_WEATHER,
    },
    "early": {
        "name":        "Early Flood — June 9-10, 2022",
        "description": "Flash flood begins. Rivers rising fast. "
                       "2 million people affected in Sylhet & Sunamganj.",
        "zones":       [
            {**z, "risk_score": z["risk_score"] * 0.65,
             "severity_level": "high" if z["risk_score"] * 0.65 >= 0.7 else "moderate",
             "affected_people": z["affected_people"] // 3}
            for z in SYLHET_2022_ZONES[:3]
        ],
        "weather": {**SYLHET_2022_WEATHER,
                    "precipitation_1h": 18.0,
                    "precipitation_24h": 145.0},
    },
    "single_zone": {
        "name":        "Single Zone — Sylhet City Only",
        "description": "Quick demo: one critical zone triggers full pipeline.",
        "zones":       [SYLHET_2022_ZONES[0]],          # Just Sylhet city
        "weather":     SYLHET_2022_WEATHER,
    },
}


def get_simulation_flood_alerts(scenario: str = "peak") -> List[Dict]:
    """
    Returns a list of flood_alert payloads matching Agent 1's publish format.
    Each alert will be picked up by Agent 2 → Agent 3 → Agent 4.
    """
    sc = SCENARIOS.get(scenario, SCENARIOS["peak"])
    alerts = []

    for zone in sc["zones"]:
        alert = {
            "message_id":    str(uuid4()),
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "sender_agent":  "agent_1_environmental",
            "receiver_agent": "agent_2_distress",
            "message_type":  "flood_alert",
            "zone_id":       zone["zone_id"],
            "priority":      zone["priority"],
            "payload": {
                "zone_id":        zone["zone_id"],
                "zone_name":      zone["zone_name"],
                "risk_score":     zone["risk_score"],
                "severity_level": zone["severity_level"],
                "confidence":     zone["confidence"],
                "risk_factors":   zone["risk_factors"],
                "timestamp":      datetime.now(timezone.utc).isoformat(),
                # Extra context for dashboard display
                "simulation":     True,
                "historical_event": "Sylhet 2022 Floods",
                "event_date":     "June 17-19, 2022",
                "flood_area_pct": zone["flood_area_pct"],
                "affected_people": zone["affected_people"],
                "rainfall_mm_24h": sc["weather"]["precipitation_24h"],
                "river_above_danger_cm": zone["river_above_danger"],
                "notes":          zone["notes"],
                "weather": {
                    "condition":         sc["weather"]["condition"],
                    "temperature":       sc["weather"]["temperature"],
                    "humidity":          sc["weather"]["humidity"],
                    "pressure":          sc["weather"]["pressure"],
                    "wind_speed":        sc["weather"]["wind_speed"],
                    "precipitation_1h":  sc["weather"]["precipitation_1h"],
                    "precipitation_24h": sc["weather"]["precipitation_24h"],
                },
            }
        }
        alerts.append(alert)

    return alerts


def get_scenario_summary(scenario: str = "peak") -> Dict:
    """Returns human-readable summary for display in Streamlit dashboard."""
    sc = SCENARIOS.get(scenario, SCENARIOS["peak"])
    total_affected = sum(z["affected_people"] for z in sc["zones"])
    max_risk = max(z["risk_score"] for z in sc["zones"])

    return {
        "scenario":        scenario,
        "name":            sc["name"],
        "description":     sc["description"],
        "zones_affected":  len(sc["zones"]),
        "total_people":    total_affected,
        "max_risk_score":  max_risk,
        "max_rainfall_mm": sc["weather"]["precipitation_24h"],
        "historical_event": "2022 Sylhet Floods — worst in 122 years",
        "source":          "OCHA, BDRCS, FFWC Bangladesh",
    }
