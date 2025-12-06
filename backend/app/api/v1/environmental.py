from fastapi import APIRouter, HTTPException
import asyncpg
import os
import json
from datetime import datetime

router = APIRouter() 

DATABASE_URL = os.getenv(
    "ENV_DB_URL",
    "postgresql://postgres:postgres@host.docker.internal:5432/disaster_db"
)

QUERY = """
    SELECT 
        fp.timestamp,
        sz.name AS zone,
        ST_Y(sz.center::geometry) AS latitude,
        ST_X(sz.center::geometry) AS longitude,
        fp.risk_score,
        fp.severity_level,
        fp.confidence,
        fp.time_to_impact_hours,
        fp.affected_area_km2,
        fp.risk_factors,
        fp.recommended_actions
    FROM flood_predictions fp
    JOIN sentinel_zones sz ON fp.zone_id = sz.id
    ORDER BY fp.timestamp DESC
    LIMIT $1;
"""

# Function to serialize datetime to ISO format
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO format string
    raise TypeError("Type not serializable")

@router.get("/predictions")
async def get_environmental_predictions(limit: int = 20):
    """Return latest flood risk predictions from the database."""
    try:
        # Connect to the database
        conn = await asyncpg.connect(DATABASE_URL)
        print(f"Connected to {DATABASE_URL}")

        # Execute the query with the limit parameter
        rows = await conn.fetch(QUERY, limit)
        await conn.close()

        # Process the results and print
        predictions = []
        for row in rows:
            r = dict(row)

            # If the risk_factors or recommended_actions columns are stored as JSON in the database, parse them here
            if isinstance(r.get("risk_factors"), str):
                try:
                    r["risk_factors"] = json.loads(r["risk_factors"])
                except json.JSONDecodeError:
                    pass

            if isinstance(r.get("recommended_actions"), str):
                try:
                    r["recommended_actions"] = json.loads(r["recommended_actions"])
                except json.JSONDecodeError:
                    pass

            predictions.append(r)

        return {
            "count": len(predictions),
            "predictions": predictions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
