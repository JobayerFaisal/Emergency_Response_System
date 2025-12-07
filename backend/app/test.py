import asyncpg
import os
import json
from datetime import datetime

DATABASE_URL = os.getenv(
    "ENV_DB_URL",
    # "postgresql://postgres:postgres@host.docker.internal:5432/disaster_db"
    "postgresql+psycopg2://postgres:postgres@db:5432/disaster_db"
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
    LIMIT 6;
"""

# Function to serialize datetime to ISO format
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO format string
    raise TypeError("Type not serializable")

async def run_query():
    try:
        # Connect to the database
        conn = await asyncpg.connect(DATABASE_URL)
        print(f"Connected to {DATABASE_URL}")

        # Execute the query
        rows = await conn.fetch(QUERY)
        await conn.close()

        # Process the results and print
        if rows:
            print(f"Found {len(rows)} results.")
            # Serialize each row before printing
            results = [json.dumps(dict(row), default=json_serial, indent=2) for row in rows]
            for result in results:
                print(result)
        else:
            print("No results found.")

    except Exception as e:
        print(f"Error: {e}")

# Run the query
import asyncio
asyncio.run(run_query())
