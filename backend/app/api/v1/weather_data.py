from fastapi import APIRouter, HTTPException
import asyncpg
import os

router = APIRouter()

# Database URL
DATABASE_URL = os.getenv(
    "ENV_DB_URL", 
    # "postgresql://postgres:postgres@localhost:5432/disaster_db"
    "postgresql://postgres:postgres@host.docker.internal:5432/disaster_db"
)

# Simplified SQL query to fetch weather data
QUERY = """
    SELECT 
        wd.timestamp,
        ST_Y(wd.location::geometry) AS latitude,
        ST_X(wd.location::geometry) AS longitude,
        wd.temperature,
        wd.humidity,
        wd.pressure,
        wd.wind_speed
    FROM weather_data wd
    ORDER BY wd.timestamp DESC
    LIMIT $1;
"""

@router.get("/")
async def get_weather_data(limit: int = 20):
    """Fetch latest weather data."""
    try:
        # Connect to the database
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Execute query with limit
        rows = await conn.fetch(QUERY, limit)
        await conn.close()

        # Return the results as a simple list of dictionaries
        return [{"timestamp": row["timestamp"], 
                 "latitude": row["latitude"], 
                 "longitude": row["longitude"], 
                 "temperature": row["temperature"], 
                 "humidity": row["humidity"], 
                 "pressure": row["pressure"], 
                 "wind_speed": row["wind_speed"]} for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
