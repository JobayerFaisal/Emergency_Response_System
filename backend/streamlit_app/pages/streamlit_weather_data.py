import os
import asyncpg
import pandas as pd
import asyncio

# Database URL (use your ENV or default)
DATABASE_URL = os.getenv(
    "ENV_DB_URL",
    "postgresql://postgres:postgres@localhost:5432/disaster_db"
)

# SQL Query for weather data
QUERY = """
    SELECT 
        timestamp,
        ST_Y(location::geometry) AS latitude,
        ST_X(location::geometry) AS longitude,
        temperature,
        humidity,
        pressure,
        wind_speed
    FROM weather_data
    ORDER BY timestamp DESC
    LIMIT $1;
"""

async def fetch_weather(limit=20):
    """Fetch latest weather data and return a Pandas DataFrame."""
    conn = await asyncpg.connect(DATABASE_URL)

    rows = await conn.fetch(QUERY, limit)
    await conn.close()

    # Convert to DataFrame
    df = pd.DataFrame([dict(r) for r in rows])
    return df

def get_weather_df(limit=20):
    """Sync wrapper for Streamlit."""
    return asyncio.run(fetch_weather(limit))
