import os
import asyncpg
import pandas as pd
import asyncio

DATABASE_URL = os.getenv(
    "ENV_DB_URL",
    "postgresql://postgres:postgres@localhost:5432/disaster_db"
)

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

async def fetch_weather(limit=100):
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch(QUERY, limit)
    await conn.close()
    return pd.DataFrame([dict(row) for row in rows])

def get_weather(limit=100):
    return asyncio.run(fetch_weather(limit))
