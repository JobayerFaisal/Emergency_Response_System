# backend/streamlit_app/db.py

import os
import asyncpg
import pandas as pd
import asyncio

DATABASE_URL = os.getenv(
    "ENV_DB_URL",
    "postgresql://postgres:postgres@db:5432/disaster_db"
    # "postgresql://postgres:postgres@host.docker.internal:5432/disaster_db"
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



def get_weather_full(limit=200):
    query = """
        SELECT 
            w.timestamp,
            z.name AS zone_name,
            ST_Y(w.location::geometry) AS latitude,
            ST_X(w.location::geometry) AS longitude,
            w.temperature,
            w.humidity,
            w.pressure,
            w.wind_speed,
            w.precipitation_1h AS precip_1h,
            w.precipitation_3h AS precip_3h,
            w.precipitation_24h AS precip_24h,
            w.condition AS weather_condition,
            w.raw_data
        FROM weather_data w
        JOIN sentinel_zones z ON w.zone_id = z.id
        ORDER BY w.timestamp DESC
        LIMIT $1;
    """

    try:
        conn = asyncio.run(asyncpg.connect(DATABASE_URL))
        rows = conn.fetch(query, limit)
        conn.close()

        df = pd.DataFrame([dict(r) for r in rows])
        return df

    except Exception as e:
        print("DB Error:", e)
        return pd.DataFrame()