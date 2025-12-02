import asyncpg
import os
from geopy.distance import geodesic

DATABASE_URL = os.getenv("ENV_DB_URL")

async def get_nearest_weather(lat, lon):
    query = """
        SELECT *,
        ST_Y(location::geometry) AS lat,
        ST_X(location::geometry) AS lon
        FROM weather_data;
    """

    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch(query)
    await conn.close()

    # Choose nearest weather station
    nearest = min(rows, key=lambda r: geodesic((lat, lon), (r["lat"], r["lon"])).km)
    return dict(nearest)


def calculate_flood_risk(weather):
    score = 0

    # Simple rules (can be replaced with an AI model later)
    if weather["humidity"] > 80:
        score += 20
    if weather["pressure"] < 1005:
        score += 20
    if weather["wind_speed"] > 10:
        score += 10
    if weather["temperature"] < 20:
        score += 10

    if score < 20:
        return "low", score
    elif score < 40:
        return "moderate", score
    elif score < 60:
        return "high", score
    else:
        return "extreme", score


async def validate_request(report):
    lat = report["latitude"]
    lon = report["longitude"]

    weather = await get_nearest_weather(lat, lon)
    risk_level, score = calculate_flood_risk(weather)

    # Claim validity example:
    claim_validity = (risk_level in ["high", "extreme"])

    return {
        "flood_risk_level": risk_level,
        "risk_score": score,
        "claim_validity": claim_validity,
        "validation_notes": f"Nearest station detected risk: {risk_level}"
    }
