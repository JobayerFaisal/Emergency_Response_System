from fastapi import APIRouter
import asyncpg
import os
from app.agents.routing.route_agent import find_route

router = APIRouter()
DATABASE_URL = os.getenv("ENV_DB_URL")

@router.post("/route/{report_id}")
async def create_route(report_id: str):
    conn = await asyncpg.connect(DATABASE_URL)

    report = await conn.fetchrow("SELECT * FROM citizen_reports WHERE id=$1", report_id)
    base = await conn.fetchrow("SELECT * FROM rescue_team_base LIMIT 1")

    route = find_route(
        base["latitude"], base["longitude"],
        report["latitude"], report["longitude"]
    )

    await conn.execute(
        """
        INSERT INTO citizen_report_routes (report_id, distance_km, duration_minutes, polyline, route_instructions)
        VALUES ($1, $2, $3, $4, $5)
        """,
        report_id,
        route["distance_km"],
        route["duration_minutes"],
        route["polyline"],
        route["instructions"],
    )

    await conn.close()

    return route
