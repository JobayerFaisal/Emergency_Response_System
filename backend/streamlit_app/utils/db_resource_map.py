import asyncpg
import pandas as pd
import asyncio
import os

DATABASE_URL = os.getenv("ENV_DB_URL")


async def fetch_data(query: str):
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch(query)
    await conn.close()
    return pd.DataFrame([dict(r) for r in rows])


# -------------------------
# Universal async wrapper
# -------------------------
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return loop.run_until_complete(coro)


def get_incidents():
    query = """
        SELECT 
            id,
            responder_id,
            hazards,
            urgency,
            timestamp,
            ST_Y(location::geometry) AS lat,
            ST_X(location::geometry) AS lon
        FROM emergency_reports
        JOIN chat_history ON chat_history.responder_id = emergency_reports.responder_id
        WHERE chat_history.latitude IS NOT NULL
        ORDER BY emergency_reports.timestamp DESC;
    """
    return run_async(fetch_data(query))


def add_tooltip(df, mode="team"):
    if mode == "team":
        df["_tooltip"] = (
            "Team: " + df["team_name"].astype(str) + "\n"
            "Type: " + df["team_type"].astype(str) + "\n"
            "Status: " + df["status"].astype(str)
        )
    else:
        df["_tooltip"] = (
            "Incident\n"
            "Urgency: " + df["urgency"].astype(str) + "\n"
            "Hazards: " + df["hazards"].astype(str)
        )
    return df


def get_teams():
    query = """
        SELECT 
            id,
            team_name,
            team_type,
            status,
            capacity,
            latitude AS lat,
            longitude AS lon
        FROM team_resources;
    """
    return run_async(fetch_data(query))


def add_team_icons(df):
    df["icon"] = df["team_type"].map({
        "rescue": "rescue.png",
        "medical": "medical.png",
        "fire": "fire.png",
        "delivery": "delivery.png",
    }).fillna("rescue.png")
    return df


def add_incident_icons(df):
    df["icon"] = "incident.png"
    return df


# ------------ Load Data ------------
incidents = add_incident_icons(get_incidents())
teams = add_team_icons(get_teams())
teams = add_tooltip(teams, mode="team")
incidents = add_tooltip(incidents, mode="incident")
