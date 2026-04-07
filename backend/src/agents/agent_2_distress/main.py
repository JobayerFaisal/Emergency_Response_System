# backend/src/agents/agent_2_distress/main.py

"""
src/agents/agent_2_distress/main.py
Agent 2 — Distress Intelligence FastAPI service.

Port: 8002
Subscribes: flood_alert     (from Agent 1)
Publishes:  distress_queue  (to Agent 3)
            agent_status    (heartbeat)

Responsibilities:
  - Listen for flood_alert messages from Agent 1
  - Convert environmental risk predictions into actionable distress incidents
  - Enrich incidents with zone population/geodata from PostgreSQL
  - Publish structured distress_queue messages to Agent 3
  - Expose REST API for manual triggers and status checks
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis import asyncio as aioredis

from shared.message_protocol import AgentMessage
from .models import DistressIncident, FloodAlert, UrgencyLevel
from .alert_processor import AlertProcessor
from .redis_handler import publish_message, listen_flood_alerts

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("agent2.main")

# ── App state ─────────────────────────────────────────────────────────────────
db_pool: asyncpg.Pool       = None
redis_client: aioredis.Redis = None
alert_processor: AlertProcessor = None

agent_state = {
    "connected":           False,
    "started_at":          None,
    "alerts_received":     0,
    "incidents_published": 0,
    "last_action":         None,
}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client, alert_processor

    DATABASE_URL = os.getenv(
        "DATABASE_URL_ASYNC",
        "postgresql://postgres:postgres@localhost:5432/disaster_response"
    )
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

    # ── Database ──────────────────────────────────────────────────────────
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    logger.info("PostgreSQL connected")

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        await redis_client.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning("Redis unavailable (%s) — running in REST-only mode", e)

    # ── Services ──────────────────────────────────────────────────────────
    alert_processor = AlertProcessor(db_pool)

    agent_state["connected"] = True
    agent_state["started_at"] = datetime.now(timezone.utc).isoformat()

    # Background listener for Agent 1 flood_alert messages
    listener_task = asyncio.create_task(
        _safe_listen_loop()
    )

    # Heartbeat
    heartbeat_task = asyncio.create_task(send_heartbeat())

    logger.info("✅ Agent 2 startup complete — listening for flood_alert")

    yield

    listener_task.cancel()
    heartbeat_task.cancel()
    await db_pool.close()
    try:
        await redis_client.aclose()
    except Exception:
        pass


async def _safe_listen_loop():
    """
    Wraps listen_flood_alerts with retry logic.
    If Redis is unavailable, retries every 30 seconds.
    """
    while True:
        try:
            await listen_flood_alerts(redis_client, handle_flood_alert)
        except Exception as e:
            logger.warning("Redis listener error: %s — retrying in 30s", e)
            await asyncio.sleep(30)


app = FastAPI(
    title="Agent 2 — Distress Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core handler ──────────────────────────────────────────────────────────────

async def handle_flood_alert(envelope: AgentMessage):
    """
    Called every time Agent 1 publishes a flood_alert.
    Converts the alert into distress incidents and publishes to distress_queue.
    """
    agent_state["alerts_received"] += 1
    agent_state["last_action"] = (
        f"Received flood_alert from {envelope.sender_agent} "
        f"at {datetime.now(timezone.utc).isoformat()}"
    )

    payload = envelope.payload
    logger.info(
        "flood_alert received: zone=%s risk=%.2f severity=%s",
        payload.get("zone_name", "?"),
        payload.get("risk_score", 0),
        payload.get("severity_level", "?"),
    )

    # Parse the alert
    try:
        alert = FloodAlert(
            zone_id=envelope.zone_id or payload.get("zone_id", "unknown"),
            zone_name=payload.get("zone_name", "Unknown Zone"),
            risk_score=float(payload.get("risk_score", 0)),
            severity_level=str(payload.get("severity_level", "minimal")),
            confidence=float(payload.get("confidence", 0.5)),
            risk_factors=payload.get("risk_factors", {}),
            timestamp=payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )
    except Exception as e:
        logger.error("Failed to parse flood_alert payload: %s", e)
        return

    # Process into incidents
    incidents = await alert_processor.process_flood_alert(alert)
    if not incidents:
        logger.info("No actionable incidents generated for zone %s", alert.zone_name)
        return

    # Publish distress_queue → Agent 3
    await _publish_distress_queue(incidents, zone_id=alert.zone_id)

    agent_state["incidents_published"] += len(incidents)
    agent_state["last_action"] = (
        f"Published {len(incidents)} incident(s) for {alert.zone_name} "
        f"at {datetime.now(timezone.utc).isoformat()}"
    )


async def _publish_distress_queue(
    incidents: List[DistressIncident],
    zone_id: Optional[str] = None,
):
    """Serialize incidents and publish to distress_queue channel."""
    incidents_payload = [
        {
            "incident_id":  inc.incident_id,
            "zone_id":      inc.zone_id,
            "zone_name":    inc.zone_name,
            "raw_message":  inc.raw_message,
            "raw_location": inc.raw_location,
            "latitude":     inc.latitude,
            "longitude":    inc.longitude,
            "urgency":      inc.urgency.value,
            "num_people":   inc.num_people,
            "medical_need": inc.medical_need,
            "priority":     inc.priority,
            "source":       inc.source.value,
            "confidence":   inc.confidence,
        }
        for inc in incidents
    ]

    max_priority = max(inc.priority for inc in incidents)

    await publish_message(
        redis=redis_client,
        db_pool=db_pool,
        channel="distress_queue",
        receiver="agent_3_resource",
        message_type="distress_queue",
        payload={"incidents": incidents_payload},
        zone_id=zone_id,
        priority=max_priority,
    )

    for inc in incidents:
        logger.info(
            "  → [%s] %s | people=%d medical=%s priority=%d",
            inc.urgency.value, inc.zone_name,
            inc.num_people, inc.medical_need, inc.priority,
        )


async def send_heartbeat():
    """Publish agent_status heartbeat every 30 seconds."""
    while True:
        await asyncio.sleep(30)
        try:
            await publish_message(
                redis=redis_client,
                db_pool=db_pool,
                channel="agent_status",
                receiver="all",
                message_type="heartbeat",
                payload={
                    "agent":               "agent_2_distress",
                    "status":              "healthy",
                    "alerts_received":     agent_state["alerts_received"],
                    "incidents_published": agent_state["incidents_published"],
                },
                priority=1,
            )
        except Exception as e:
            logger.warning("Heartbeat failed: %s", e)


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "agent":         "Agent 2 — Distress Intelligence",
        "version":       "1.0.0",
        "subscribes_to": "flood_alert",
        "publishes_to":  ["distress_queue", "agent_status"],
    }


@app.get("/health")
async def health():
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False

    try:
        await redis_client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    status = "healthy" if db_ok else "degraded"
    return {
        "status": status,
        "db":     "ok" if db_ok else "unavailable",
        "redis":  "ok" if redis_ok else "unavailable (REST-only mode)",
    }


@app.get("/status")
async def get_status():
    return {
        "agent":               "agent_2_distress",
        "connected":           agent_state["connected"],
        "started_at":          agent_state["started_at"],
        "alerts_received":     agent_state["alerts_received"],
        "incidents_published": agent_state["incidents_published"],
        "last_action":         agent_state["last_action"],
    }


@app.post("/trigger/flood-alert")
async def trigger_flood_alert(payload: dict):
    """
    Manually inject a flood_alert payload — simulates Agent 1 output.
    Useful for testing without Agent 1 running.

    Body example:
    {
        "zone_id": "fbfb97dc-830e-4103-b4fe-ff66fcd3035a",
        "zone_name": "Dhaka Central",
        "risk_score": 0.75,
        "severity_level": "high",
        "confidence": 0.85,
        "risk_factors": {}
    }
    """
    zone_id = payload.get("zone_id", "unknown")

    fake_envelope = AgentMessage(
        sender_agent="manual_trigger",
        receiver_agent="agent_2_distress",
        message_type="flood_alert",
        zone_id=zone_id,
        priority=5,
        payload=payload,
    )
    await handle_flood_alert(fake_envelope)
    return {
        "status": "triggered",
        "zone":   payload.get("zone_name", zone_id),
    }


@app.post("/trigger/distress-queue")
async def trigger_distress_queue(payload: dict):
    """
    Manually inject a pre-built distress_queue payload — bypasses NLP processing.
    Directly publishes incidents to Agent 3.

    Body: {"incidents": [...list of incident dicts...]}
    """
    incidents_raw = payload.get("incidents", [])
    if not incidents_raw:
        raise HTTPException(400, "incidents list is empty")

    incidents = []
    for inc in incidents_raw:
        try:
            incidents.append(DistressIncident(
                zone_id=inc.get("zone_id", "unknown"),
                zone_name=inc.get("zone_name", "Unknown"),
                raw_message=inc.get("raw_message", ""),
                raw_location=inc.get("raw_location", ""),
                latitude=float(inc["latitude"]),
                longitude=float(inc["longitude"]),
                urgency=UrgencyLevel(inc.get("urgency", "MODERATE")),
                num_people=int(inc.get("num_people", 0)),
                medical_need=bool(inc.get("medical_need", False)),
                priority=int(inc.get("priority", 3)),
            ))
        except Exception as e:
            raise HTTPException(400, f"Invalid incident data: {e}")

    await _publish_distress_queue(incidents)
    return {
        "status":         "triggered",
        "incident_count": len(incidents),
    }


@app.get("/zones")
async def get_monitored_zones():
    """List all sentinel zones from DB — shows what Agent 2 monitors."""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    id, name, risk_level,
                    ST_Y(center::geometry) AS lat,
                    ST_X(center::geometry) AS lon,
                    population_density, elevation, drainage_capacity,
                    last_monitored
                FROM sentinel_zones
                ORDER BY name
            """)
            return {
                "zones": [
                    {
                        "id":                 str(r["id"]),
                        "name":               r["name"],
                        "risk_level":         r["risk_level"],
                        "latitude":           r["lat"],
                        "longitude":          r["lon"],
                        "population_density": r["population_density"],
                        "elevation":          r["elevation"],
                        "drainage_capacity":  r["drainage_capacity"],
                        "last_monitored":     str(r["last_monitored"]) if r["last_monitored"] else None,
                    }
                    for r in rows
                ]
            }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/messages/recent")
async def get_recent_messages(limit: int = 20):
    """Show recent agent_messages log — useful for debugging the pipeline."""
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT message_id, timestamp, sender_agent, receiver_agent,
                       message_type, zone_id, priority
                FROM agent_messages
                WHERE sender_agent = 'agent_2_distress'
                   OR receiver_agent = 'agent_2_distress'
                ORDER BY timestamp DESC
                LIMIT $1
            """, limit)
            return {
                "messages": [
                    {
                        "message_id":    str(r["message_id"]),
                        "timestamp":     str(r["timestamp"]),
                        "sender":        r["sender_agent"],
                        "receiver":      r["receiver_agent"],
                        "type":          r["message_type"],
                        "zone_id":       r["zone_id"],
                        "priority":      r["priority"],
                    }
                    for r in rows
                ]
            }
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AGENT_PORT", "8002"))
    uvicorn.run(
        "src.agents.agent_2_distress.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
