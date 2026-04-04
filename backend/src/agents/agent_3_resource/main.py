"""
src/agents/agent_3_resource/main.py
Agent 3 — Resource Management FastAPI service.

Port: 8003
Subscribes: distress_queue  (from Agent 2)
Publishes:  dispatch_order  (to Agent 4)
            inventory_update (to Dashboard)
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis import asyncio as aioredis
from uuid import UUID

from shared.message_protocol import AgentMessage
from src.agents.agent_3_resource.models import ResourceType, RestockRequest, InventorySnapshot
from src.agents.agent_3_resource.inventory_manager import InventoryManager
from .allocator import ResourceAllocator
from .redis_handler import publish_message, listen_distress_queue
from .seed_data import seed_initial_resources

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("agent3.main")

# ── App state ────────────────────────────────────────────────────────────────
db_pool: asyncpg.Pool = None
redis_client: aioredis.Redis = None
inv_manager: InventoryManager = None
allocator: ResourceAllocator = None
agent_state = {
    "connected": False,
    "last_action": None,
    "messages_processed": 0,
    "started_at": None,
}


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client, inv_manager, allocator

    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/disaster_response")
    REDIS_URL    = os.getenv("REDIS_URL",    "redis://localhost:6379")

    # DB
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    logger.info("PostgreSQL connected")

    # Redis
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Redis connected")

    # Services
    inv_manager = InventoryManager(db_pool)
    allocator   = ResourceAllocator(inv_manager)

    # Seed data (skips if table already populated)
    await seed_initial_resources(inv_manager)

    # Background listener
    agent_state["connected"] = True
    agent_state["started_at"] = datetime.now(timezone.utc).isoformat()
    listener_task = asyncio.create_task(
        listen_distress_queue(redis_client, handle_distress_queue)
    )

    # Heartbeat
    heartbeat_task = asyncio.create_task(send_heartbeat())

    yield

    listener_task.cancel()
    heartbeat_task.cancel()
    await db_pool.close()
    await redis_client.aclose()


app = FastAPI(title="Agent 3 — Resource Management", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Core message handler ──────────────────────────────────────────────────────

async def handle_distress_queue(envelope: AgentMessage):
    """
    Called every time Agent 2 publishes to distress_queue.
    Payload expected: {"incidents": [...]} — list of distress incidents.
    """
    agent_state["messages_processed"] += 1
    agent_state["last_action"] = f"Received distress_queue at {datetime.now(timezone.utc).isoformat()}"

    incidents = envelope.payload.get("incidents", [])
    if not incidents:
        logger.warning("Empty incidents list in distress_queue message")
        return

    logger.info("Processing %d incidents from Agent 2", len(incidents))
    allocations = await allocator.process_distress_queue(incidents)

    for alloc in allocations:
        # 1. Publish dispatch_order → Agent 4
        await publish_message(
            redis=redis_client,
            db_pool=db_pool,
            channel="dispatch_order",
            receiver="agent_4_dispatch",
            message_type="dispatch_order",
            payload=alloc.model_dump(mode="json"),
            zone_id=alloc.zone_id,
            priority=alloc.priority,
        )

        # 2. Publish inventory_update → Dashboard
        snapshot = await inv_manager.get_inventory_summary()
        await publish_message(
            redis=redis_client,
            db_pool=db_pool,
            channel="inventory_update",
            receiver="all",
            message_type="inventory_update",
            payload={"summary": snapshot, "allocation_id": str(alloc.allocation_id)},
            priority=2,
        )

    agent_state["last_action"] = (
        f"Processed {len(allocations)} allocations at {datetime.now(timezone.utc).isoformat()}"
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
                    "agent": "agent_3_resource",
                    "status": "healthy",
                    "messages_processed": agent_state["messages_processed"],
                },
                priority=1,
            )
        except Exception as e:
            logger.error("Heartbeat failed: %s", e)


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "agent": "Agent 3 — Resource Management",
        "version": "1.0.0",
        "subscribes_to": "distress_queue",
        "publishes_to": ["dispatch_order", "inventory_update", "agent_status"],
    }


@app.get("/health")
async def health():
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        await redis_client.ping()
        return {"status": "healthy", "db": "ok", "redis": "ok"}
    except Exception as e:
        raise HTTPException(503, detail=str(e))


@app.get("/inventory")
async def get_inventory():
    summary = await inv_manager.get_inventory_summary()
    return InventorySnapshot(resources=summary)


@app.get("/inventory/{resource_type}")
async def get_inventory_by_type(resource_type: str):
    try:
        rtype = ResourceType(resource_type)
    except ValueError:
        raise HTTPException(400, f"Unknown resource_type: {resource_type}")
    units = await inv_manager.get_available_units(rtype)
    return {"resource_type": resource_type, "units": units}


@app.post("/inventory/restock")
async def restock(req: RestockRequest):
    new_ids = await inv_manager.add_resource_units(
        resource_type=req.resource_type,
        name_prefix=f"{req.resource_type.value.replace('_','-').title()}-Restocked",
        quantity=req.quantity,
        lat=req.location.latitude,
        lon=req.location.longitude,
    )
    snapshot = await inv_manager.get_inventory_summary()

    await publish_message(
        redis=redis_client,
        db_pool=db_pool,
        channel="inventory_update",
        receiver="all",
        message_type="inventory_update",
        payload={"summary": snapshot, "restock": {"type": req.resource_type, "qty": req.quantity}},
        priority=2,
    )

    return {
        "status": "restocked",
        "resource_type": req.resource_type,
        "quantity_added": req.quantity,
        "new_unit_ids": [str(i) for i in new_ids],
    }


@app.get("/allocations")
async def get_allocations(limit: int = 20):
    return {"allocations": await inv_manager.get_recent_allocations(limit)}


@app.get("/allocations/{allocation_id}")
async def get_allocation(allocation_id: UUID):
    result = await inv_manager.get_allocation_by_id(allocation_id)
    if not result:
        raise HTTPException(404, "Allocation not found")
    return result


@app.post("/trigger")
async def manual_trigger(payload: dict):
    """
    Manually inject a distress_queue payload — useful for testing
    without Agent 2 running.

    Body: {"incidents": [...list of incident dicts...]}
    """
    incidents = payload.get("incidents", [])
    if not incidents:
        raise HTTPException(400, "incidents list is empty")

    fake_envelope = AgentMessage(
        sender_agent="manual_trigger",
        receiver_agent="agent_3_resource",
        message_type="distress_queue",
        payload={"incidents": incidents},
        priority=5,
    )
    await handle_distress_queue(fake_envelope)
    return {"status": "triggered", "incident_count": len(incidents)}


@app.get("/status")
async def get_status():
    return {
        "agent": "agent_3_resource",
        "connected": agent_state["connected"],
        "started_at": agent_state["started_at"],
        "messages_processed": agent_state["messages_processed"],
        "last_action": agent_state["last_action"],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AGENT_PORT", 8000))
    uvicorn.run("src.agents.agent_3_resource.main:app", host="0.0.0.0", port=port, reload=False)
