"""
backend/app/services/redis_bridge.py
=====================================
Reads ALL Redis pub/sub channels used by the agent pipeline and forwards
each message to every connected WebSocket client.

Channels listened to:
  flood_alert        (Agent 1 → Agent 2)
  distress_queue     (Agent 2 → Agent 3)
  dispatch_order     (Agent 3 → Agent 4)
  route_assignment   (Agent 4 → Dashboard)
  inventory_update   (Agent 3 → Dashboard)
  agent_status       (all agents heartbeats)

The bridge runs as a single long-lived asyncio task started in main.py
lifespan.  WebSocket clients register/deregister themselves via the
ConnectionManager (also defined here).
"""

import asyncio
import json
import logging
import os
from typing import Set

from redis import asyncio as aioredis
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

logger = logging.getLogger("dashboard.redis_bridge")

# ── Channels the dashboard cares about ───────────────────────────────────────
SUBSCRIBED_CHANNELS = [
    "flood_alert",
    "distress_queue",
    "dispatch_order",
    "route_assignment",
    "inventory_update",
    "agent_status",
]

# ── Module-level state ────────────────────────────────────────────────────────
_redis: aioredis.Redis | None = None
_bridge_task: asyncio.Task | None = None


# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    """
    Keeps track of all active WebSocket connections.
    Thread-safe for asyncio (single-threaded event loop).
    """

    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)
        logger.info("WS client connected  (total=%d)", len(self._active))

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)
        logger.info("WS client disconnected (total=%d)", len(self._active))

    async def broadcast(self, message: str) -> None:
        """Send a raw JSON string to all connected clients."""
        dead: Set[WebSocket] = set()
        for ws in list(self._active):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def active_connections(self) -> int:
        return len(self._active)


# Singleton manager — imported by websocket.py
manager = ConnectionManager()


# ── Redis initialisation / teardown ──────────────────────────────────────────

async def init_redis() -> None:
    """Create the Redis client.  Called once from app lifespan."""
    global _redis

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    _redis = aioredis.from_url(redis_url, decode_responses=True)

    try:
        await _redis.ping()
        logger.info("Redis bridge connected → %s", redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — bridge disabled", exc)
        _redis = None


async def close_redis() -> None:
    """Close the Redis client.  Called once from app lifespan."""
    global _redis
    if _redis:
        await _redis.aclose()
        logger.info("Redis bridge closed")
        _redis = None


# ── Background bridge task ────────────────────────────────────────────────────

async def start_bridge() -> None:
    """
    Start the pub/sub listener task.
    Returns immediately — the task runs in the background.
    """
    global _bridge_task
    if _redis is None:
        logger.warning("Redis not available — skipping bridge start")
        return

    _bridge_task = asyncio.create_task(_listen_forever())
    logger.info("Redis→WS bridge task started (channels=%s)", SUBSCRIBED_CHANNELS)


async def stop_bridge() -> None:
    global _bridge_task
    if _bridge_task and not _bridge_task.done():
        _bridge_task.cancel()
        try:
            await _bridge_task
        except asyncio.CancelledError:
            pass
    _bridge_task = None
    logger.info("Redis→WS bridge task stopped")


async def _listen_forever() -> None:
    """
    Subscribes to all agent channels and forwards every message to
    WebSocket clients.  Reconnects automatically if Redis drops.
    """
    while True:
        try:
            if _redis is None:
                await asyncio.sleep(10)
                continue

            pubsub = _redis.pubsub()
            await pubsub.subscribe(*SUBSCRIBED_CHANNELS)
            logger.info("Subscribed to channels: %s", SUBSCRIBED_CHANNELS)

            async for raw in pubsub.listen():
                if raw["type"] != "message":
                    continue

                channel: str = raw["channel"]
                data: str = raw["data"]

                # Try to parse as JSON to validate + enrich
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    payload = {"raw": data}

                # Wrap with channel metadata so the frontend knows what it is
                envelope = json.dumps(
                    {
                        "channel": channel,
                        "event": channel,          # alias for frontend switch
                        "data": payload,
                    }
                )

                if manager.active_connections > 0:
                    await manager.broadcast(envelope)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Bridge error: %s — reconnecting in 5s", exc)
            await asyncio.sleep(5)


# ── Direct publish helper (used by /api/dispatch POST) ───────────────────────

async def publish(channel: str, payload: dict) -> bool:
    """
    Publish a message directly to a Redis channel from the dashboard API.
    Returns True if published, False if Redis is unavailable.
    """
    if _redis is None:
        return False
    try:
        await _redis.publish(channel, json.dumps(payload))
        return True
    except Exception as exc:
        logger.error("publish() failed: %s", exc)
        return False


# ── Redis health helper ───────────────────────────────────────────────────────

async def redis_ok() -> bool:
    if _redis is None:
        return False
    try:
        await _redis.ping()
        return True
    except Exception:
        return False