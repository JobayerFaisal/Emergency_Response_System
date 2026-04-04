"""
src/agents/agent_2_distress/redis_handler.py
Subscribe/publish helpers for Agent 2.

Subscribes to: flood_alert        (from Agent 1)
Publishes to:  distress_queue     (to Agent 3)
               agent_status       (heartbeat)
"""

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg
from redis import asyncio as aioredis

from shared.message_protocol import AgentMessage

logger = logging.getLogger("agent2.redis")

AGENT_ID = "agent_2_distress"


async def publish_message(
    redis: aioredis.Redis,
    db_pool: asyncpg.Pool,
    channel: str,
    receiver: str,
    message_type: str,
    payload: dict,
    zone_id: str = None,
    priority: int = 3,
) -> AgentMessage:
    msg = AgentMessage(
        sender_agent=AGENT_ID,
        receiver_agent=receiver,
        message_type=message_type,
        zone_id=zone_id,
        priority=priority,
        payload=payload,
    )
    # Publish to Redis (skip gracefully if unavailable)
    try:
        await redis.publish(channel, msg.model_dump_json())
        logger.info("Published %s → %s (priority=%d)", message_type, channel, priority)
    except Exception as e:
        logger.warning("Redis unavailable — skipping publish to %s: %s", channel, e)

    # Always log to DB
    await _log_to_db(db_pool, msg)
    return msg


async def _log_to_db(db_pool: asyncpg.Pool, msg: AgentMessage) -> None:
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO agent_messages
                    (message_id, timestamp, sender_agent, receiver_agent,
                     message_type, zone_id, priority, payload)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                msg.message_id, msg.timestamp.replace(tzinfo=None),
                msg.sender_agent, msg.receiver_agent,
                msg.message_type, msg.zone_id,
                msg.priority, json.dumps(msg.payload),
            )
    except Exception as e:
        logger.error("Failed to log message to DB: %s", e)


async def listen_flood_alerts(redis: aioredis.Redis, handler):
    """Subscribe to flood_alert channel and call handler for each message."""
    pubsub = redis.pubsub()
    await pubsub.subscribe("flood_alert")
    logger.info("Subscribed to flood_alert channel")
    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                data = json.loads(message["data"])
                envelope = AgentMessage(**data)
                await handler(envelope)
            except Exception as e:
                logger.error("Error handling flood_alert message: %s", e)
