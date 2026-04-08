# backend/src/agents/agent_3_resource/redis_handler.py

import json
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable

from redis import asyncio as aioredis
from pydantic import ValidationError

from shared.message_protocol import AgentMessage

logger = logging.getLogger("agent3.redis")


def _looks_like_raw_incident(data: dict) -> bool:
    if not isinstance(data, dict):
        return False

    if isinstance(data.get("incidents"), list):
        return True

    single_keys = {"id", "report_id", "text", "latitude", "longitude"}
    if any(k in data for k in single_keys):
        return True

    alt_keys = {"incident_id", "zone_name", "raw_message", "latitude", "longitude"}
    if any(k in data for k in alt_keys):
        return True

    return False


def _severity_to_priority(severity) -> int:
    try:
        sev = int(severity)
    except Exception:
        return 3
    return max(1, min(5, sev))


def _severity_to_urgency(severity) -> str:
    try:
        sev = int(severity)
    except Exception:
        return "MODERATE"

    if sev >= 5:
        return "LIFE_THREATENING"
    if sev >= 4:
        return "URGENT"
    return "MODERATE"


def _normalize_single_incident(item: dict) -> dict:
    if "incident_id" in item:
        return {
            "incident_id": item.get("incident_id"),
            "zone_id": item.get("zone_id") or item.get("district") or item.get("zone_name") or "unknown-zone",
            "zone_name": item.get("zone_name") or item.get("district") or "Unknown Zone",
            "raw_message": item.get("raw_message") or item.get("text") or item.get("message") or "Distress incident",
            "raw_location": item.get("raw_location") or item.get("district") or item.get("zone_name") or "Unknown",
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude"),
            "urgency": item.get("urgency") or "MODERATE",
            "num_people": item.get("num_people") or item.get("num_people_affected") or 0,
            "medical_need": item.get("medical_need") or item.get("requires_medical") or False,
            "priority": item.get("priority") or 3,
            "confidence": item.get("confidence") or item.get("credibility") or 0.8,
        }

    severity = item.get("severity", item.get("urgency", 3))
    priority = item.get("priority") or _severity_to_priority(severity)

    return {
        "incident_id": item.get("id") or item.get("report_id") or f"INC-{datetime.now(timezone.utc).timestamp()}",
        "zone_id": item.get("zone_id") or item.get("district") or item.get("zone_name") or "unknown-zone",
        "zone_name": item.get("zone_name") or item.get("district") or "Unknown Zone",
        "raw_message": item.get("text") or item.get("message") or "Distress incident",
        "raw_location": item.get("district") or item.get("zone_name") or "Unknown",
        "latitude": item.get("latitude"),
        "longitude": item.get("longitude"),
        "urgency": _severity_to_urgency(severity),
        "num_people": item.get("num_people") or 0,
        "medical_need": bool(item.get("medical_need", False)),
        "priority": priority,
        "confidence": item.get("credibility") or 0.8,
    }


def _wrap_legacy_distress(data: dict) -> AgentMessage:
    if "incidents" in data and isinstance(data["incidents"], list):
        incidents = [_normalize_single_incident(x) for x in data["incidents"]]
    else:
        incidents = [_normalize_single_incident(data)]

    top = incidents[0] if incidents else {}
    return AgentMessage(
        sender_agent="replay_engine",
        receiver_agent="agent_3_resource",
        message_type="distress_queue",
        zone_id=top.get("zone_id"),
        priority=int(top.get("priority", 3)),
        payload={"incidents": incidents},
        timestamp=datetime.now(timezone.utc),
    )


async def publish_message(
    redis,
    db_pool,
    channel: str,
    receiver: str,
    message_type: str,
    payload: dict,
    zone_id: str = None,
    priority: int = 3,
):
    envelope = AgentMessage(
        sender_agent="agent_3_resource",
        receiver_agent=receiver,
        message_type=message_type,
        zone_id=zone_id,
        priority=priority,
        payload=payload,
    )
    try:
        await redis.publish(channel, envelope.model_dump_json())
    except Exception as e:
        logger.error("Failed to publish message to %s: %s", channel, e)


async def listen_distress_queue(
    redis: aioredis.Redis,
    callback: Callable[[AgentMessage], Awaitable[None]],
) -> None:
    pubsub = redis.pubsub()
    await pubsub.subscribe("distress_queue")
    logger.info("Subscribed to distress_queue")

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            raw = message["data"]

            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except Exception as e:
                logger.error("Invalid JSON on distress_queue: %s", e)
                continue

            try:
                envelope = AgentMessage.model_validate(parsed)
            except ValidationError as ve:
                if _looks_like_raw_incident(parsed):
                    logger.warning(
                        "Legacy/raw distress_queue payload detected; auto-wrapping into AgentMessage"
                    )
                    envelope = _wrap_legacy_distress(parsed)
                else:
                    logger.error(
                        "Skipping malformed distress_queue message; not AgentMessage-compatible: %s | %s",
                        parsed,
                        ve.errors(),
                    )
                    continue

            try:
                await callback(envelope)
            except Exception as e:
                logger.exception("Error handling distress_queue message: %s", e)
    finally:
        await pubsub.unsubscribe("distress_queue")
        await pubsub.aclose()