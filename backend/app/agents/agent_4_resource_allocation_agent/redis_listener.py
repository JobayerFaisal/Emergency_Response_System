import asyncio
import json
import logging
from backend.app.core.redis_client import get_redis_client
from dispatcher import allocate_resources
from db import save_dispatch

logger = logging.getLogger(__name__)


async def subscribe_to_reports():
    client = get_redis_client()
    if client is None:
            logger.error("❌ Redis client not available. Retrying in 3 seconds...")
            await asyncio.sleep(3)
            return await subscribe_to_reports()

    pubsub = client.pubsub(ignore_subscribe_messages=True)

    pubsub.subscribe("reports.raw")

    logger.info("📡 Agent 4 listening on channel: reports.raw")

    while True:
        message = pubsub.get_message()
        if message:
            try:
                data = json.loads(message["data"])
                report_id = data.get("id")

                logger.info(f"📥 Received emergency report: {report_id}")

                team, supplies, reasoning = allocate_resources(data)

                if team:
                    save_dispatch(
                        report_id=report_id,
                        team_id=team.id,
                        supplies=supplies,
                        eta=10,
                        reasoning=reasoning
                    )

                    logger.info(f"🚑 Dispatch created for report {report_id}")

                    client.publish(
                        "dispatch.action",
                        json.dumps({
                            "report_id": report_id,
                            "team": team.team_name,
                            "supplies": supplies,
                            "eta": 10
                        })
                    )

            except Exception as e:
                logger.error(f"[Redis Listener] Error: {e}")

        await asyncio.sleep(0.2)  # prevents high CPU usage
