import asyncio
import logging

from config import settings
from backend.app.agents.agent_4_resource_allocation_agent.redis_listener import subscribe_to_reports
from polling import start_db_polling
from logger import setup_logging

logger = logging.getLogger(__name__)


async def main():
    setup_logging(settings.LOG_LEVEL)

    logger.info("🚀 Agent 4: Resource Allocation & Dispatch Optimization started")

    # Run Redis listener + DB polling concurrently
    await asyncio.gather(
        subscribe_to_reports(),
        start_db_polling()
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAgent stopped.")
