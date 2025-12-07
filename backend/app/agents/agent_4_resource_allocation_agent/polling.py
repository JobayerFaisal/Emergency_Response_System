import asyncio
import logging
from config import settings
from db import get_pending_reports
from dispatcher import allocate_resources
from db import save_dispatch

logger = logging.getLogger(__name__)


async def start_db_polling():
    logger.info("⏱ DB polling enabled")

    while True:
        reports = get_pending_reports()

        for r in reports:
            best_team, supplies, reasoning = allocate_resources(r)

            if best_team:
                save_dispatch(
                    r.id,
                    best_team.id,
                    supplies,
                    eta=10,
                    reasoning=reasoning
                )
                logger.info(f"✔ DB dispatch created for report {r.id}")

        await asyncio.sleep(settings.POLL_INTERVAL)
