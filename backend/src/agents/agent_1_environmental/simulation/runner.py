"""
src/agents/agent_1_environmental/simulation/runner.py

Simulation runner — injects Sylhet 2022 flood data into the live pipeline.
Triggered via POST /simulate endpoint on Agent 1 (port 8001).
"""

import asyncio
import json
import logging
from typing import Optional

from redis import asyncio as aioredis

from .sylhet_2022 import get_simulation_flood_alerts, get_scenario_summary, SCENARIOS

logger = logging.getLogger("agent1.simulation")


class SimulationRunner:

    def __init__(self, redis_client: Optional[aioredis.Redis]):
        self.redis_client = redis_client

    async def run(self, scenario: str = "peak", delay_seconds: float = 1.0) -> dict:
        """
        Publishes Sylhet 2022 flood alerts to Redis → triggers full pipeline.
        Agent 2 picks them up → Agent 3 allocates → Agent 4 routes.

        Args:
            scenario: "peak" | "early" | "single_zone"
            delay_seconds: pause between zone alerts (for dramatic effect in demo)
        """
        if scenario not in SCENARIOS:
            return {
                "status": "error",
                "message": f"Unknown scenario '{scenario}'. "
                           f"Available: {list(SCENARIOS.keys())}"
            }

        alerts = get_simulation_flood_alerts(scenario)
        summary = get_scenario_summary(scenario)

        logger.info(
            "🌊 SIMULATION START: %s — %d zones, %s people affected",
            summary["name"], summary["zones_affected"],
            f"{summary['total_people']:,}"
        )

        published = 0
        skipped = 0

        for alert in alerts:
            zone_name = alert["payload"]["zone_name"]
            risk = alert["payload"]["risk_score"]

            if self.redis_client:
                try:
                    await self.redis_client.publish(
                        "flood_alert", json.dumps(alert)
                    )
                    logger.info(
                        "📡 SIM: Published flood_alert → %s (risk=%.2f, severity=%s)",
                        zone_name, risk, alert["payload"]["severity_level"]
                    )
                    published += 1
                    # Stagger alerts for demo effect
                    if delay_seconds > 0 and published < len(alerts):
                        await asyncio.sleep(delay_seconds)
                except Exception as e:
                    logger.error("Failed to publish simulation alert for %s: %s", zone_name, e)
                    skipped += 1
            else:
                logger.warning(
                    "Redis not available — cannot publish simulation for %s", zone_name
                )
                skipped += 1

        result = {
            "status":          "completed" if published > 0 else "failed",
            "scenario":        scenario,
            "scenario_name":   summary["name"],
            "description":     summary["description"],
            "zones_published": published,
            "zones_skipped":   skipped,
            "total_affected":  summary["total_people"],
            "max_risk_score":  summary["max_risk_score"],
            "max_rainfall_mm": summary["max_rainfall_mm"],
            "historical_event": summary["historical_event"],
            "source":          summary["source"],
            "message": (
                f"✅ {published} flood alerts published to Redis. "
                f"Agent 2 → 3 → 4 pipeline triggered."
                if published > 0
                else "❌ Redis not available. Start Redis first."
            )
        }

        logger.info(
            "🌊 SIMULATION COMPLETE: %d/%d zones published",
            published, len(alerts)
        )

        return result
