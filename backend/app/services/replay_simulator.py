# backend/app/services/replay_simulator.py

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timezone

from redis import asyncio as aioredis

logger = logging.getLogger("dashboard.replay_simulator")


class ReplaySimulator:
    def __init__(self) -> None:
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.scenario_mode = os.getenv("SCENARIO_MODE", "LIVE")
        self.scenario_date = os.getenv("SCENARIO_DATE", "2022-06-17T09:00:00Z")

        self.replay_mode_name = (
            self.scenario_mode if self.scenario_mode.startswith("REPLAY") else "REPLAY_HISTORICAL"
        )

        self._mode_override: str | None = None
        self._redis: aioredis.Redis | None = None
        self._task: asyncio.Task | None = None

        self._running = False
        self._paused = False
        self._tick = 0

        # Slower replay pacing to avoid UI blinking
        self.step_interval_seconds = float(os.getenv("REPLAY_STEP_INTERVAL_SECONDS", "15"))
        self.phase_gap_seconds = float(os.getenv("REPLAY_PHASE_GAP_SECONDS", "2"))

        self.zones = [
            {
                "zone_name": "Sylhet Sadar",
                "center": [91.8720, 24.8975],
                "severity": 5,
                "reports": [
                    "Families stranded near low-lying housing block",
                    "Water rising quickly near market road",
                ],
            },
            {
                "zone_name": "Gowainghat",
                "center": [92.0167, 25.1000],
                "severity": 5,
                "reports": [
                    "Village access road submerged",
                    "Multiple households requesting evacuation support",
                ],
            },
            {
                "zone_name": "Companiganj",
                "center": [91.6333, 25.0333],
                "severity": 4,
                "reports": [
                    "Riverbank overflow reported by locals",
                    "School shelter requested for displaced residents",
                ],
            },
            {
                "zone_name": "Kanaighat",
                "center": [92.2667, 24.9667],
                "severity": 4,
                "reports": [
                    "Bridge approach flooded, transport slowed",
                    "Rescue assistance needed for elderly residents",
                ],
            },
        ]

        self.teams = [
            {
                "team_id": "TEAM-ALPHA",
                "base": [91.8685, 24.8940],
                "status": "ready",
                "skills": ["boat_rescue", "first_aid"],
                "equipment": ["boat", "medkit", "lifejackets"],
            },
            {
                "team_id": "TEAM-BRAVO",
                "base": [91.9000, 24.8200],
                "status": "ready",
                "skills": ["evacuation", "communications"],
                "equipment": ["truck", "radios", "food_packs"],
            },
            {
                "team_id": "TEAM-CHARLIE",
                "base": [92.0200, 24.7600],
                "status": "ready",
                "skills": ["medical", "search_support"],
                "equipment": ["ambulance", "medkit"],
            },
        ]

    @property
    def current_mode(self) -> str:
        return self._mode_override or self.scenario_mode

    @property
    def enabled(self) -> bool:
        return self.current_mode.startswith("REPLAY")

    async def start(self) -> None:
        if not self.enabled:
            logger.info("Replay simulator not started (mode=%s)", self.current_mode)
            return

        if self._running:
            self._paused = False
            logger.info("Replay simulator already running; resumed")
            return

        if self._redis is None:
            self._redis = aioredis.from_url(self.redis_url, decode_responses=True)

        self._running = True
        self._paused = False
        self._task = asyncio.create_task(self._run())
        logger.info(
            "Replay simulator started for %s (step_interval=%ss)",
            self.current_mode,
            self.step_interval_seconds,
        )

    async def stop(self) -> None:
        self._running = False
        self._paused = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._redis:
            await self._redis.aclose()
            self._redis = None

        logger.info("Replay simulator stopped")

    async def pause(self) -> None:
        if not self._running:
            return
        self._paused = True
        logger.info("Replay simulator paused")

    async def resume(self) -> None:
        if not self.enabled:
            logger.info("Replay simulator resume ignored (mode=%s)", self.current_mode)
            return
        if not self._running:
            await self.start()
            return
        self._paused = False
        logger.info("Replay simulator resumed")

    async def reset(self) -> None:
        self._tick = 0
        logger.info("Replay simulator reset")

    async def switch_to_live(self) -> None:
        await self.stop()
        self._mode_override = "LIVE"
        logger.info("Switched replay simulator to LIVE mode")

    async def switch_to_replay(self) -> None:
        was_live = self.current_mode == "LIVE"
        self._mode_override = self.replay_mode_name
        if was_live:
            self._tick = 0
        await self.start()
        logger.info("Switched replay simulator to %s mode", self.current_mode)

    def status(self) -> dict:
        return {
            "mode": self.current_mode,
            "enabled": self.enabled,
            "running": self._running,
            "paused": self._paused,
            "tick": self._tick,
            "scenario_mode": self.current_mode,
            "scenario_date": self.scenario_date,
            "step_interval_seconds": self.step_interval_seconds,
        }

    async def _publish(self, channel: str, payload: dict) -> None:
        if not self._redis:
            return
        await self._redis.publish(channel, json.dumps(payload))

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _jitter(self, lon: float, lat: float, dx: float = 0.01, dy: float = 0.01) -> tuple[float, float]:
        return (
            round(lon + random.uniform(-dx, dx), 6),
            round(lat + random.uniform(-dy, dy), 6),
        )

    async def _emit_team_statuses(self) -> None:
        for team in self.teams:
            lon, lat = self._jitter(team["base"][0], team["base"][1], 0.015, 0.015)
            payload = {
                "team_id": team["team_id"],
                "latitude": lat,
                "longitude": lon,
                "status": "en_route" if self._tick % 3 else "ready",
                "skills": team["skills"],
                "equipment": team["equipment"],
                "zone_name": "Sylhet Operations",
                "available": self._tick % 4 != 0,
                "timestamp": self._now(),
            }
            await self._publish("agent_status", payload)

    async def _emit_distress_event(self) -> None:
        zone = self.zones[self._tick % len(self.zones)]
        lon, lat = self._jitter(zone["center"][0], zone["center"][1], 0.02, 0.02)
        text = zone["reports"][self._tick % len(zone["reports"])]

        payload = {
            "id": f"DIST-{self._tick:03d}",
            "report_id": f"DIST-{self._tick:03d}",
            "zone_name": zone["zone_name"],
            "district": zone["zone_name"],
            "latitude": lat,
            "longitude": lon,
            "severity": zone["severity"],
            "urgency": zone["severity"],
            "credibility": 0.92,
            "text": text,
            "timestamp": self._now(),
        }
        await self._publish("distress_queue", payload)

    async def _emit_route_assignment(self) -> None:
        zone = self.zones[self._tick % len(self.zones)]
        team = self.teams[self._tick % len(self.teams)]

        start_lon, start_lat = team["base"]
        end_lon, end_lat = zone["center"]

        payload = {
            "dispatch_id": f"ROUTE-{self._tick:03d}",
            "mission_id": f"MISSION-{self._tick:03d}",
            "team_id": team["team_id"],
            "status": "ASSIGNED" if self._tick % 2 else "EN_ROUTE",
            "eta_minutes": 12 + (self._tick % 8),
            "distance_km": round(4.5 + (self._tick % 5) * 1.7, 1),
            "zone_name": zone["zone_name"],
            "route_geometry": {
                "type": "LineString",
                "coordinates": [
                    [start_lon, start_lat],
                    [round((start_lon + end_lon) / 2, 6), round((start_lat + end_lat) / 2 + 0.01, 6)],
                    [end_lon, end_lat],
                ],
            },
            "timestamp": self._now(),
        }
        await self._publish("route_assignment", payload)

    async def _emit_inventory_update(self) -> None:
        payload = {
            "zone_name": "Sylhet Logistics Hub",
            "boats_available": 6 - (self._tick % 3),
            "medical_kits": max(10, 42 - self._tick),
            "food_packs": max(40, 180 - (self._tick * 3)),
            "water_units": max(50, 260 - (self._tick * 4)),
            "status": "depleting" if self._tick > 3 else "stable",
            "timestamp": self._now(),
        }
        await self._publish("inventory_update", payload)

    async def _run(self) -> None:
        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(0.5)
                    continue

                self._tick += 1

                await self._emit_team_statuses()
                await asyncio.sleep(self.phase_gap_seconds)

                if self._paused or not self._running:
                    continue

                await self._emit_distress_event()
                await asyncio.sleep(self.phase_gap_seconds)

                if self._paused or not self._running:
                    continue

                await self._emit_route_assignment()

                if self._tick % 2 == 0 and not self._paused and self._running:
                    await asyncio.sleep(self.phase_gap_seconds)
                    await self._emit_inventory_update()

                await asyncio.sleep(self.step_interval_seconds)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Replay simulator loop error: %s", exc)
                await asyncio.sleep(3.0)