# backend/app/agents/dispatch/dispatch_agent.py

import json
import math
from typing import Any, Dict, List

from app.agents.base import BaseAgent
from app.core.redis_client import get_redis_client


class RescueDispatchAgent(BaseAgent):
    """
    Listens for rescue requests and automatically assigns
    the nearest available team.
    """

    OUTPUT_CHANNEL = "dispatch_orders"

    def __init__(self):
        super().__init__(input_channel="rescue_requests")
        self.publisher = get_redis_client()

        # For now, we simulate a few teams in memory.
        # Later you can load this from PostgreSQL.
        self.teams: List[Dict[str, Any]] = [
            {"id": "T1", "name": "Boat Team 1", "lat": 23.8100, "lon": 90.4120, "status": "available"},
            {"id": "T2", "name": "Rescue Team 2", "lat": 23.7500, "lon": 90.3900, "status": "available"},
            {"id": "T3", "name": "Ambulance Team 3", "lat": 23.9000, "lon": 90.4200, "status": "available"},
        ]

    def handle_message(self, payload: Dict[str, Any]):
        print(f"[RescueDispatchAgent] New rescue request: {payload}")

        lat = float(payload.get("lat", 0.0))
        lon = float(payload.get("lon", 0.0))

        team = self._select_nearest_available_team(lat, lon)

        if not team:
            print("[RescueDispatchAgent] No available teams!")
            # You could publish a 'no_team_available' event here
            return

        # Mark team as busy
        team["status"] = "en-route"

        dispatch_order = {
            "team_id": team["id"],
            "team_name": team["name"],
            "target_lat": lat,
            "target_lon": lon,
            "requester_name": payload.get("name"),
            "requester_phone": payload.get("phone"),
            "details": payload.get("details", ""),
        }

        print(f"[RescueDispatchAgent] Dispatching team {team['id']} to requester {payload.get('name')}")
        print(f"[RescueDispatchAgent] Dispatch order: {dispatch_order}")

        # Publish dispatch order so other parts of the system
        # (e.g. WebSocket / dashboard / SMS notifier) can react.
        self.publisher.publish(self.OUTPUT_CHANNEL, json.dumps(dispatch_order))

    def _select_nearest_available_team(self, lat: float, lon: float) -> Dict[str, Any] | None:
        available_teams = [t for t in self.teams if t["status"] == "available"]
        if not available_teams:
            return None

        def distance_sq(t):
            return (t["lat"] - lat) ** 2 + (t["lon"] - lon) ** 2

        nearest = min(available_teams, key=distance_sq)
        return nearest
