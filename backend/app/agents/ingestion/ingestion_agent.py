# app/agents/ingestion/ingestion_agent.py
import json
from typing import Any, Dict

from app.agents.base import BaseAgent
from app.core.redis_client import get_redis_client


class IngestionAgent(BaseAgent):
    OUTPUT_CHANNEL = "normalized_incidents"

    def __init__(self):
        super().__init__(input_channel="raw_incidents")
        self.publisher = get_redis_client()

    def handle_message(self, payload: Dict[str, Any]):
        print(f"[IngestionAgent] Received raw payload: {payload}")

        normalized = {
            "type": payload.get("type", "unknown"),
            "description": payload.get("description", ""),
            "source": payload.get("source", "unknown"),
            "lat": float(payload.get("lat", 0.0)),
            "lon": float(payload.get("lon", 0.0)),
        }

        print(f"[IngestionAgent] Normalized incident: {normalized}")

        self.publisher.publish(self.OUTPUT_CHANNEL, json.dumps(normalized))
        print(f"[IngestionAgent] Published to '{self.OUTPUT_CHANNEL}'")
