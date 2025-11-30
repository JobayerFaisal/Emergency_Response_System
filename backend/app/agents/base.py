# app/agents/base.py
import json
import threading
from typing import Callable, Any, Dict

from app.core.redis_client import get_redis_client


class BaseAgent:
    def __init__(self, input_channel: str):
        self.input_channel = input_channel
        self.redis = get_redis_client()
        self._stop_event = threading.Event()

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        pubsub = self.redis.pubsub()
        pubsub.subscribe(self.input_channel)

        print(f"[{self.__class__.__name__}] Subscribed to '{self.input_channel}'")

        for message in pubsub.listen():
            if self._stop_event.is_set():
                print(f"[{self.__class__.__name__}] Stopping listener")
                break

            if message["type"] != "message":
                continue

            data = message["data"]
            try:
                text = data.decode("utf-8")
                payload = json.loads(text)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Failed to parse message: {e}")
                continue

            try:
                self.handle_message(payload)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error handling message: {e}")

    def handle_message(self, payload: Dict[str, Any]):
        raise NotImplementedError
