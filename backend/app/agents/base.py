# app/agents/base.py

import json
import threading
import time
from typing import Any, Dict, Optional

from app.core.redis_client import get_redis_client


class BaseAgent:
    """
    Docker-friendly Redis pub/sub agent that:
    - Waits for Redis to start
    - Avoids crashes if Redis is unavailable
    - Recovers if Redis disconnects
    - Passes static type checking cleanly
    """

    def __init__(self, input_channel: str):
        self.input_channel = input_channel
        self.redis: Optional[Any] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Start agent in background thread
    # ------------------------------------------------------------------
    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Wait until Redis is available
    # ------------------------------------------------------------------
    def _wait_for_redis(self) -> bool:
        while not self._stop_event.is_set():
            client = get_redis_client()
            if client is not None:
                try:
                    client.ping()
                    print(f"[{self.__class__.__name__}] Connected to Redis.")
                    self.redis = client
                    return True
                except Exception:
                    pass

            print(f"[{self.__class__.__name__}] Waiting for Redis...")
            time.sleep(2)

        return False

    # ------------------------------------------------------------------
    # Main worker loop
    # ------------------------------------------------------------------
    def _run(self):
        # Wait for redis before subscribing
        if not self._wait_for_redis():
            print(f"[{self.__class__.__name__}] Stopped before Redis became available.")
            return

        # Ensure redis is valid
        if self.redis is None:
            print(f"[{self.__class__.__name__}] Redis client is None. Cannot subscribe.")
            return

        # Subscribe safely
        try:
            pubsub = self.redis.pubsub()
            pubsub.subscribe(self.input_channel)
            print(f"[{self.__class__.__name__}] Subscribed to '{self.input_channel}'")
        except Exception as e:
            print(f"[{self.__class__.__name__}] Failed to subscribe: {e}")
            return

        # Listening loop
        while not self._stop_event.is_set():
            try:
                message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
                if message is None:
                    continue

                raw = message["data"]

                try:
                    payload = json.loads(raw)
                except Exception as e:
                    print(f"[{self.__class__.__name__}] JSON parse error: {e}")
                    continue

                try:
                    self.handle_message(payload)
                except Exception as e:
                    print(f"[{self.__class__.__name__}] Error handling message: {e}")

            except Exception as e:
                print(f"[{self.__class__.__name__}] Redis error: {e}")
                print(f"[{self.__class__.__name__}] Reconnecting to Redis...")
                time.sleep(2)
                self._wait_for_redis()

        print(f"[{self.__class__.__name__}] Listener stopped.")

    # ------------------------------------------------------------------
    # Must be overridden by subclasses
    # ------------------------------------------------------------------
    def handle_message(self, payload: Dict[str, Any]):
        raise NotImplementedError
