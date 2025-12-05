# app/core/redis_client.py

import os
import redis

def get_redis_client():
    """
    Returns a Redis client configured for Docker Compose networking.
    Does NOT force a ping() here to avoid race condition blocking.
    BaseAgent will test connectivity when needed.
    """
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

    try:
        client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,   # quick fail if redis is not yet available
            socket_timeout=2
        )
        return client

    except Exception as e:
        print(f"[Redis] ERROR: Invalid Redis URL {redis_url}: {e}")
        return None
