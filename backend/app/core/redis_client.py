import os
import redis

# Simple: read from env, fallback to default
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


def get_redis_client() -> redis.Redis:
    return redis.from_url(REDIS_URL)
