import os
import time
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


def connect_redis():
    """
    Attempt a single Redis connection.
    Returns a Redis client or None.
    """
    try:
        client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()  # test actual connectivity
        return client

    except Exception as e:
        print(f"[Redis] ❌ Connection failed: {REDIS_URL} → {e}")
        return None


def get_redis_client(retries=5, delay=2):
    """
    Auto-retry Redis connection.
    Safe for API usage.
    """
    for attempt in range(retries):
        client = connect_redis()
        if client:
            print(f"[Redis] 🟢 Connected successfully (attempt {attempt+1})")
            return client

        print(f"[Redis] 🔄 Retry {attempt+1}/{retries} in {delay}s...")
        time.sleep(delay)

    print("[Redis] ❌ Giving up — returning None.")
    return None


def get_redis_client_auto():
    """
    Infinite retry loop for background agents (safe & recommended).
    NEVER returns None.
    """
    while True:
        client = connect_redis()
        if client:
            print("[Redis] 🟢 Connected successfully")
            return client

        print("[Redis] 🔄 Redis unavailable, retrying in 2s...")
        time.sleep(2)
