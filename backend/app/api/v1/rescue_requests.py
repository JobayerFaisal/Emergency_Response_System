# path: backend/app/api/v1/rescue_requests.py

from fastapi import APIRouter, HTTPException
import os
import time
import redis

# -----------------------------
# FASTAPI ROUTER INITIALIZATION
# -----------------------------
router = APIRouter(prefix="/rescue", tags=["Rescue Requests"])

# -----------------------------
# REDIS CONFIGURATION
# -----------------------------
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
        client.ping()  # test connectivity
        return client

    except Exception as e:
        print(f"[Redis] ❌ Connection failed: {REDIS_URL} → {e}")
        return None


def get_redis_client(retries=5, delay=2):
    """
    Auto-retry Redis connection for API usage.
    """
    for attempt in range(1, retries + 1):
        client = connect_redis()
        if client:
            print(f"[Redis] 🟢 Connected (attempt {attempt})")
            return client

        print(f"[Redis] 🔄 Retry {attempt}/{retries} in {delay}s...")
        time.sleep(delay)

    print("[Redis] ❌ Giving up — returning None.")
    return None


def get_redis_client_auto():
    """
    Infinite retry loop for background agents.
    NEVER returns None.
    """
    while True:
        client = connect_redis()
        if client:
            print("[Redis] 🟢 Connected successfully")
            return client

        print("[Redis] 🔄 Redis unavailable, retrying in 2s...")
        time.sleep(2)


# ---------------------------------
# API ENDPOINTS FOR RESCUE REQUESTS
# ---------------------------------

@router.get("/status")
def rescue_status():
    """
    Simple test endpoint to confirm router is working.
    """
    return {"message": "Rescue Request API is running"}


@router.post("/add")
def add_rescue_request(request: dict):
    """
    Add a rescue request into Redis queue.
    """
    client = get_redis_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    client.lpush("rescue_queue", str(request))
    return {"status": "queued", "data": request}


@router.get("/pending")
def get_pending_requests():
    """
    Get all pending rescue requests from Redis.
    """
    client = get_redis_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    items = client.lrange("rescue_queue", 0, -1)
    return {"pending_requests": items}
