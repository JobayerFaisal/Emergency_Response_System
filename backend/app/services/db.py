# backend/app/services/db.py
"""
backend/app/services/db.py
==========================
Shared asyncpg connection pool for the Dashboard API (port 8005).

All routers import `get_db` as a FastAPI dependency:

    from app.services.db import get_db
    async def my_route(conn=Depends(get_db)):
        rows = await conn.fetch("SELECT ...")

The pool is created once on startup (via `init_db`) and closed on shutdown
(via `close_db`).  Both are called from `app/main.py` lifespan.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg

logger = logging.getLogger("dashboard.db")

# ── module-level pool ─────────────────────────────────────────────────────────
_pool: asyncpg.Pool | None = None


async def init_db() -> None:
    """Create the asyncpg pool.  Called once during app startup."""
    global _pool

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/disaster_response",
    )

    _pool = await asyncpg.create_pool(
        database_url,
        min_size=2,
        max_size=20,
        command_timeout=30,
    )
    logger.info("Dashboard DB pool created → %s", database_url.split("@")[-1])


async def close_db() -> None:
    """Close the asyncpg pool.  Called once during app shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        logger.info("Dashboard DB pool closed")
        _pool = None


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    FastAPI dependency — yields a live connection from the shared pool.

    Usage:
        @router.get("/example")
        async def example(conn: asyncpg.Connection = Depends(get_db)):
            return await conn.fetch("SELECT 1")
    """
    if _pool is None:
        raise RuntimeError("Database pool is not initialised.  Was init_db() called?")

    async with _pool.acquire() as conn:
        yield conn


# ── Raw pool accessor (for services that need multiple queries in one tx) ────

def get_pool() -> asyncpg.Pool:
    """Return the raw pool.  Raises if init_db() was not called."""
    if _pool is None:
        raise RuntimeError("Database pool is not initialised.")
    return _pool