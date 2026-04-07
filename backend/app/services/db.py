# backend/app/services/db.py
import json
import logging
import os
from typing import AsyncGenerator

import asyncpg

logger = logging.getLogger("dashboard.db")

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    # Decode PostgreSQL json/jsonb into Python dict/list automatically
    await conn.set_type_codec(
        "json",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def init_db() -> None:
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
        init=_init_connection,
    )
    logger.info("Dashboard DB pool created -> %s", database_url.split("@")[-1])


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        logger.info("Dashboard DB pool closed")
        _pool = None


async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    if _pool is None:
        raise RuntimeError("Database pool is not initialised. Was init_db() called?")

    async with _pool.acquire() as conn:
        yield conn


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialised.")
    return _pool