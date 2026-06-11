# backend/app/services/db.py
import json
import logging
import os
import pathlib
from typing import AsyncGenerator

import asyncpg

logger = logging.getLogger("dashboard.db")

_pool: asyncpg.Pool | None = None

# Directory containing ordered SQL migration files (001_*.sql, 002_*.sql …)
_MIGRATIONS_DIR = pathlib.Path(__file__).parent.parent.parent / "database"


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


async def _run_migrations(pool: asyncpg.Pool) -> None:
    """
    Run all SQL migration files in /database/ in filename order.

    Each file is wrapped in a transaction so a failure in one migration
    does not leave a half-applied schema.  Files that have already been
    applied are idempotent (every statement uses CREATE … IF NOT EXISTS
    / CREATE INDEX IF NOT EXISTS), so re-running them on restart is safe.
    """
    if not _MIGRATIONS_DIR.is_dir():
        logger.warning(
            "Migrations directory not found at %s — skipping", _MIGRATIONS_DIR
        )
        return

    sql_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        logger.warning("No .sql files found in %s", _MIGRATIONS_DIR)
        return

    logger.info("Running %d migration file(s) from %s", len(sql_files), _MIGRATIONS_DIR)

    async with pool.acquire() as conn:
        for sql_path in sql_files:
            sql = sql_path.read_text(encoding="utf-8")
            try:
                async with conn.transaction():
                    await conn.execute(sql)
                logger.info("  ✓ %s", sql_path.name)
            except Exception as exc:
                logger.error("  ✗ %s failed: %s", sql_path.name, exc)
                raise

    logger.info("All migrations applied successfully")


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

    # Apply all SQL migrations on every startup.
    # Statements are IF NOT EXISTS so this is safe to run repeatedly.
    await _run_migrations(_pool)


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