"""
backend/app/main.py
====================
Dashboard API — FastAPI root application.
Runs on port 8005 (separate from the 4 agents on 8001-8004).

Startup sequence:
  1. init_db()        — asyncpg pool to PostgreSQL
  2. init_redis()     — aioredis client
  3. start_bridge()   — Redis pub/sub → WebSocket broadcast task

Shutdown sequence (reverse):
  1. stop_bridge()
  2. close_redis()
  3. close_db()

All routers are mounted under /api/* with CORS enabled for React dev server.
WebSocket endpoint is mounted at /ws.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.db import init_db, close_db
from app.services.redis_bridge import init_redis, close_redis, start_bridge, stop_bridge
from app.routers import dashboard, zones, predictions, distress, resources, dispatch
from app.websocket import router as ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)s %(message)s",
)
logger = logging.getLogger("dashboard.main")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook."""

    logger.info("═" * 60)
    logger.info("  Emergency Response System — Dashboard API  v1.0.0")
    logger.info("═" * 60)

    # ── Startup ──────────────────────────────────────────────────────────
    await init_db()
    await init_redis()
    await start_bridge()

    logger.info("✅ Dashboard API ready on port 8005")
    logger.info("   Routers : /api/dashboard | /api/zones | /api/predictions")
    logger.info("             /api/distress  | /api/resources | /api/dispatch")
    logger.info("   WebSocket: ws://localhost:8005/ws")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down Dashboard API...")
    await stop_bridge()
    await close_redis()
    await close_db()
    logger.info("Dashboard API shutdown complete")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Emergency Response System — Dashboard API",
        description=(
            "Real-time dashboard backend for the Bangladesh Flood Intelligence System. "
            "Aggregates data from all 4 AI agents and exposes REST + WebSocket endpoints "
            "consumed by the React frontend."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    # Allow React dev server (port 3000) and production build (port 80)
    allowed_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:80,http://frontend:3000",
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(dashboard.router)
    app.include_router(zones.router)
    app.include_router(predictions.router)
    app.include_router(distress.router)
    app.include_router(resources.router)
    app.include_router(dispatch.router)
    app.include_router(ws_router)

    # ── Health / root ─────────────────────────────────────────────────────
    @app.get("/", tags=["health"])
    async def root():
        return {
            "service":  "Emergency Response System — Dashboard API",
            "version":  "1.0.0",
            "status":   "operational",
            "endpoints": {
                "dashboard":   "/api/dashboard",
                "zones":       "/api/zones",
                "predictions": "/api/predictions",
                "distress":    "/api/distress",
                "resources":   "/api/resources",
                "dispatch":    "/api/dispatch",
                "websocket":   "/ws",
                "docs":        "/docs",
            },
        }

    @app.get("/health", tags=["health"])
    async def health():
        from app.services.redis_bridge import redis_ok
        from app.services.db import get_pool

        db_status = "ok"
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as exc:
            db_status = f"error: {exc}"

        redis_status = "ok" if await redis_ok() else "unavailable"

        return {
            "status":        "healthy" if db_status == "ok" else "degraded",
            "database":      db_status,
            "redis":         redis_status,
            "version":       "1.0.0",
        }

    return app


# ── Singleton used by uvicorn ─────────────────────────────────────────────────
app = create_app()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("DASHBOARD_PORT", "8005")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info",
    )