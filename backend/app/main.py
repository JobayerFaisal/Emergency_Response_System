# backend/app/main.py

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .services.db import init_db, close_db
from .services.redis_bridge import init_redis, close_redis, start_bridge, stop_bridge, redis_ok
from .services.replay_simulator import ReplaySimulator
from .routers import dashboard, zones, predictions, distress, resources, dispatch
from .routers import kpi as kpi_router
from .websocket import router as ws_router
from .config.scenario import set_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)s %(message)s",
)
logger = logging.getLogger("dashboard.main")

replay_simulator = ReplaySimulator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═" * 60)
    logger.info("  Emergency Response System — Dashboard API  v1.0.0")
    logger.info("═" * 60)

    await init_db()
    await init_redis()
    await start_bridge()

    app.state.replay_simulator = replay_simulator

    if replay_simulator.enabled:
        logger.warning("⏪ Replay mode active: %s", replay_simulator.current_mode)
        if replay_simulator.scenario_date:
            logger.warning("⏪ Replay scenario timestamp: %s", replay_simulator.scenario_date)
        await replay_simulator.start()
    else:
        logger.info("🟢 Live mode active")

    logger.info("✅ Dashboard API ready on port 8005")
    yield

    logger.info("Shutting down Dashboard API...")

    await replay_simulator.stop()
    await stop_bridge()
    await close_redis()
    await close_db()

    logger.info("Dashboard API shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Emergency Response System — Dashboard API",
        version="1.0.0",
        lifespan=lifespan,
    )

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

    app.include_router(kpi_router.router)
    app.include_router(dashboard.router)
    app.include_router(zones.router)
    app.include_router(predictions.router)
    app.include_router(distress.router)
    app.include_router(resources.router)
    app.include_router(dispatch.router)
    app.include_router(ws_router)

    @app.get("/")
    async def root():
        status = app.state.replay_simulator.status()
        return {
            "service": "Emergency Response System — Dashboard API",
            "version": "1.0.0",
            "scenario": {
                "mode": status["mode"],
                "date": status["scenario_date"],
                "replay": status["enabled"],
                "running": status["running"],
                "paused": status["paused"],
                "tick": status["tick"],
            },
        }

    @app.get("/health")
    async def health():
        from app.services.db import get_pool

        db_status = "ok"
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as exc:
            db_status = f"error: {exc}"

        redis_status = "ok" if await redis_ok() else "unavailable"
        replay_status_data = app.state.replay_simulator.status()

        return {
            "status": "healthy" if db_status == "ok" else "degraded",
            "database": db_status,
            "redis": redis_status,
            "version": "1.0.0",
            "scenario_mode": replay_status_data["mode"],
            "scenario_date": replay_status_data["scenario_date"],
            "replay_running": replay_status_data["running"],
            "replay_paused": replay_status_data["paused"],
            "replay_tick": replay_status_data["tick"],
        }

    @app.get("/api/replay/status")
    async def replay_status():
        simulator = app.state.replay_simulator
        return simulator.status()

    @app.post("/api/replay/start")
    async def replay_start():
        simulator = app.state.replay_simulator
        set_mode("REPLAY_2022", "2022-06-17T09:00:00Z")
        await simulator.switch_to_replay()
        return simulator.status()

    @app.post("/api/replay/pause")
    async def replay_pause():
        simulator = app.state.replay_simulator
        if not simulator.enabled:
            raise HTTPException(status_code=400, detail="Replay mode is not active")
        await simulator.pause()
        return simulator.status()

    @app.post("/api/replay/reset")
    async def replay_reset():
        simulator = app.state.replay_simulator
        if not simulator.enabled:
            raise HTTPException(status_code=400, detail="Replay mode is not active")
        await simulator.reset()
        return simulator.status()

    @app.post("/api/replay/stop")
    async def replay_stop():
        simulator = app.state.replay_simulator
        set_mode("LIVE")
        await simulator.switch_to_live()
        return simulator.status()

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("DASHBOARD_PORT", "8005")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level="info",
    )