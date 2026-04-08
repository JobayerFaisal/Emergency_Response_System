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

    scenario_mode = os.getenv("SCENARIO_MODE", "LIVE")
    scenario_date = os.getenv("SCENARIO_DATE", "")

    await init_db()
    await init_redis()
    await start_bridge()

    app.state.replay_simulator = replay_simulator

    if scenario_mode.startswith("REPLAY"):
        logger.warning("⏪ Replay mode active: %s", scenario_mode)
        if scenario_date:
            logger.warning("⏪ Replay scenario timestamp: %s", scenario_date)
        await replay_simulator.start()
    else:
        logger.info("🟢 Live mode active")

    logger.info("✅ Dashboard API ready on port 8005")
    yield

    logger.info("Shutting down Dashboard API...")

    if scenario_mode.startswith("REPLAY"):
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
        scenario_mode = os.getenv("SCENARIO_MODE", "LIVE")
        scenario_date = os.getenv("SCENARIO_DATE", "")
        return {
            "service": "Emergency Response System — Dashboard API",
            "version": "1.0.0",
            "scenario": {
                "mode": scenario_mode,
                "date": scenario_date,
                "replay": scenario_mode.startswith("REPLAY"),
            },
        }

    @app.get("/health")
    async def health():
        from app.services.db import get_pool

        scenario_mode = os.getenv("SCENARIO_MODE", "LIVE")
        scenario_date = os.getenv("SCENARIO_DATE", "")

        db_status = "ok"
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as exc:
            db_status = f"error: {exc}"

        redis_status = "ok" if await redis_ok() else "unavailable"

        return {
            "status": "healthy" if db_status == "ok" else "degraded",
            "database": db_status,
            "redis": redis_status,
            "version": "1.0.0",
            "scenario_mode": scenario_mode,
            "scenario_date": scenario_date,
        }

    @app.get("/api/replay/status")
    async def replay_status():
        simulator = app.state.replay_simulator
        return simulator.status()

    @app.post("/api/replay/start")
    async def replay_start():
        simulator = app.state.replay_simulator
        if not simulator.enabled:
            raise HTTPException(status_code=400, detail="Replay mode is not enabled")
        await simulator.resume()
        return simulator.status()

    @app.post("/api/replay/pause")
    async def replay_pause():
        simulator = app.state.replay_simulator
        if not simulator.enabled:
            raise HTTPException(status_code=400, detail="Replay mode is not enabled")
        await simulator.pause()
        return simulator.status()

    @app.post("/api/replay/reset")
    async def replay_reset():
        simulator = app.state.replay_simulator
        if not simulator.enabled:
            raise HTTPException(status_code=400, detail="Replay mode is not enabled")
        await simulator.reset()
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