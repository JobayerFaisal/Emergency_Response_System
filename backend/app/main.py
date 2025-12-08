# backend/app/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
import json






# Routers
from app.api.v1.raw_incidents import router as raw_incidents_router
from app.api.v1.rescue_requests import router as rescue_requests_router
from app.api.v1 import environmental as environmental_router
from app.api.v1 import weather_data as weather_data_router
from app.api.v1.chat import router as chat_router
from app.api.v1 import emergency_reports as emergency_reports_router

# Agents
from app.agents.ingestion.ingestion_agent import IngestionAgent
from app.agents.dispatch import RescueDispatchAgent

# DB
from app.core.db import Base, engine


# -------------------------------------------------------------------
# Lifespan manager
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):

    # ---------------------------------------------------------------
    # Create DB tables (ONLY once here)
    # ---------------------------------------------------------------
    try:
        print("🌱 [DB] Initializing database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ [DB] Tables created / already present.")
    except Exception as e:
        print("❌ [DB] Failed to initialize tables:", e)

    # ---------------------------------------------------------------
    # Start background agents
    # ---------------------------------------------------------------
    ingestion_agent = IngestionAgent()
    dispatch_agent = RescueDispatchAgent()

    logging.info("[main] Starting ingestion agent...")
    ingestion_agent.start()

    logging.info("[main] Starting dispatch agent...")
    dispatch_agent.start()

    app.state.ingestion_agent = ingestion_agent
    app.state.dispatch_agent = dispatch_agent

    yield

    # ---------------------------------------------------------------
    # Cleanup on shutdown
    # ---------------------------------------------------------------
    logging.info("[main] Stopping ingestion agent...")
    ingestion_agent.stop()

    logging.info("[main] Stopping dispatch agent...")
    dispatch_agent.stop()


# -------------------------------------------------------------------
# FastAPI Init
# -------------------------------------------------------------------
app = FastAPI(
    title="Emergency Response Backend",
    description="Backend engine for disaster data ingestion, responder chat, and rescue dispatching.",
    version="1.0.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# -------------------------------------------------------------------
# WebSocket: Live Dispatch Stream
# -------------------------------------------------------------------
@app.websocket("/ws/dispatches")
async def websocket_dispatches(websocket: WebSocket):
    await websocket.accept()
    logging.info("[ws] Client connected → /ws/dispatches")

    try:
        while True:
            dispatch_event = {
                "team_id": "T1",
                "team_name": "Boat Team 100",
                "target_lat": 23.8103,
                "target_lon": 90.4125,
                "requester_name": "Jobayer Faisal Fahim",
                "requester_phone": "0170003300000",
                "details": "Water rising fast, need boat assistance."
            }

            await websocket.send_text(json.dumps(dispatch_event))
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        logging.info("[ws] Client disconnected.")


# -------------------------------------------------------------------
# Health Check
# -------------------------------------------------------------------
@app.get("/api/v1/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


# -------------------------------------------------------------------
# Register API Routers
# -------------------------------------------------------------------
app.include_router(raw_incidents_router, prefix="/api/v1/raw-incidents", tags=["Raw Incidents"])
app.include_router(rescue_requests_router, prefix="/api/v1/rescue-requests", tags=["Rescue Requests"])
app.include_router(environmental_router.router, prefix="/api/v1/environmental", tags=["Environmental Data"])
app.include_router(weather_data_router.router, prefix="/api/v1/weather", tags=["Weather Data"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Responder Chat"])
app.include_router(emergency_reports_router.router, prefix="/api/v1/emergency-reports", tags=["Emergency Reports"])
