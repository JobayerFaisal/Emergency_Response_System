# backend/app/main.py


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import logging
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware


from app.agents.ingestion.ingestion_agent import IngestionAgent
from app.agents.dispatch import RescueDispatchAgent


from app.api.v1.raw_incidents import router as raw_incidents_router
from app.api.v1.rescue_requests import router as rescue_requests_router
from app.api.v1 import environmental as environmental_router
from app.api.v1 import weather_data as weather_data_router  # Import the weather data router
from app.api.v1.chat import router as chat_router

# Update this to allow the frontend container to access the backend container
origins = [
    "http://disaster_frontend:3000",  # Allow frontend in Docker to access backend
    "http://localhost:3000",           # Allow frontend on the host to access backend (for local testing)
]


# Create FastAPI app instance
app = FastAPI(title="Emergency Response Backend")

# Initialize agents
ingestion_agent = IngestionAgent()
dispatch_agent = RescueDispatchAgent()



app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket connection for dispatch orders
@app.websocket("/ws/dispatches")
async def websocket_dispatches(websocket: WebSocket):
    """
    WebSocket route to stream dispatch orders in real-time.
    """
    await websocket.accept()
    logging.info("[ws] Client connected to /ws/dispatches")

    try:
        # This is where the real dispatch data will be streamed
        # Example loop simulating dispatch orders
        while True:
            # Simulate a dispatch order message (you would actually get this from Redis or your agent)
            # In your real application, you'll get dispatches from Redis and send that
            dispatch = {
                "team_id": "T1",
                "team_name": "Boat Team 100",
                "target_lat": 23.8103,
                "target_lon": 90.4125,
                "requester_name": "Jobayer Faisal Fahim ",
                "requester_phone": "0170003300000",
                "details": "Water rising fast, need boat."
            }
            # Send the dispatch data to the connected WebSocket clients
            await websocket.send_text(json.dumps(dispatch))

            # For demo purposes, sending every 5 seconds. You can adjust this based on your real flow.
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        logging.info("[ws] Client disconnected from /ws/dispatches")


# Startup and shutdown event handlers for agents
@app.on_event("startup")
def start_agents():
    print("[main] Starting ingestion agent...")
    ingestion_agent.start()

    print("[main] Starting rescue dispatch agent...")
    dispatch_agent.start()


@app.on_event("shutdown")
def stop_agents():
    print("[main] Stopping ingestion agent...")
    ingestion_agent.stop()

    print("[main] Stopping rescue dispatch agent...")
    dispatch_agent.stop()


# Basic health check endpoint
@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}


# Include API routers for raw incidents and rescue requests
app.include_router(
    raw_incidents_router,
    prefix="/api/v1/raw-incidents",
    tags=["raw-incidents"],
)

app.include_router(
    rescue_requests_router,
    prefix="/api/v1/rescue-requests",
    tags=["rescue-requests"],
)

app.include_router(
    environmental_router.router,
    prefix="/api/v1/environmental",
    tags=["environmental"],
)


# Include the weather data router
app.include_router(
    weather_data_router.router,  # Include the weather data router
    prefix="/api/v1/weather-data",  # Base path for weather data
    tags=["weather-data"],
)

app.include_router(
    chat_router,                # Include the chat router
    prefix="/api/v1/chat",      # Base path for chat
    tags=["Responder Chat"],    # Tag for the chat endpoints
)


