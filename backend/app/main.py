from fastapi import FastAPI

from app.agents.ingestion.ingestion_agent import IngestionAgent
from app.agents.dispatch import RescueDispatchAgent






from app.api.v1.raw_incidents import router as raw_incidents_router
from app.api.v1.rescue_requests import router as rescue_requests_router 







app = FastAPI(title="Emergency Response Backend")



ingestion_agent = IngestionAgent()
dispatch_agent = RescueDispatchAgent()



# Start and stop agents with the app lifecycle events
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


# Include API routers
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
