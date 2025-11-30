from fastapi import FastAPI

from app.agents.ingestion.ingestion_agent import IngestionAgent
from app.api.v1.raw_incidents import router as raw_incidents_router

app = FastAPI(title="Emergency Response Backend")
ingestion_agent = IngestionAgent()

@app.on_event("startup")
def start_agents():
    print("[main] Starting ingestion agent...")
    ingestion_agent.start()

@app.on_event("shutdown")
def stop_agents():
    print("[main] Stopping ingestion agent...")
    ingestion_agent.stop()

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}

app.include_router(
    raw_incidents_router,
    prefix="/api/v1/raw-incidents",
    tags=["raw-incidents"],
)
