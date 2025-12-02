from fastapi import APIRouter
from pydantic import BaseModel
from app.core.redis_client import get_redis_client
import json

router = APIRouter() 

redis_client = get_redis_client()


class RawIncident(BaseModel):
    type: str
    description: str
    lat: float
    lon: float
    source: str = "api"


@router.post("/")
def publish_raw_incident(incident: RawIncident):
    payload = incident.dict()
    redis_client.publish("raw_incidents", json.dumps(payload))
    return {"status": "published", "payload": payload}
