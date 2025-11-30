# backend/app/api/v1/rescue_requests.py

from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.core.redis_client import get_redis_client
import json

router = APIRouter()
redis_client = get_redis_client()


class RescueRequest(BaseModel):
    name: str 
    phone: str 
    lat: float 
    lon: float 
    details: str 


@router.post("/")
def create_rescue_request(request: RescueRequest):
    payload = request.dict()
    # Publish to Redis so the agent can handle it
    redis_client.publish("rescue_requests", json.dumps(payload))
    return {"status": "received", "payload": payload}
