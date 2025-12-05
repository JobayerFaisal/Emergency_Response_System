# backend/app/api/v1/chat.py

from fastapi import APIRouter, WebSocket
from app.core.db import SessionLocal
from app.agents.agent_2_responder_chat.agent import ResponderChatAgent
from app.agents.agent_2_responder_chat.extractor import ExtractorAgent
import redis

router = APIRouter(prefix="/chat", tags=["Responder Chat"])

chat_agent = ResponderChatAgent()
extractor_agent = ExtractorAgent()
redis_client = redis.Redis(host="localhost", port=6379)

@router.websocket("/{responder_id}")
async def chat_socket(websocket: WebSocket, responder_id: str):
    await websocket.accept()
    db = SessionLocal()

    while True:
        msg = await websocket.receive_text()

        reply = chat_agent.reply(db, responder_id, msg)
        await websocket.send_text(reply)

        structured = extractor_agent.extract(responder_id, msg)
        if structured:
            redis_client.publish("reports.raw", structured.json())

    db.close()
