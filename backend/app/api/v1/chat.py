# path: backend/app/api/v1/chat.py


from fastapi import APIRouter, WebSocket, Depends
from sqlalchemy.orm import Session
from app.core.db import SessionLocal
from app.agents.agent_2_responder_chat.agent import ResponderChatAgent
from app.agents.agent_2_responder_chat.repository import ChatMessage
from app.agents.agent_2_responder_chat.extractor import ExtractorAgent
import redis

router = APIRouter(prefix="/chat", tags=["Responder Chat"])

chat_agent = ResponderChatAgent()
extractor_agent = ExtractorAgent()
redis_client = redis.Redis(host="localhost", port=6379)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.websocket("/{responder_id}")
async def chat_socket(websocket: WebSocket, responder_id: str):
    db = SessionLocal()
    await websocket.accept()

    while True:
        msg = await websocket.receive_text()

        # Save user message
        user_msg = ChatMessage(
            responder_id=responder_id,
            role="user",
            message=msg
        )
        db.add(user_msg)
        db.commit()

        # Generate AI reply
        reply = chat_agent.reply(db, responder_id, msg)

        bot_msg = ChatMessage(
            responder_id=responder_id,
            role="assistant",
            message=reply
        )
        db.add(bot_msg)
        db.commit()

        await websocket.send_text(reply)

        # Extraction → Redis
        structured = extractor_agent.extract(responder_id, msg)
        if structured:
            redis_client.publish("reports.raw", structured.json())

    await websocket.close()
    db.close()
