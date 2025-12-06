# backend/app/api/v1/chat.py


from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.db import SessionLocal
from app.agents.agent_2_responder_chat.agent import ResponderChatAgent
from app.agents.agent_2_responder_chat.extractor import ExtractorAgent
import asyncio
import logging
import json
import redis.asyncio as aioredis  # async redis client

router = APIRouter(tags=["Responder Chat"])  # NOTE: no prefix here

chat_agent = ResponderChatAgent()
extractor_agent = ExtractorAgent()

# Async Redis client
redis_client = aioredis.Redis(host="localhost", port=6379, decode_responses=True)


@router.websocket("/{responder_id}")
async def chat_socket(websocket: WebSocket, responder_id: str):
    """
    Real-time chat endpoint for field responders.
    Handles:
    - AI conversation
    - Extraction pipeline
    - Redis publishing
    - Connection lifecycle
    """

    await websocket.accept()
    logging.info(f"[chat] WebSocket connected → responder_id={responder_id}")

    # Create DB session (but do not block loop)
    db = SessionLocal()

    try:
        while True:
            try:
                # Receive the incoming message
                msg = await websocket.receive_text()
                logging.info(f"[chat] Received from {responder_id}: {msg}")

            except WebSocketDisconnect:
                logging.info(f"[chat] Client disconnected: {responder_id}")
                break

            except Exception as e:
                logging.error(f"[chat] Error receiving message: {e}")
                await websocket.send_text("Error receiving your message.")
                continue

            # Generate AI reply (sync → run in thread)
            reply = await asyncio.to_thread(chat_agent.reply, db, responder_id, msg)

            # Send reply
            await websocket.send_text(reply)
            logging.info(f"[chat] Sent AI reply to {responder_id}")

            # Extract structured info (sync → run in thread)
            structured = await asyncio.to_thread(extractor_agent.extract, responder_id, msg)

            if structured:
                logging.info(f"[chat] Extracted structured report: {structured.json()}")

                # 1️⃣ Save to PostgreSQL
                from app.agents.agent_2_responder_chat.repository import save_emergency_report
                await asyncio.to_thread(save_emergency_report, db, structured)

                # 2️⃣ Publish to Redis
                try:
                    await redis_client.publish("reports.raw", structured.json())
                except Exception as e:
                    logging.error(f"[chat] Redis publish failed: {e}")

        # Loop ends → disconnect
        await websocket.close()

    finally:
        # Always close DB session
        db.close()
        logging.info(f"[chat] Closed DB session for {responder_id}")
