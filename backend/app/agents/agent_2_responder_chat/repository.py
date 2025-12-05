# path: backend/app/agents/agent_2_responder_chat/repository.py

from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from app.core.db import Base

class ChatMessage(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    responder_id = Column(String, index=True)
    role = Column(String)  # "user" or "assistant"
    message = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
