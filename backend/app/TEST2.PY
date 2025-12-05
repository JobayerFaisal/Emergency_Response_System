# backend/app/TEST2.py

import sys
import os

# Add backend/ to sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_ROOT)

from app.core.db import SessionLocal, Base, engine
from app.agents.agent_2_responder_chat.agent import ResponderChatAgent
from app.agents.agent_2_responder_chat.extractor import ExtractorAgent

Base.metadata.create_all(engine)

def main():
    db = SessionLocal()

    responder = ResponderChatAgent()
    extractor = ExtractorAgent()

    responder_id = "test-user"
    user_msg = "Two injured people are trapped and water is rising fast."

    print("\nUSER:", user_msg)

    reply = responder.reply(db, responder_id, user_msg)
    print("ASSISTANT:", reply)

    extracted = extractor.extract(responder_id, user_msg)
    print("\n📦 Extracted Emergency Data:", extracted)

    db.close()

if __name__ == "__main__":
    main()
