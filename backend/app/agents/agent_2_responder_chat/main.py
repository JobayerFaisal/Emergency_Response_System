# backend/app/agents/agent_2_responder_chat/main.py

import sys
import os

# Compute correct project root: D:/Emergency_Response_System/backend
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

from app.core.db import SessionLocal, Base, engine
from app.agents.agent_2_responder_chat.agent import ResponderChatAgent
from app.agents.agent_2_responder_chat.extractor import ExtractorAgent

Base.metadata.create_all(engine)

def main():
    db = SessionLocal()

    responder = ResponderChatAgent()
    extractor = ExtractorAgent()

    responder_id = "test-user"

    print("🚨 Minimal Chat System Test")
    print("---------------------------")

    user_msg = "Two injured people are trapped and water is rising fast."

    print(f"\nUSER: {user_msg}")

    reply = responder.reply(db, responder_id, user_msg)
    print(f"\nASSISTANT: {reply}")

    extracted = extractor.extract(responder_id, user_msg)
    print("\n📦 Extracted Emergency Data:")
    print(extracted)

    db.close()


if __name__ == "__main__":
    main()
