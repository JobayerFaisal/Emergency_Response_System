import sys
import os
import json

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

    print("🚨 Agent-2 Operational Chat Test")
    print("================================")

    # UPDATED TEST MESSAGE FOR AGENT-2 OPERATIONAL EXTRACTION
    user_msg = (
        "We rescued 3 people but 2 are still trapped. "
        "Our boat engine failed and we need diesel and 2 ropes. "
        "One person is seriously injured and needs urgent treatment. "
        "Road to Sector 7 is blocked by debris."
    )

    print(f"\nUSER: {user_msg}")

    # Test agent reply
    reply = responder.reply(db, responder_id, user_msg)
    print(f"\nASSISTANT REPLY:\n{reply}")

    # Test structured extraction
    extracted = extractor.extract(responder_id, user_msg)

    print("\n📦 Extracted Operational Data:")
    print("--------------------------------")
    if extracted:
        print(json.dumps(extracted.dict(), indent=2))
    else:
        print("No extractable information.")

    db.close()


if __name__ == "__main__":
    main()
