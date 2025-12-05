# path: backend/app/agents/agent_2_responder_chat/agent.py

from openai import OpenAI
from sqlalchemy.orm import Session
from app.agents.agent_2_responder_chat.repository import ChatMessage

# Typed imports required for SDK 2.9.0
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam

client = OpenAI()

SYSTEM_PROMPT = """
You are an emergency response assistant. Your job is to help field responders quickly,
ask clarifying questions, and stay focused on actionable emergency information only.
Keep messages short and clear.
"""

class ResponderChatAgent:

    def _extract_text(self, content):
        # content always comes as list[ContentPart]
        if isinstance(content, list):
            combined = ""
            for part in content:
                if hasattr(part, "text") and part.text:
                    combined += part.text
            return combined.strip()

        if isinstance(content, str):
            return content.strip()

        return ""

    def reply(self, db: Session, responder_id: str, message: str):

        # Load chat history from DB
        history_rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.responder_id == responder_id)
            .order_by(ChatMessage.timestamp.desc())
            .limit(10)
            .all()
        )
        history_rows = list(reversed(history_rows))

        # -----------------------------
        # Build typed message list
        # -----------------------------
        messages: list[ChatCompletionMessageParam] = []

        # SYSTEM message
        messages.append(
            ChatCompletionSystemMessageParam(
                role="system",
                content=[ChatCompletionContentPartTextParam(type="text", text=SYSTEM_PROMPT)]
            )
        )

        # HISTORY messages
        for row in history_rows:
            role_value = str(row.role)        # FORCE conversion (fixes ColumnElement bool error)
            message_value = str(row.message)  # also ensures no SQLAlchemy object leaks

            if role_value == "user":
                messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            ChatCompletionContentPartTextParam(type="text", text=message_value)
                        ]
                    )
                )
            else:
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=[
                            ChatCompletionContentPartTextParam(type="text", text=message_value)
                        ]
                    )
                )


        # NEW USER MESSAGE
        messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content=[ChatCompletionContentPartTextParam(type="text", text=message)]
            )
        )

        # -----------------------------
        # Call OpenAI
        # -----------------------------
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3
        )

        # -----------------------------
        # Extract assistant reply
        # -----------------------------
        msg_obj = response.choices[0].message
        reply_text = self._extract_text(msg_obj.content)

        return reply_text
