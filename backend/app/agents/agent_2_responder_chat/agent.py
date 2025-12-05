# backend/app/agents/agent_2_responder_chat/agent.py

from openai import OpenAI
from sqlalchemy.orm import Session
from app.agents.agent_2_responder_chat.repository import ChatMessage

from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)

client = OpenAI()

RESPONDER_SYSTEM_PROMPT = """
You are an emergency response assistant.
Be concise, ask clarifying questions only when helpful,
and stay focused on actionable emergency information.
"""

class ResponderChatAgent:

    def _extract_text(self, content):
        if isinstance(content, list):
            return "".join([c.text for c in content if hasattr(c, "text")]).strip()
        if isinstance(content, str):
            return content.strip()
        return ""

    def reply(self, db: Session, responder_id: str, message: str):

        history = (
            db.query(ChatMessage)
            .filter(ChatMessage.responder_id == responder_id)
            .order_by(ChatMessage.timestamp.desc())
            .limit(10)
            .all()
        )
        history = list(reversed(history))

        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=[ChatCompletionContentPartTextParam(type="text", text=RESPONDER_SYSTEM_PROMPT)]
            )
        ]

        for row in history:
            role_value = str(row.role)
            message_value = str(row.message)

            if role_value == "user":
                messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=[
                            ChatCompletionContentPartTextParam(
                                type="text",
                                text=message_value
                            )
                        ]
                    )
                )
            else:
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=[
                            ChatCompletionContentPartTextParam(
                                type="text",
                                text=message_value
                            )
                        ]
                    )
                )


        messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content=[ChatCompletionContentPartTextParam(type="text", text=message)]
            )
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.3
        )

        reply_text = self._extract_text(response.choices[0].message.content)

        db.add(ChatMessage(responder_id=responder_id, role="user", message=message))
        db.add(ChatMessage(responder_id=responder_id, role="assistant", message=reply_text))
        db.commit()

        return reply_text
