from openai import OpenAI
from sqlalchemy.orm import Session

from app.agents.agent_2_responder_chat.repository import ChatMessage, save_emergency_report
from app.agents.agent_2_responder_chat.extractor import ExtractorAgent

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
You are AI-Agent 2, an Emergency Operations Manager.

Your tasks:
- Coordinate rescue teams and track their updates.
- Understand supply needs, hazards, team status, and progress.
- Respond concisely with clear next steps.
- Ask for missing details only when required.
- Always focus on operational clarity.
"""

class ResponderChatAgent:

    def __init__(self):
        self.extractor = ExtractorAgent()

    def _extract_text(self, content):
        if isinstance(content, list):
            return "".join([c.text for c in content if hasattr(c, "text")]).strip()
        if isinstance(content, str):
            return content.strip()
        return ""

    def reply(self, db: Session, responder_id: str, message: str, latitude=None, longitude=None):

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
                        content=[ChatCompletionContentPartTextParam(type="text", text=message_value)]
                    )
                )
            else:
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=[ChatCompletionContentPartTextParam(type="text", text=message_value)]
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

        # Store message with location
        db.add(ChatMessage(
            responder_id=responder_id,
            role="user",
            message=message,
            latitude=latitude,
            longitude=longitude
        ))

        # Extract structured operational data
        extracted = self.extractor.extract(responder_id, message)
        if extracted:
            save_emergency_report(db, extracted)

            # Append operational summary to agent reply
            summary = []
            if extracted.people:
                summary.append(f"People affected: {extracted.people}")
            if extracted.needs:
                summary.append(f"Supply needs: {extracted.needs}")
            if extracted.hazards:
                summary.append(f"Hazards: {', '.join(extracted.hazards)}")
            if extracted.urgency:
                summary.append(f"Urgency: {extracted.urgency}")

            if summary:
                reply_text += "\n\n(Operational Summary)\n" + "\n".join(summary)

        # Store assistant reply
        db.add(ChatMessage(
            responder_id=responder_id,
            role="assistant",
            message=reply_text
        ))

        db.commit()
        return reply_text
