# path: backend/app/agents/agent_2_responder_chat/extractor.py

import json
import re
from openai import OpenAI
from app.agents.agent_2_responder_chat.schema import DistressUpdate

# Typed message imports required for SDK 2.9.0
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam

client = OpenAI()

SYSTEM_PROMPT = """
Extract ONLY actionable emergency information from a responder message.
Output STRICT JSON with exactly these fields:
- people
- needs
- hazards
- urgency
- confidence

If no information is extractable, return {}.
"""

class ExtractorAgent:

    def _safe_read_text(self, content):
        """
        Extracts text safely from OpenAI typed message content.
        Handles: None, strings, and list[ContentPartText].
        """
        if content is None:
            return ""

        # Text-only response as string
        if isinstance(content, str):
            return content.strip()

        # New SDK format: list of content blocks
        if isinstance(content, list):
            combined = ""
            for part in content:
                if hasattr(part, "text") and part.text:
                    combined += part.text
                elif isinstance(part, dict) and "text" in part:
                    combined += part["text"]
            return combined.strip()

        return str(content).strip()

    def extract(self, responder_id: str, message: str):

        # Build messages using typed parameters
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=[ChatCompletionContentPartTextParam(type="text", text=SYSTEM_PROMPT)]
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=[ChatCompletionContentPartTextParam(type="text", text=message)]
            )
        ]

        # Call OpenAI Chat Completion
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0
        )

        # Extract model output safely
        msg = response.choices[0].message
        raw_output = self._safe_read_text(msg.content)

        if not raw_output:
            return None

        # Try parsing strict JSON
        try:
            parsed = json.loads(raw_output)
        except:
            # fallback scan for { ... }
            m = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if not m:
                return None
            parsed = json.loads(m.group(0))

        # If extraction indicates no actionable info:
        if parsed == {}:
            return None

        # Build Pydantic model
        return DistressUpdate(
            responder_id=responder_id,
            raw_message=message,
            **parsed
        )
