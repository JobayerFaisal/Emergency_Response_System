# backend/app/agents/agent_2_responder_chat/extractor.py

import json
import re
from openai import OpenAI
from app.agents.agent_2_responder_chat.schema import DistressUpdate

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)

client = OpenAI()

def normalize_extracted(parsed: dict):
    result = {}

    # people
    people = parsed.get("people")
    if isinstance(people, int):
        result["people"] = {"count": people}
    elif isinstance(people, dict):
        result["people"] = people
    else:
        result["people"] = None

    # needs
    needs = parsed.get("needs")
    if isinstance(needs, list):
        result["needs"] = {"items": needs}
    elif isinstance(needs, dict):
        result["needs"] = needs
    else:
        result["needs"] = None

    # hazards
    hazards = parsed.get("hazards")
    result["hazards"] = hazards if isinstance(hazards, list) else []

    # urgency
    result["urgency"] = parsed.get("urgency")

    # confidence → numeric
    conf = parsed.get("confidence")
    if isinstance(conf, (int, float)):
        result["confidence"] = float(conf)
    elif isinstance(conf, str):
        mapping = {"high": 0.9, "medium": 0.5, "low": 0.2}
        result["confidence"] = mapping.get(conf.lower(), None)
    else:
        result["confidence"] = None

    return result


EXTRACTOR_SYSTEM_PROMPT = """
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

    def _extract_text(self, content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            combined = ""
            for c in content:
                if hasattr(c, "text") and c.text:
                    combined += c.text
            return combined.strip()
        return str(content).strip()

    def extract(self, responder_id: str, message: str):
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=[ChatCompletionContentPartTextParam(type="text", text=EXTRACTOR_SYSTEM_PROMPT)]
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=[ChatCompletionContentPartTextParam(type="text", text=message)]
            ),
        ]

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0
        )

        raw = self._extract_text(response.choices[0].message.content)
        if not raw:
            return None

        try:
            parsed = json.loads(raw)
        except:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            parsed = json.loads(match.group(0))

        if parsed == {}:
            return None

        normalized = normalize_extracted(parsed)

        return DistressUpdate(
            responder_id=responder_id,
            raw_message=message,
            **normalized
        )
