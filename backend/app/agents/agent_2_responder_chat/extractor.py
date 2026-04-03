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

    # ------------------------------
    # People
    # ------------------------------
    people = parsed.get("people")
    if isinstance(people, int):
        result["people"] = {"count": people}
    elif isinstance(people, dict):
        result["people"] = people
    else:
        result["people"] = None

    # ------------------------------
    # Needs
    # ------------------------------
    needs = parsed.get("needs")
    if isinstance(needs, list):
        result["needs"] = {"items": needs}
    elif isinstance(needs, dict):
        result["needs"] = needs
    else:
        result["needs"] = None

    # ------------------------------
    # Hazards
    # ------------------------------
    hazards = parsed.get("hazards")
    result["hazards"] = hazards if isinstance(hazards, list) else []

    # ------------------------------
    # Urgency
    # ------------------------------
    result["urgency"] = parsed.get("urgency")

    # ------------------------------
    # Confidence
    # ------------------------------
    conf = parsed.get("confidence")
    if isinstance(conf, (int, float)):
        result["confidence"] = float(conf)
    elif isinstance(conf, str):
        mapping = {"high": 0.9, "medium": 0.5, "low": 0.2}
        result["confidence"] = mapping.get(conf.lower(), None)
    else:
        result["confidence"] = None

    # ------------------------------
    # NEW FIELDS FOR AGENT-2
    # ------------------------------

    # team_status
    result["team_status"] = parsed.get("team_status")

    # supply_request
    sr = parsed.get("supply_request")
    result["supply_request"] = sr if isinstance(sr, list) else None

    # mobility_issues
    mi = parsed.get("mobility_issues")
    result["mobility_issues"] = mi if isinstance(mi, list) else None

    # rescue_progress
    result["rescue_progress"] = parsed.get("rescue_progress")

    # medical_needs
    mn = parsed.get("medical_needs")
    result["medical_needs"] = mn if isinstance(mn, list) else None

    return result


# -------------------------------------------------------
# UPDATED SYSTEM PROMPT FOR TEAM OPERATIONS EXTRACTION
# -------------------------------------------------------
EXTRACTOR_SYSTEM_PROMPT = """
Extract ONLY actionable emergency and operational information.
Output STRICT JSON with exactly these fields:

Required emergency fields:
- people
- needs
- hazards
- urgency
- confidence

Additional fields for rescue-team operations:
- team_status        (string or null)
- supply_request     (list or null)
- mobility_issues    (list or null)
- rescue_progress    (string or null)
- medical_needs      (list or null)

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

        # Try parsing strict JSON
        try:
            parsed = json.loads(raw)
        except:
            # Attempt to salvage JSON from response
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
