"""
src/agents/agent_3_resource/ai_allocator.py
AI-powered allocation decision engine using OpenAI GPT-4o.

Replaces the fixed if/else rules with GPT-4o reasoning.
Falls back to rule-based logic if OpenAI API is unavailable.

How it works:
  1. Gathers full inventory + incident details
  2. Sends a structured prompt to GPT-4o
  3. GPT-4o returns a JSON decision with reasoning
  4. We execute that decision (mark units deployed, save to DB)
"""

import json
import logging
import os
from typing import Optional

import httpx

from shared.geo_utils import haversine_km
from .models import ResourceType, ALLOCATION_RULES

logger = logging.getLogger("agent3.ai_allocator")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL   = "gpt-4o"


class AIDecisionEngine:
    """
    Uses OpenAI GPT-4o to make intelligent resource allocation decisions.
    Falls back to rule-based logic if API is unavailable or key not set.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.enabled = bool(self.api_key)
        if self.enabled:
            logger.info("AI decision engine ENABLED — using OpenAI GPT-4o")
        else:
            logger.warning(
                "OPENAI_API_KEY not set — AI engine disabled, using rule-based fallback"
            )

    async def decide_allocation(
        self,
        incident: dict,
        available_resources: dict,
        dest_lat: float,
        dest_lon: float,
    ) -> dict:
        """
        Ask GPT-4o to decide which resources to allocate for this incident.

        Returns:
        {
            "ai_used": True/False,
            "reasoning": "GPT-4o's explanation in plain English",
            "decision": {
                "rescue_boat": 2,
                "medical_team": 1,
                "medical_kit": 2,
                "food_supply": 0,
                "water_supply": 0
            },
            "ai_notes": "Any special instructions from AI"
        }
        """
        if not self.enabled:
            return self._rule_based_fallback(incident)

        try:
            prompt = self._build_prompt(
                incident, available_resources, dest_lat, dest_lon
            )
            response = await self._call_openai(prompt)
            result   = self._parse_response(response)
            result["ai_used"] = True
            logger.info(
                "AI decision for %s: %s",
                incident["incident_id"],
                result["reasoning"][:120]
            )
            return result

        except Exception as e:
            logger.error(
                "OpenAI API error (%s) — falling back to rule-based logic", e
            )
            fallback = self._rule_based_fallback(incident)
            fallback["reasoning"] = (
                f"[AI FALLBACK] OpenAI unavailable ({e}). "
                + fallback["reasoning"]
            )
            return fallback

    # ─────────────────────────────────────────────────────────────────────
    # PROMPT BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        incident: dict,
        available_resources: dict,
        dest_lat: float,
        dest_lon: float,
    ) -> str:
        """Build a detailed situation prompt for GPT-4o."""

        # Format each resource type with distance info
        resource_lines = []
        for rtype, units in available_resources.items():
            if not units:
                resource_lines.append(f"  {rtype}: NONE AVAILABLE")
                continue
            units_with_dist = sorted(
                units,
                key=lambda u: haversine_km(u["lat"], u["lon"], dest_lat, dest_lon)
            )
            closest = units_with_dist[0]
            dist    = haversine_km(closest["lat"], closest["lon"], dest_lat, dest_lon)
            resource_lines.append(
                f"  {rtype}: {len(units)} available "
                f"(closest: {closest['name']} at {dist:.1f} km)"
            )

        resource_summary = "\n".join(resource_lines)

        prompt = f"""You are an AI coordinator for Bangladesh Flood Disaster Response System.
A flood emergency has been detected. Decide which resources to allocate.

=== EMERGENCY SITUATION ===
Incident ID   : {incident.get('incident_id')}
Location      : {incident.get('zone_name')} ({dest_lat:.4f}°N, {dest_lon:.4f}°E)
Original message: "{incident.get('raw_message', 'N/A')}"
Urgency level : {incident.get('urgency')}
People trapped: {incident.get('num_people', 'unknown')}
Medical need  : {incident.get('medical_need', False)}
Priority      : {incident.get('priority')}/5

=== AVAILABLE RESOURCES ===
{resource_summary}

=== YOUR TASK ===
Decide exactly how many of each resource type to send.
Consider:
- Urgency and number of people
- Distance of resources (closer = faster help)
- Don't over-allocate (leave some for other emergencies)
- Medical need should influence medical_team and medical_kit allocation
- Boats are needed for rescue in floodwater
- Food/water for sustained relief, not immediate rescue

Respond ONLY with this exact JSON format (no extra text):
{{
  "reasoning": "2-3 sentences explaining your decision in plain English",
  "decision": {{
    "rescue_boat": <number>,
    "medical_team": <number>,
    "medical_kit": <number>,
    "food_supply": <number>,
    "water_supply": <number>
  }},
  "ai_notes": "any special instruction or concern"
}}"""
        return prompt

    # ─────────────────────────────────────────────────────────────────────
    # OPENAI API CALL
    # ─────────────────────────────────────────────────────────────────────

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT-4o and return the raw response text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        body = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a disaster response AI coordinator for Bangladesh. "
                        "You make precise, life-saving resource allocation decisions. "
                        "Always respond with valid JSON only — no markdown, no extra text."
                    )
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature":  0.2,   # Low temp = more consistent, less random
            "max_tokens":   500,
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(OPENAI_API_URL, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"]

    # ─────────────────────────────────────────────────────────────────────
    # RESPONSE PARSER
    # ─────────────────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> dict:
        """Parse GPT-4o JSON response into our decision format."""
        parsed = json.loads(raw)

        decision = parsed.get("decision", {})

        # Validate all resource types present, default to 0
        clean_decision = {
            "rescue_boat":  max(0, int(decision.get("rescue_boat",  0))),
            "medical_team": max(0, int(decision.get("medical_team", 0))),
            "medical_kit":  max(0, int(decision.get("medical_kit",  0))),
            "food_supply":  max(0, int(decision.get("food_supply",  0))),
            "water_supply": max(0, int(decision.get("water_supply", 0))),
        }

        return {
            "ai_used":   True,
            "reasoning": parsed.get("reasoning", "No reasoning provided"),
            "decision":  clean_decision,
            "ai_notes":  parsed.get("ai_notes", ""),
        }

    # ─────────────────────────────────────────────────────────────────────
    # RULE-BASED FALLBACK
    # ─────────────────────────────────────────────────────────────────────

    def _rule_based_fallback(self, incident: dict) -> dict:
        """Original rule-based logic — used when AI is unavailable."""
        urgency     = incident.get("urgency", "MODERATE").upper()
        medical_need = incident.get("medical_need", False)

        rule_key = urgency
        if urgency == "URGENT" and medical_need:
            rule_key = "URGENT_MEDICAL"

        rules = ALLOCATION_RULES.get(rule_key, ALLOCATION_RULES["MODERATE"])

        decision = {rtype.value: qty for rtype, qty in rules.items()}

        reasoning = (
            f"Rule-based decision: urgency={urgency}, medical_need={medical_need}. "
            f"Applied standard allocation rules for {rule_key}."
        )

        return {
            "ai_used":   False,
            "reasoning": reasoning,
            "decision":  decision,
            "ai_notes":  "Fallback — OpenAI not available",
        }
