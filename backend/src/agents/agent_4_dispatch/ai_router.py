"""
src/agents/agent_4_dispatch/ai_router.py
AI-powered routing advisor for Agent 4 using OpenAI GPT-4o.

GPT-4o analyzes each team's situation and recommends:
- Transport mode (road vs waterway)
- Speed adjustment based on conditions
- Safety warnings
- Special instructions for the team

Falls back to rule-based routing if API unavailable.
"""

import json
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("agent4.ai_router")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL   = "gpt-4o"


class AIRoutingAdvisor:
    """
    Uses GPT-4o to advise on routing strategy for each dispatched team.
    The math (distance, ETA) is still computed by RouteComputer.
    GPT-4o adds intelligence: should they take a detour? Is the road safe?
    Any special instructions for this team given the flood conditions?
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.enabled = bool(self.api_key)
        if self.enabled:
            logger.info("AI routing advisor ENABLED — using OpenAI GPT-4o")
        else:
            logger.warning("OPENAI_API_KEY not set — AI routing disabled, using standard routing")

    async def advise_route(
        self,
        unit_name: str,
        resource_type: str,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        dest_zone: str,
        urgency: str,
        num_people: int,
        distance_km: float,
        eta_minutes: float,
        route_safety_score: float,
        flood_condition: str,
        transport_mode: str,
    ) -> dict:
        """
        Ask GPT-4o for routing advice for a specific team.

        Returns:
        {
            "ai_used": True/False,
            "transport_mode": "road" or "waterway",
            "speed_adjustment": 1.0 (multiplier — 0.8 means 20% slower),
            "safety_warning": "any safety concern",
            "team_instruction": "plain English instruction for the team",
            "reasoning": "GPT-4o explanation"
        }
        """
        if not self.enabled:
            return self._standard_advice(transport_mode, resource_type)

        try:
            prompt = self._build_prompt(
                unit_name, resource_type, origin_lat, origin_lon,
                dest_lat, dest_lon, dest_zone, urgency, num_people,
                distance_km, eta_minutes, route_safety_score,
                flood_condition, transport_mode
            )
            response = await self._call_openai(prompt)
            result   = self._parse_response(response)
            result["ai_used"] = True
            logger.info("AI route advice for %s: %s", unit_name, result["team_instruction"])
            return result

        except Exception as e:
            logger.error("OpenAI routing API error (%s) — using standard routing", e)
            fallback = self._standard_advice(transport_mode, resource_type)
            fallback["reasoning"] = f"[AI FALLBACK] {e}. Using standard routing."
            return fallback

    async def advise_full_dispatch(
        self,
        zone_name: str,
        urgency: str,
        num_people: int,
        teams: list,
        flood_condition: str,
    ) -> dict:
        """
        Ask GPT-4o for an overall dispatch strategy for all teams going to one incident.
        This gives a high-level coordination view — who should go first, any coordination needed?

        Returns a summary advice dict.
        """
        if not self.enabled:
            return {
                "ai_used": False,
                "overall_strategy": "Standard dispatch — all teams proceed independently.",
                "coordination_note": "No AI coordination available.",
                "priority_team": teams[0]["unit_name"] if teams else "N/A",
            }

        try:
            team_lines = "\n".join([
                f"  - {t['unit_name']} ({t['resource_type']}): "
                f"{t['distance_km']:.1f} km away, ETA {t['eta_minutes']:.0f} min, "
                f"mode={t['transport_mode']}"
                for t in teams
            ])

            prompt = f"""You are a disaster response coordinator for Bangladesh floods.
Multiple rescue teams are being dispatched to ONE incident. Give coordination advice.

=== INCIDENT ===
Location : {zone_name}
Urgency  : {urgency}
People   : {num_people} trapped
Flood    : {flood_condition}

=== TEAMS BEING DISPATCHED ===
{team_lines}

=== YOUR TASK ===
Give a short coordination strategy. Who should arrive first? 
Should any team wait for another? Any special coordination needed?

Respond ONLY with this JSON:
{{
  "overall_strategy": "1-2 sentence dispatch strategy",
  "coordination_note": "any specific coordination instruction",
  "priority_team": "name of team that should arrive first",
  "reasoning": "brief explanation"
}}"""

            response = await self._call_openai(prompt)
            result   = json.loads(response)
            result["ai_used"] = True
            logger.info("AI dispatch strategy for %s: %s", zone_name, result["overall_strategy"])
            return result

        except Exception as e:
            logger.error("AI dispatch strategy error: %s", e)
            return {
                "ai_used": False,
                "overall_strategy": "Standard dispatch — all teams proceed independently.",
                "coordination_note": f"AI unavailable: {e}",
                "priority_team": teams[0]["unit_name"] if teams else "N/A",
                "reasoning": "Fallback to standard routing",
            }

    # ─────────────────────────────────────────────────────────────────────
    # PROMPT BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        unit_name, resource_type, origin_lat, origin_lon,
        dest_lat, dest_lon, dest_zone, urgency, num_people,
        distance_km, eta_minutes, route_safety_score,
        flood_condition, transport_mode
    ) -> str:
        safety_text = (
            "SAFE (no flooded roads)" if route_safety_score >= 0.9
            else f"WARNING — route safety {route_safety_score:.0%} (some flooded roads)"
        )

        return f"""You are a routing advisor for Bangladesh Flood Disaster Response.
Advise on the best routing strategy for this specific rescue team.

=== TEAM ===
Unit name     : {unit_name}
Resource type : {resource_type}
Origin        : {origin_lat:.4f}°N, {origin_lon:.4f}°E
Destination   : {dest_zone} ({dest_lat:.4f}°N, {dest_lon:.4f}°E)

=== SITUATION ===
Urgency       : {urgency}
People trapped: {num_people}
Distance      : {distance_km:.1f} km
Estimated ETA : {eta_minutes:.0f} minutes
Route safety  : {safety_text}
Flood condition: {flood_condition}
Current mode  : {transport_mode}

=== NOTES ===
- rescue_boat travels through floodwater (waterway mode) — faster in floods
- medical_team and supplies travel by road — may be blocked by flooding
- food_supply and water_supply go by road truck

=== YOUR TASK ===
Give routing advice for this specific team.
Consider: flood depth, road safety, urgency, resource type.

Respond ONLY with this JSON:
{{
  "transport_mode": "{transport_mode}",
  "speed_adjustment": 1.0,
  "safety_warning": "any specific safety warning or empty string",
  "team_instruction": "clear plain English instruction for the team (1 sentence)",
  "reasoning": "1-2 sentences explaining your routing recommendation"
}}"""

    # ─────────────────────────────────────────────────────────────────────
    # OPENAI API CALL
    # ─────────────────────────────────────────────────────────────────────

    async def _call_openai(self, prompt: str) -> str:
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
                        "You are a disaster response routing coordinator for Bangladesh. "
                        "Give precise, actionable routing instructions. "
                        "Always respond with valid JSON only."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "temperature":  0.2,
            "max_tokens":   400,
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
        parsed = json.loads(raw)
        return {
            "ai_used":          True,
            "transport_mode":   parsed.get("transport_mode", "road"),
            "speed_adjustment": float(parsed.get("speed_adjustment", 1.0)),
            "safety_warning":   parsed.get("safety_warning", ""),
            "team_instruction": parsed.get("team_instruction", "Proceed to destination."),
            "reasoning":        parsed.get("reasoning", ""),
        }

    # ─────────────────────────────────────────────────────────────────────
    # STANDARD FALLBACK
    # ─────────────────────────────────────────────────────────────────────

    def _standard_advice(self, transport_mode: str, resource_type: str) -> dict:
        if resource_type == "rescue_boat":
            instruction = "Navigate through floodwater directly to destination."
        elif resource_type == "medical_team":
            instruction = "Take fastest road route to destination. Avoid visibly flooded roads."
        else:
            instruction = "Deliver supplies via road to destination."

        return {
            "ai_used":          False,
            "transport_mode":   transport_mode,
            "speed_adjustment": 1.0,
            "safety_warning":   "",
            "team_instruction": instruction,
            "reasoning":        "Standard routing — AI not available.",
        }
