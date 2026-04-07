# backend/src/agents/agent_3_resource/allocator.py

"""
src/agents/agent_3_resource/allocator.py
AI-powered resource allocation for Agent 3.

Flow:
  1. Resolve exact location (same as before)
  2. Get full inventory of available resources
  3. Ask GPT-4o: "given this situation, what should we send?"
  4. Execute GPT-4o decision (mark units deployed)
  5. Fall back to rule-based if AI unavailable
"""

import logging
from typing import List, Optional

from shared.geo_utils import haversine_km
from shared.location_resolver import resolver as location_resolver
from .models import ResourceType, ResourceAllocation
from .ai_allocator import AIDecisionEngine

logger = logging.getLogger("agent3.allocator")

_ai_engine: Optional[AIDecisionEngine] = None


def get_ai_engine() -> AIDecisionEngine:
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIDecisionEngine()
    return _ai_engine


class ResourceAllocator:
    def __init__(self, inventory_manager):
        self.inv = inventory_manager
        self.ai  = get_ai_engine()

    async def allocate_for_incident(self, incident: dict) -> Optional[ResourceAllocation]:

        # Step 1: Resolve exact location
        resolved = location_resolver.resolve(
            raw_message=incident.get("raw_message", ""),
            coarse_location=incident.get("raw_location", incident.get("zone_name", "")),
        )
        dest_lat = resolved.latitude
        dest_lon = resolved.longitude
        a2_lat = incident.get("latitude")
        a2_lon = incident.get("longitude")
        if a2_lat and a2_lon and resolved.confidence < 0.8:
            dest_lat = a2_lat
            dest_lon = a2_lon

        location_note = (
            f"Location resolved via {resolved.resolution_method} "
            f"(confidence={resolved.confidence:.0%}, "
            f"±{resolved.uncertainty_radius_m}m) → {resolved.display_name}"
        )
        if resolved.needs_followup:
            location_note += " | ⚠ Low confidence — Agent 2 should request GPS"
        logger.info("Incident %s: %s", incident["incident_id"], location_note)

        # Step 2: Get full inventory
        available_resources = {}
        for rtype in ResourceType:
            units = await self.inv.get_available_units(rtype)
            available_resources[rtype.value] = units

        # Step 3: Ask GPT-4o
        logger.info("Asking GPT-4o for allocation decision — incident %s", incident["incident_id"])
        ai_result = await self.ai.decide_allocation(
            incident=incident,
            available_resources=available_resources,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
        )
        decision  = ai_result["decision"]
        reasoning = ai_result["reasoning"]
        ai_used   = ai_result["ai_used"]
        ai_notes  = ai_result.get("ai_notes", "")
        logger.info("Decision (%s): %s", "GPT-4o" if ai_used else "rule-based", reasoning)

        # Step 4: Execute decision
        allocated_units = []
        deficit_log     = []

        for resource_type_str, needed_qty in decision.items():
            if needed_qty <= 0:
                continue
            try:
                rtype = ResourceType(resource_type_str)
            except ValueError:
                continue
            available = available_resources.get(resource_type_str, [])
            if not available:
                deficit_log.append(f"No {resource_type_str} available (AI wanted {needed_qty})")
                logger.warning("Resource deficit: %s", deficit_log[-1])
                continue
            available_sorted = sorted(
                available,
                key=lambda u: haversine_km(u["lat"], u["lon"], dest_lat, dest_lon),
            )
            selected = available_sorted[:needed_qty]
            if len(selected) < needed_qty:
                deficit_log.append(f"Only {len(selected)}/{needed_qty} {resource_type_str} available")
            for unit in selected:
                await self.inv.mark_deployed(
                    unit_id=unit["id"],
                    incident_id=incident["incident_id"],
                    zone_id=incident["zone_id"],
                )
                distance_km = haversine_km(unit["lat"], unit["lon"], dest_lat, dest_lon)
                allocated_units.append({
                    "unit_id":       str(unit["id"]),
                    "resource_type": unit["resource_type"],
                    "name":          unit["name"],
                    "current_location": {"latitude": unit["lat"], "longitude": unit["lon"]},
                    "distance_to_incident_km": round(distance_km, 2),
                })

        if not allocated_units:
            logger.error("ZERO resources allocated for %s", incident["incident_id"])
            return None

        # Step 5: Log transactions
        await self.inv.log_allocation_transaction(
            units=allocated_units,
            incident_id=incident["incident_id"],
            zone_id=incident["zone_id"],
        )

        # Step 6: Build allocation with AI reasoning in notes
        notes_parts = [location_note]
        if ai_used:
            notes_parts.append(f"[GPT-4o] {reasoning}")
        else:
            notes_parts.append(f"[Rule-based] {reasoning}")
        if ai_notes:
            notes_parts.append(f"AI notes: {ai_notes}")
        if deficit_log:
            notes_parts.append("PARTIAL: " + "; ".join(deficit_log))

        allocation = ResourceAllocation(
            incident_id=incident["incident_id"],
            zone_id=incident["zone_id"],
            zone_name=incident.get("zone_name", resolved.display_name),
            destination={"latitude": dest_lat, "longitude": dest_lon},
            priority=incident.get("priority", 3),
            urgency=incident.get("urgency", "MODERATE"),
            num_people_affected=incident.get("num_people", 0),
            allocated_resources=allocated_units,
            requires_medical=incident.get("medical_need", False),
            partial_allocation=bool(deficit_log),
            notes=" | ".join(notes_parts),
        )
        await self.inv.save_allocation(allocation)
        logger.info(
            "Allocation %s: %d units → %s | By: %s",
            allocation.allocation_id, len(allocated_units),
            allocation.zone_name, "GPT-4o" if ai_used else "rule-based",
        )
        return allocation

    async def process_distress_queue(self, incidents: List[dict]) -> List[ResourceAllocation]:
        sorted_incidents = sorted(incidents, key=lambda x: x.get("priority", 3), reverse=True)
        results = []
        for incident in sorted_incidents:
            allocation = await self.allocate_for_incident(incident)
            if allocation:
                results.append(allocation)
        return results
