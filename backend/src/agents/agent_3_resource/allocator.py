# backend/src/agents/agent_3_resource/allocator.py

"""
src/agents/agent_3_resource/allocator.py
AI-powered resource allocation for Agent 3.

Flow:
  1. Resolve exact location
  2. Get full inventory of available resources
  3. Ask GPT-4o or rule-based engine what to send
  4. Execute that decision (mark units deployed)
  5. Save allocation + log transactions
"""

import logging
from typing import List, Optional

from shared.geo_utils import haversine_km
from shared.location_resolver import resolver as location_resolver

from .models import ResourceType, ResourceAllocation
from .ai_allocator import AIDecisionEngine
from shared.severity import GeoPoint


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
        self.ai = get_ai_engine()

    async def allocate_for_incident(self, incident: dict) -> Optional[ResourceAllocation]:
        incident_id = incident.get("incident_id") or incident.get("id") or "unknown-incident"
        zone_id = (
            incident.get("zone_id")
            or incident.get("district")
            or incident.get("zone_name")
            or "unknown-zone"
        )
        zone_name = incident.get("zone_name") or incident.get("district") or "Unknown Zone"
        raw_message = incident.get("raw_message") or incident.get("text") or incident.get("message") or ""
        raw_location = incident.get("raw_location") or incident.get("district") or zone_name

        resolved = location_resolver.resolve(
            raw_message=raw_message,
            coarse_location=raw_location,
        )

        dest_lat = resolved.latitude
        dest_lon = resolved.longitude

        provided_lat = incident.get("latitude")
        provided_lon = incident.get("longitude")
        if provided_lat is not None and provided_lon is not None and resolved.confidence < 0.8:
            dest_lat = provided_lat
            dest_lon = provided_lon

        location_note = (
            f"Location resolved via {resolved.resolution_method} "
            f"(confidence={resolved.confidence:.0%}, ±{resolved.uncertainty_radius_m}m) "
            f"→ {resolved.display_name}"
        )
        if resolved.needs_followup:
            location_note += " | Low confidence - Agent 2 should request GPS"

        logger.info("Incident %s: %s", incident_id, location_note)

        # Step 2: Gather full available inventory
        available_resources = {}
        for rtype in ResourceType:
            units = await self.inv.get_available_units(rtype)
            available_resources[rtype.value] = units

        inventory_snapshot = {
            key: len(value) for key, value in available_resources.items()
        }
        logger.info(
            "Available resources for incident %s: %s",
            incident_id,
            inventory_snapshot,
        )

        # Step 3: AI / fallback decision
        normalized_incident = {
            "incident_id": incident_id,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "raw_message": raw_message,
            "raw_location": raw_location,
            "latitude": dest_lat,
            "longitude": dest_lon,
            "urgency": incident.get("urgency", "MODERATE"),
            "num_people": int(incident.get("num_people", 0) or 0),
            "medical_need": bool(incident.get("medical_need", False)),
            "priority": int(incident.get("priority", 3) or 3),
            "confidence": float(incident.get("confidence", 0.8) or 0.8),
        }

        logger.info("Asking allocator engine for incident %s", incident_id)
        ai_result = await self.ai.decide_allocation(
            incident=normalized_incident,
            available_resources=available_resources,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
        )

        decision = ai_result["decision"]
        reasoning = ai_result["reasoning"]
        ai_used = ai_result["ai_used"]
        ai_notes = ai_result.get("ai_notes", "")

        logger.info(
            "Decision (%s) for %s: %s",
            "GPT-4o" if ai_used else "rule-based",
            incident_id,
            reasoning,
        )

        # Step 4: Execute decision
        allocated_units = []
        deficit_log = []

        for resource_type_str, needed_qty in decision.items():
            if needed_qty <= 0:
                continue

            try:
                rtype = ResourceType(resource_type_str)
            except ValueError:
                logger.warning("Unknown resource type from allocator engine: %s", resource_type_str)
                continue

            available = available_resources.get(resource_type_str, [])

            if not available:
                msg = f"No {resource_type_str} available (allocator requested {needed_qty})"
                deficit_log.append(msg)
                logger.warning("Resource deficit: %s", msg)
                continue

            available_sorted = sorted(
                available,
                key=lambda u: haversine_km(u["lat"], u["lon"], dest_lat, dest_lon),
            )

            selected = available_sorted[:needed_qty]

            if len(selected) < needed_qty:
                msg = f"Only {len(selected)}/{needed_qty} {resource_type_str} available"
                deficit_log.append(msg)
                logger.warning("Resource deficit: %s", msg)

            for unit in selected:
                await self.inv.mark_deployed(
                    unit_id=unit["id"],
                    incident_id=incident_id,
                    zone_id=zone_id,
                )

                distance_km = haversine_km(
                    unit["lat"], unit["lon"], dest_lat, dest_lon
                )

                allocated_units.append({
                    "unit_id": str(unit["id"]),
                    "resource_type": unit["resource_type"],
                    "name": unit["name"],
                    "current_location": {
                        "latitude": unit["lat"],
                        "longitude": unit["lon"],
                    },
                    "distance_to_incident_km": round(distance_km, 2),
                })

        # Emergency fallback: if AI/rules asked for unavailable types, try any available unit
        if not allocated_units:
            logger.error(
                "ZERO resources allocated for %s — attempting emergency fallback",
                incident_id,
            )

            fallback_units = []
            for units in available_resources.values():
                fallback_units.extend(units)

            if fallback_units:
                fallback_units = sorted(
                    fallback_units,
                    key=lambda u: haversine_km(u["lat"], u["lon"], dest_lat, dest_lon),
                )
                selected = fallback_units[:2]

                for unit in selected:
                    await self.inv.mark_deployed(
                        unit_id=unit["id"],
                        incident_id=incident_id,
                        zone_id=zone_id,
                    )

                    distance_km = haversine_km(
                        unit["lat"], unit["lon"], dest_lat, dest_lon
                    )

                    allocated_units.append({
                        "unit_id": str(unit["id"]),
                        "resource_type": unit["resource_type"],
                        "name": unit["name"],
                        "current_location": {
                            "latitude": unit["lat"],
                            "longitude": unit["lon"],
                        },
                        "distance_to_incident_km": round(distance_km, 2),
                    })

                deficit_log.append(
                    "Emergency fallback used: allocated nearest available units regardless of requested type"
                )

        if not allocated_units:
            logger.error("ZERO resources allocated for %s", incident_id)
            return None

        # Step 5: Log transactions
        await self.inv.log_allocation_transaction(
            units=allocated_units,
            incident_id=incident_id,
            zone_id=zone_id,
        )

        # Step 6: Build notes
        notes_parts = [location_note]
        notes_parts.append(f"[GPT-4o] {reasoning}" if ai_used else f"[Rule-based] {reasoning}")

        if ai_notes:
            notes_parts.append(f"AI notes: {ai_notes}")

        if deficit_log:
            notes_parts.append("PARTIAL: " + "; ".join(deficit_log))

        allocation = ResourceAllocation(
            incident_id=incident_id,
            zone_id=zone_id,
            zone_name=zone_name or resolved.display_name,
            destination=GeoPoint(
                latitude=float(dest_lat),
                longitude=float(dest_lon),
            ),
            priority=int(incident.get("priority", 3) or 3),
            urgency=incident.get("urgency", "MODERATE"),
            num_people_affected=int(incident.get("num_people", 0) or 0),
            allocated_resources=allocated_units,
            requires_medical=bool(incident.get("medical_need", False)),
            partial_allocation=bool(deficit_log),
            notes=" | ".join(notes_parts),
        )



        await self.inv.save_allocation(allocation)

        logger.info(
            "Allocation %s created for %s: %d unit(s) → %s",
            allocation.allocation_id,
            incident_id,
            len(allocated_units),
            allocation.zone_name,
        )

        return allocation

    async def process_distress_queue(self, incidents: List[dict]) -> List[ResourceAllocation]:
        sorted_incidents = sorted(
            incidents,
            key=lambda x: x.get("priority", 3),
            reverse=True,
        )

        results = []
        for incident in sorted_incidents:
            try:
                allocation = await self.allocate_for_incident(incident)
                if allocation:
                    results.append(allocation)
            except Exception as exc:
                logger.exception(
                    "Failed to allocate incident %s: %s",
                    incident.get("incident_id") or incident.get("id") or "unknown",
                    exc,
                )
        return results