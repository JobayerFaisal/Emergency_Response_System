"""
src/agents/agent_2_distress/alert_processor.py
Converts Agent 1 flood_alert messages into structured DistressIncidents.

When Agent 1 detects a high-risk zone, this processor:
  1. Reads the zone's coordinates from the DB
  2. Estimates num_people from population_density
  3. Determines urgency from risk_score + severity_level
  4. Checks if medical resources are likely needed
  5. Produces a DistressIncident ready for Agent 3
"""

import logging
from typing import List, Optional, Tuple
import asyncpg

from .models import (
    FloodAlert, DistressIncident, UrgencyLevel, IncidentSource,
    SEVERITY_TO_URGENCY, SEVERITY_TO_PRIORITY,
)

logger = logging.getLogger("agent2.processor")


class AlertProcessor:
    """
    Converts raw FloodAlert (from Agent 1) into actionable DistressIncidents.
    Uses PostgreSQL to enrich with zone geodata and population info.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def process_flood_alert(self, alert: FloodAlert) -> List[DistressIncident]:
        """
        Main entry point. Takes one FloodAlert → returns list of incidents.
        Usually one incident per alert, but could be multiple for large zones.
        """
        logger.info(
            "Processing flood_alert: zone=%s risk=%.2f severity=%s",
            alert.zone_name, alert.risk_score, alert.severity_level,
        )

        # Only act on meaningful risk
        if alert.risk_score < 0.3:
            logger.info("Risk score %.2f too low — skipping", alert.risk_score)
            return []

        # Fetch zone details from DB
        zone = await self._get_zone(alert.zone_id)
        if not zone:
            # Fallback: create incident from alert data alone
            logger.warning("Zone %s not in DB — using alert data only", alert.zone_id)
            return [self._incident_from_alert_only(alert)]

        lat, lon = zone["lat"], zone["lon"]
        population_density = zone.get("population_density") or 10000
        drainage = zone.get("drainage_capacity") or "moderate"
        elevation = zone.get("elevation") or 5.0

        # Determine urgency and priority
        urgency = self._determine_urgency(alert, drainage, elevation)
        priority = self._determine_priority(alert, urgency)

        # Estimate affected people
        num_people = self._estimate_affected_people(
            alert.risk_score, population_density
        )

        # Determine medical need
        medical_need = self._needs_medical(alert, urgency)

        incident = DistressIncident(
            zone_id=str(zone["id"]),
            zone_name=alert.zone_name,
            raw_message=(
                f"Automated alert: {alert.severity_level} flood risk detected "
                f"in {alert.zone_name} (risk={alert.risk_score:.0%}, "
                f"confidence={alert.confidence:.0%})"
            ),
            raw_location=alert.zone_name,
            latitude=lat,
            longitude=lon,
            urgency=urgency,
            num_people=num_people,
            medical_need=medical_need,
            priority=priority,
            source=IncidentSource.FLOOD_ALERT,
            confidence=alert.confidence,
        )

        logger.info(
            "Created incident %s: urgency=%s priority=%d people=%d medical=%s",
            incident.incident_id, urgency.value, priority, num_people, medical_need,
        )
        return [incident]

    async def _get_zone(self, zone_id: str) -> Optional[dict]:
        """Fetch zone row from sentinel_zones table."""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT
                        id, name,
                        ST_Y(center::geometry) AS lat,
                        ST_X(center::geometry) AS lon,
                        population_density, elevation, drainage_capacity
                    FROM sentinel_zones
                    WHERE id::text = $1
                       OR name ILIKE $2
                    LIMIT 1
                """, zone_id, f"%{zone_id.replace('-', ' ')}%")
                return dict(row) if row else None
        except Exception as e:
            logger.error("DB error fetching zone %s: %s", zone_id, e)
            return None

    def _determine_urgency(
        self, alert: FloodAlert, drainage: str, elevation: float
    ) -> UrgencyLevel:
        """
        Map risk_score + severity + zone characteristics → UrgencyLevel.
        Poor drainage and low elevation escalate urgency.
        """
        base = SEVERITY_TO_URGENCY.get(alert.severity_level.lower(), UrgencyLevel.MODERATE)

        # Escalate if poor drainage + high risk
        if alert.risk_score >= 0.75 and drainage == "poor":
            if base == UrgencyLevel.URGENT:
                return UrgencyLevel.LIFE_THREATENING
            if base == UrgencyLevel.MODERATE:
                return UrgencyLevel.URGENT

        # Escalate if low elevation (flood-prone)
        if alert.risk_score >= 0.8 and elevation < 5:
            if base in (UrgencyLevel.MODERATE, UrgencyLevel.LOW):
                return UrgencyLevel.URGENT

        return base

    def _determine_priority(self, alert: FloodAlert, urgency: UrgencyLevel) -> int:
        """Map urgency → priority integer (1–5)."""
        mapping = {
            UrgencyLevel.LIFE_THREATENING: 5,
            UrgencyLevel.URGENT_MEDICAL:   5,
            UrgencyLevel.URGENT:           4,
            UrgencyLevel.MODERATE:         3,
            UrgencyLevel.LOW:              2,
        }
        base_priority = mapping.get(urgency, 3)

        # Boost by 1 if very high confidence
        if alert.confidence >= 0.85 and base_priority < 5:
            base_priority += 1

        return min(base_priority, 5)

    def _estimate_affected_people(
        self, risk_score: float, population_density: int
    ) -> int:
        """
        Rough estimate of people at risk.
        Uses risk_score as a fraction of population in a ~1km² hotspot.
        """
        # Assume 1 km² impact area at minimum, scaled by risk
        impact_area_km2 = max(0.5, risk_score * 2.0)
        affected = int(population_density * impact_area_km2 * risk_score * 0.01)
        return max(affected, 1)

    def _needs_medical(self, alert: FloodAlert, urgency: UrgencyLevel) -> bool:
        """Determine if medical resources are likely needed."""
        if urgency in (UrgencyLevel.LIFE_THREATENING, UrgencyLevel.URGENT_MEDICAL):
            return True
        # High risk + poor conditions likely means injuries
        if alert.risk_score >= 0.7:
            return True
        return False

    def _incident_from_alert_only(self, alert: FloodAlert) -> DistressIncident:
        """Fallback when zone not found in DB — uses alert data only."""
        urgency = SEVERITY_TO_URGENCY.get(
            alert.severity_level.lower(), UrgencyLevel.MODERATE
        )
        priority = SEVERITY_TO_PRIORITY.get(alert.severity_level.lower(), 3)

        return DistressIncident(
            zone_id=alert.zone_id,
            zone_name=alert.zone_name,
            raw_message=f"Automated alert: {alert.severity_level} flood risk in {alert.zone_name}",
            raw_location=alert.zone_name,
            latitude=23.8103,   # Default to Dhaka Central
            longitude=90.4125,
            urgency=urgency,
            num_people=10,
            medical_need=alert.risk_score >= 0.7,
            priority=priority,
            source=IncidentSource.FLOOD_ALERT,
            confidence=alert.confidence,
        )



