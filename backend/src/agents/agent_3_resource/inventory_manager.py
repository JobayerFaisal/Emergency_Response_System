# backend/src/agents/agent_3_resource/inventory_manager.py

"""
src/agents/agent_3_resource/inventory_manager.py
Database CRUD for resource_units, inventory_transactions, resource_allocations.
Uses asyncpg directly (no ORM) for fast async queries.
"""

import json
import logging
from typing import List, Optional, Dict
from uuid import UUID
from datetime import datetime, timezone

import asyncpg

from shared.geo_utils import haversine_km, postgis_point_wkt
from .models import ResourceUnit, ResourceType, ResourceStatus, ResourceAllocation

logger = logging.getLogger("agent3.inventory")


class InventoryManager:
    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool

    # ── READ ──────────────────────────────────────────────────────────────

    async def get_all_units(self) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, resource_type, name, status, capacity,
                       ST_Y(current_location::geometry) AS lat,
                       ST_X(current_location::geometry) AS lon,
                       ST_Y(base_location::geometry)    AS base_lat,
                       ST_X(base_location::geometry)    AS base_lon,
                       assigned_zone_id, assigned_incident_id, deployed_at
                FROM resource_units
                ORDER BY resource_type, name
            """)
            return [dict(r) for r in rows]

    async def get_available_units(
        self, resource_type: ResourceType
    ) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, resource_type, name, status, capacity,
                       ST_Y(current_location::geometry) AS lat,
                       ST_X(current_location::geometry) AS lon
                FROM resource_units
                WHERE resource_type = $1 AND status = 'available'
                ORDER BY name
            """, resource_type.value)
            return [dict(r) for r in rows]

    async def get_inventory_summary(self) -> Dict[str, dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT resource_type,
                       COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE status='available')   AS available,
                       COUNT(*) FILTER (WHERE status='deployed')    AS deployed,
                       COUNT(*) FILTER (WHERE status='returning')   AS returning,
                       COUNT(*) FILTER (WHERE status='maintenance') AS maintenance
                FROM resource_units
                GROUP BY resource_type
            """)
            return {
                r["resource_type"]: {
                    "total":       int(r["total"]),
                    "available":   int(r["available"]),
                    "deployed":    int(r["deployed"]),
                    "returning":   int(r["returning"]),
                    "maintenance": int(r["maintenance"]),
                }
                for r in rows
            }

    async def get_recent_allocations(self, limit: int = 20) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, timestamp, incident_id, zone_id, zone_name,
                       urgency, num_people_affected, partial_allocation, status
                FROM resource_allocations
                ORDER BY timestamp DESC LIMIT $1
            """, limit)
            return [dict(r) for r in rows]

    async def get_allocation_by_id(self, allocation_id: UUID) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT *, ST_Y(destination::geometry) AS dest_lat,
                          ST_X(destination::geometry) AS dest_lon
                FROM resource_allocations WHERE id = $1
            """, allocation_id)
            return dict(row) if row else None

    # ── WRITE ─────────────────────────────────────────────────────────────

    async def mark_deployed(
        self,
        unit_id: UUID,
        incident_id: str,
        zone_id: str,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE resource_units
                SET status = 'deployed',
                    assigned_zone_id = $2,
                    assigned_incident_id = $3,
                    deployed_at = NOW(),
                    updated_at  = NOW()
                WHERE id = $1
            """, unit_id, zone_id, incident_id)

    async def mark_returned(self, unit_id: UUID) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE resource_units
                SET status = 'available',
                    assigned_zone_id = NULL,
                    assigned_incident_id = NULL,
                    deployed_at = NULL,
                    updated_at  = NOW()
                WHERE id = $1
            """, unit_id)

    async def add_resource_units(
        self,
        resource_type: ResourceType,
        name_prefix: str,
        quantity: int,
        lat: float,
        lon: float,
        capacity: int = 10,
    ) -> List[UUID]:
        """Insert new resource units (restock). Returns list of new UUIDs."""
        wkt = postgis_point_wkt(lat, lon)
        new_ids = []
        async with self.pool.acquire() as conn:
            for i in range(quantity):
                name = f"{name_prefix}-{i+1}" if quantity > 1 else name_prefix
                row = await conn.fetchrow("""
                    INSERT INTO resource_units
                        (resource_type, name, status, capacity,
                         current_location, base_location)
                    VALUES ($1, $2, 'available', $3,
                            $4::geography, $4::geography)
                    RETURNING id
                """, resource_type.value, name, capacity, wkt)
                new_ids.append(row["id"])

        await self._log_transaction(
            resource_type=resource_type,
            unit_ids=new_ids,
            direction="restocked",
            quantity=quantity,
            triggered_by="manual_restock",
        )
        return new_ids

    async def save_allocation(self, allocation: ResourceAllocation) -> None:
        wkt = postgis_point_wkt(
            allocation.destination.latitude, allocation.destination.longitude
        )
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO resource_allocations
                    (id, timestamp, incident_id, zone_id, zone_name,
                     destination, priority, urgency, num_people_affected,
                     allocated_units, partial_allocation, requires_medical, status)
                VALUES ($1,$2,$3,$4,$5,$6::geography,$7,$8,$9,$10,$11,$12,'pending')
            """,
                allocation.allocation_id,
                allocation.timestamp,
                allocation.incident_id,
                allocation.zone_id,
                allocation.zone_name,
                wkt,
                allocation.priority,
                allocation.urgency,
                allocation.num_people_affected,
                json.dumps(allocation.allocated_resources),
                allocation.partial_allocation,
                allocation.requires_medical,
            )

    async def _log_transaction(
        self,
        resource_type: ResourceType,
        unit_ids: List[UUID],
        direction: str,
        quantity: int,
        triggered_by: str,
        incident_id: str = None,
        zone_id: str = None,
    ) -> None:
        async with self.pool.acquire() as conn:
            for uid in unit_ids:
                await conn.execute("""
                    INSERT INTO inventory_transactions
                        (resource_type, unit_id, direction, quantity,
                         triggered_by, incident_id, zone_id)
                    VALUES ($1,$2,$3,$4,$5,$6,$7)
                """, resource_type.value, uid, direction, quantity,
                    triggered_by, incident_id, zone_id)

    # Expose for allocator to call after marking deployed
    async def log_allocation_transaction(
        self, units: List[dict], incident_id: str, zone_id: str
    ) -> None:
        async with self.pool.acquire() as conn:
            for u in units:
                await conn.execute("""
                    INSERT INTO inventory_transactions
                        (resource_type, unit_id, direction, quantity,
                         triggered_by, incident_id, zone_id)
                    VALUES ($1,$2,'allocated',1,'agent_3_auto',$3,$4)
                """, u["resource_type"], u["unit_id"], incident_id, zone_id)
