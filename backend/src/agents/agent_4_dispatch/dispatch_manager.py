"""
src/agents/agent_4_dispatch/dispatch_manager.py
Database CRUD for dispatch_routes and team_routes tables.
"""

import json
import logging
from typing import List, Optional
from uuid import UUID

import asyncpg

from shared.geo_utils import postgis_point_wkt
from .models import RouteAssignment, TeamAssignment, TeamStatus

logger = logging.getLogger("agent4.dispatch_manager")


class DispatchManager:

    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool

    # ── READ ──────────────────────────────────────────────────────────────

    async def get_active_routes(self) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT dr.id, dr.timestamp, dr.incident_id, dr.zone_name,
                       dr.priority, dr.total_eta_minutes, dr.route_safety_score,
                       dr.status,
                       ST_Y(dr.destination::geometry) AS dest_lat,
                       ST_X(dr.destination::geometry) AS dest_lon,
                       COUNT(tr.id) AS team_count
                FROM dispatch_routes dr
                LEFT JOIN team_routes tr ON tr.dispatch_id = dr.id
                WHERE dr.status = 'active'
                GROUP BY dr.id
                ORDER BY dr.priority DESC, dr.timestamp DESC
            """)
            return [dict(r) for r in rows]

    async def get_route_by_id(self, route_id: UUID) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT dr.*,
                       ST_Y(dr.destination::geometry) AS dest_lat,
                       ST_X(dr.destination::geometry) AS dest_lon
                FROM dispatch_routes dr WHERE dr.id = $1
            """, route_id)
            if not row:
                return None
            result = dict(row)
            # Attach team routes
            teams = await conn.fetch("""
                SELECT tr.*,
                       ST_Y(tr.origin::geometry)      AS origin_lat,
                       ST_X(tr.origin::geometry)      AS origin_lon,
                       ST_Y(tr.destination::geometry) AS dest_lat,
                       ST_X(tr.destination::geometry) AS dest_lon
                FROM team_routes tr WHERE tr.dispatch_id = $1
            """, route_id)
            result["teams"] = [dict(t) for t in teams]
            return result

    async def get_route_geojson(self, route_id: UUID) -> Optional[dict]:
        """Return all team route geometries as a GeoJSON FeatureCollection."""
        async with self.pool.acquire() as conn:
            teams = await conn.fetch("""
                SELECT unit_name, resource_type, transport_mode,
                       eta_minutes, status, route_geometry
                FROM team_routes WHERE dispatch_id = $1
            """, route_id)

        if not teams:
            return None

        features = []
        for t in teams:
            geom = t["route_geometry"]
            if isinstance(geom, str):
                geom = json.loads(geom)
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "unit_name":      t["unit_name"],
                    "resource_type":  t["resource_type"],
                    "transport_mode": t["transport_mode"],
                    "eta_minutes":    t["eta_minutes"],
                    "status":         t["status"],
                },
            })
        return {"type": "FeatureCollection", "features": features}

    async def get_all_teams(self) -> List[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT tr.id, tr.unit_name, tr.resource_type, tr.transport_mode,
                       tr.eta_minutes, tr.status, tr.departed_at, tr.arrived_at,
                       dr.zone_name, dr.priority,
                       ST_Y(tr.destination::geometry) AS dest_lat,
                       ST_X(tr.destination::geometry) AS dest_lon
                FROM team_routes tr
                JOIN dispatch_routes dr ON dr.id = tr.dispatch_id
                WHERE dr.status = 'active'
                ORDER BY dr.priority DESC, tr.eta_minutes ASC
            """)
            return [dict(r) for r in rows]

    async def get_team_by_id(self, team_id: UUID) -> Optional[dict]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM team_routes WHERE id = $1", team_id
            )
            return dict(row) if row else None

    # ── WRITE ─────────────────────────────────────────────────────────────

    async def save_route_assignment(self, assignment: RouteAssignment) -> None:
        dest_wkt = postgis_point_wkt(
            assignment.destination.latitude,
            assignment.destination.longitude,
        )
        async with self.pool.acquire() as conn:
            # Insert dispatch_routes row
            await conn.execute("""
                INSERT INTO dispatch_routes
                    (id, timestamp, incident_id, zone_id, zone_name,
                     destination, priority, total_eta_minutes,
                     route_safety_score, status)
                VALUES ($1,$2,$3,$4,$5,$6::geography,$7,$8,$9,'active')
                ON CONFLICT (id) DO NOTHING
            """,
                assignment.assignment_id,
                assignment.timestamp,
                assignment.incident_id,
                assignment.zone_id,
                assignment.zone_name,
                dest_wkt,
                assignment.priority,
                assignment.total_eta_minutes,
                assignment.route_safety_score,
            )

            # Insert team_routes rows
            for team in assignment.teams:
                origin_wkt = postgis_point_wkt(
                    team.origin.latitude, team.origin.longitude
                )
                team_dest_wkt = postgis_point_wkt(
                    team.destination.latitude, team.destination.longitude
                )
                geom_json = json.dumps(team.route_geometry) if team.route_geometry else None
                await conn.execute("""
                    INSERT INTO team_routes
                        (dispatch_id, unit_id, unit_name, resource_type,
                         transport_mode, origin, destination,
                         route_geometry, distance_km, eta_minutes, status)
                    VALUES ($1,$2,$3,$4,$5,$6::geography,$7::geography,
                            $8::jsonb,$9,$10,$11)
                """,
                    assignment.assignment_id,
                    team.unit_id,
                    team.unit_name,
                    team.resource_type,
                    team.transport_mode.value,
                    origin_wkt,
                    team_dest_wkt,
                    geom_json,
                    team.distance_km,
                    team.eta_minutes,
                    team.status.value,
                )

    async def update_team_status(
        self, team_id: UUID, new_status: TeamStatus, notes: str = None
    ) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE team_routes SET status = $2,
                    arrived_at = CASE WHEN $2 = 'arrived' THEN NOW() ELSE arrived_at END,
                    departed_at = CASE WHEN $2 = 'en_route' THEN NOW() ELSE departed_at END
                WHERE id = $1
            """, team_id, new_status.value)
            return result != "UPDATE 0"

    async def mark_dispatch_completed(self, route_id: UUID) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE dispatch_routes
                SET status = 'completed', completed_at = NOW()
                WHERE id = $1
            """, route_id)
