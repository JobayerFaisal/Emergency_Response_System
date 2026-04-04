"""
src/agents/agent_4_dispatch/main.py
Agent 4 — Dispatch Optimization FastAPI service.

Port: 8004
Subscribes: dispatch_order   (from Agent 3)
Publishes:  route_assignment (to Dashboard)
            agent_status     (heartbeat)
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import UUID

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis import asyncio as aioredis

from shared.message_protocol import AgentMessage
from shared.severity import GeoPoint
from .models import (
    RouteAssignment, TeamAssignment,
    TeamStatusUpdate, TeamStatus, TransportMode,
)
from .route_computer import RouteComputer
from .safety_checker import RouteSafetyChecker
from .dispatch_manager import DispatchManager
from .redis_handler import publish_message, listen_dispatch_order
from .ai_router import AIRoutingAdvisor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("agent4.main")

# ── App state ────────────────────────────────────────────────────────────────
db_pool: asyncpg.Pool        = None
redis_client: aioredis.Redis = None
route_computer: RouteComputer       = None
safety_checker: RouteSafetyChecker  = None
dispatch_manager: DispatchManager   = None
ai_router: AIRoutingAdvisor          = None

agent_state = {
    "connected": False,
    "last_action": None,
    "routes_computed": 0,
    "started_at": None,
}


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client, route_computer, safety_checker, dispatch_manager, ai_router

    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://disaster_admin:disaster123@localhost:5432/disaster_response")
    REDIS_URL    = os.getenv("REDIS_URL",    "redis://localhost:6379")
    OSRM_URL     = os.getenv("OSRM_URL",     "http://localhost:5000")

    db_pool      = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    logger.info("DB + Redis connected. OSRM at %s", OSRM_URL)

    route_computer   = RouteComputer(osrm_url=OSRM_URL)
    safety_checker   = RouteSafetyChecker(db_pool)
    dispatch_manager = DispatchManager(db_pool)
    ai_router        = AIRoutingAdvisor()

    agent_state["connected"] = True
    agent_state["started_at"] = datetime.now(timezone.utc).isoformat()

    listener_task  = asyncio.create_task(
        listen_dispatch_order(redis_client, handle_dispatch_order)
    )
    heartbeat_task = asyncio.create_task(send_heartbeat())

    yield

    listener_task.cancel()
    heartbeat_task.cancel()
    await db_pool.close()
    await redis_client.aclose()


app = FastAPI(title="Agent 4 — Dispatch Optimization", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Core handler ──────────────────────────────────────────────────────────────

async def handle_dispatch_order(envelope: AgentMessage):
    """
    Receives a ResourceAllocation from Agent 3 via dispatch_order channel.
    Computes optimal routes for each allocated resource unit.
    Publishes RouteAssignment to route_assignment channel.
    """
    agent_state["last_action"] = f"Received dispatch_order at {datetime.now(timezone.utc).isoformat()}"

    payload = envelope.payload
    try:
        allocation = _parse_allocation(payload)
    except Exception as e:
        logger.error("Failed to parse dispatch_order payload: %s", e)
        return

    logger.info(
        "Computing routes for incident %s — %d units → %s",
        allocation.incident_id, len(allocation.allocated_resources), allocation.zone_name,
    )

    teams = []
    safety_scores = []

    for resource in allocation.allocated_resources:
        origin = GeoPoint(
            latitude=resource["current_location"]["latitude"],
            longitude=resource["current_location"]["longitude"],
        )
        destination = allocation.destination

        rtype = resource["resource_type"]

        # Boats use waterway; everything else uses road
        transport_mode = (
            TransportMode.WATERWAY
            if rtype == "rescue_boat"
            else TransportMode.ROAD
        )

        # For boats, determine flood condition at destination
        flood_condition = "flood_shallow"
        if transport_mode == TransportMode.WATERWAY:
            flood_condition = await safety_checker.get_flood_condition_at_point(
                destination.latitude, destination.longitude
            )

        route_data = await route_computer.compute_route(
            origin=origin,
            destination=destination,
            transport_mode=transport_mode,
            flood_condition=flood_condition,
        )

        # Safety check
        safety = await safety_checker.check_route_safety(
            route_geojson=route_data["route_geometry"],
            transport_mode=transport_mode.value,
        )
        safety_scores.append(safety)

        # Ask GPT-4o for routing advice
        ai_advice = await ai_router.advise_route(
            unit_name=resource["name"],
            resource_type=rtype,
            origin_lat=origin.latitude,
            origin_lon=origin.longitude,
            dest_lat=destination.latitude,
            dest_lon=destination.longitude,
            dest_zone=allocation.zone_name,
            urgency=allocation.urgency,
            num_people=allocation.num_people_affected,
            distance_km=route_data["distance_km"],
            eta_minutes=route_data["eta_minutes"],
            route_safety_score=safety,
            flood_condition=flood_condition,
            transport_mode=transport_mode.value,
        )

        # Apply AI speed adjustment
        adjusted_eta = route_data["eta_minutes"]
        speed_adj    = ai_advice.get("speed_adjustment", 1.0)
        if speed_adj != 1.0:
            adjusted_eta = round(route_data["eta_minutes"] / speed_adj, 1)
            logger.info("AI adjusted ETA for %s: %.1f → %.1f min", resource["name"], route_data["eta_minutes"], adjusted_eta)

        if ai_advice.get("team_instruction"):
            logger.info("AI instruction for %s: %s", resource["name"], ai_advice["team_instruction"])
        if ai_advice.get("safety_warning"):
            logger.warning("AI safety warning for %s: %s", resource["name"], ai_advice["safety_warning"])

        teams.append(TeamAssignment(
            unit_id=resource["unit_id"],
            unit_name=resource["name"],
            resource_type=rtype,
            transport_mode=transport_mode,
            origin=origin,
            destination=destination,
            route_geometry=route_data["route_geometry"],
            distance_km=route_data["distance_km"],
            eta_minutes=adjusted_eta,
            status=TeamStatus.DISPATCHED,
            route_safety_score=safety,
        ))

    if not teams:
        logger.error("No teams to dispatch for incident %s", allocation.incident_id)
        return

    assignment = RouteAssignment(
        incident_id=allocation.incident_id,
        allocation_id=str(allocation.allocation_id),
        zone_id=allocation.zone_id,
        zone_name=allocation.zone_name,
        destination=allocation.destination,
        priority=allocation.priority,
        urgency=allocation.urgency,
        teams=teams,
        total_eta_minutes=max(t.eta_minutes for t in teams),
        route_safety_score=min(safety_scores) if safety_scores else 1.0,
        partial_allocation=allocation.partial_allocation,
        notes=allocation.notes,
    )

    # Save to DB
    await dispatch_manager.save_route_assignment(assignment)

    # Publish to Dashboard
    await publish_message(
        redis=redis_client,
        db_pool=db_pool,
        channel="route_assignment",
        receiver="all",
        message_type="route_assignment",
        payload=assignment.model_dump(mode="json"),
        zone_id=assignment.zone_id,
        priority=assignment.priority,
    )

    agent_state["routes_computed"] += 1
    agent_state["last_action"] = (
        f"Dispatched {len(teams)} teams → {assignment.zone_name} "
        f"(ETA={assignment.total_eta_minutes:.0f}min)"
    )
    logger.info(
        "RouteAssignment %s: %d teams, max ETA=%.0f min, safety=%.2f",
        assignment.assignment_id, len(teams),
        assignment.total_eta_minutes, assignment.route_safety_score,
    )


def _parse_allocation(payload: dict):
    """Parse ResourceAllocation from Agent 3 payload dict."""
    from shared.severity import GeoPoint
    from .models import RouteAssignment

    dest = payload.get("destination", {})
    # Handles both nested dict and flat lat/lon
    if isinstance(dest, dict) and "latitude" in dest:
        destination = GeoPoint(latitude=dest["latitude"], longitude=dest["longitude"])
    else:
        destination = GeoPoint(latitude=dest.get("lat", 0), longitude=dest.get("lon", 0))

    class _Alloc:
        pass

    alloc = _Alloc()
    alloc.incident_id          = payload["incident_id"]
    alloc.allocation_id        = payload["allocation_id"]
    alloc.zone_id              = payload["zone_id"]
    alloc.zone_name            = payload.get("zone_name", "Unknown")
    alloc.destination          = destination
    alloc.priority             = payload.get("priority", 3)
    alloc.urgency              = payload.get("urgency", "MODERATE")
    alloc.allocated_resources  = payload.get("allocated_resources", [])
    alloc.requires_medical     = payload.get("requires_medical", False)
    alloc.partial_allocation   = payload.get("partial_allocation", False)
    alloc.notes                = payload.get("notes", "")
    return alloc


async def send_heartbeat():
    while True:
        await asyncio.sleep(30)
        try:
            await publish_message(
                redis=redis_client,
                db_pool=db_pool,
                channel="agent_status",
                receiver="all",
                message_type="heartbeat",
                payload={
                    "agent": "agent_4_dispatch",
                    "status": "healthy",
                    "routes_computed": agent_state["routes_computed"],
                },
                priority=1,
            )
        except Exception as e:
            logger.error("Heartbeat failed: %s", e)


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "agent": "Agent 4 — Dispatch Optimization",
        "version": "1.0.0",
        "subscribes_to": "dispatch_order",
        "publishes_to": ["route_assignment", "agent_status"],
    }


@app.get("/health")
async def health():
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        await redis_client.ping()
        # Check OSRM
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            osrm_resp = await client.get(f"{os.getenv('OSRM_URL','http://localhost:5000')}/")
        osrm_ok = osrm_resp.status_code < 500
    except Exception:
        osrm_ok = False
    return {
        "status": "healthy",
        "db": "ok",
        "redis": "ok",
        "osrm": "ok" if osrm_ok else "unavailable (fallback active)",
    }


@app.get("/routes")
async def get_routes():
    return {"routes": await dispatch_manager.get_active_routes()}


@app.get("/routes/{route_id}")
async def get_route(route_id: UUID):
    result = await dispatch_manager.get_route_by_id(route_id)
    if not result:
        raise HTTPException(404, "Route not found")
    return result


@app.get("/routes/{route_id}/geojson")
async def get_route_geojson(route_id: UUID):
    """Returns pure GeoJSON FeatureCollection — consumed by Folium/Leaflet map."""
    result = await dispatch_manager.get_route_geojson(route_id)
    if not result:
        raise HTTPException(404, "Route not found or has no geometry")
    return result


@app.get("/teams")
async def get_teams():
    return {"teams": await dispatch_manager.get_all_teams()}


@app.get("/teams/{team_id}")
async def get_team(team_id: UUID):
    result = await dispatch_manager.get_team_by_id(team_id)
    if not result:
        raise HTTPException(404, "Team not found")
    return result


@app.post("/teams/{team_id}/status")
async def update_team_status(team_id: UUID, update: TeamStatusUpdate):
    success = await dispatch_manager.update_team_status(team_id, update.status)
    if not success:
        raise HTTPException(404, "Team not found")

    # If team arrived, also update the resource unit in Agent 3's DB
    if update.status == TeamStatus.ARRIVED:
        logger.info("Team %s marked ARRIVED — resource should be returned by Agent 3", team_id)

    return {"status": "updated", "team_id": str(team_id), "new_status": update.status}


@app.post("/trigger")
async def manual_trigger(payload: dict):
    """
    Manually inject a dispatch_order payload — for testing without Agent 3.

    Body: same structure as ResourceAllocation model_dump
    Example:
    {
      "incident_id": "INC-TEST-001",
      "allocation_id": "...",
      "zone_id": "mirpur-10",
      "zone_name": "Mirpur Section 10, Dhaka",
      "destination": {"latitude": 23.8058, "longitude": 90.3689},
      "priority": 5,
      "urgency": "LIFE_THREATENING",
      "allocated_resources": [
        {
          "unit_id": "...",
          "resource_type": "rescue_boat",
          "name": "Boat Mirpur-1",
          "current_location": {"latitude": 23.8041, "longitude": 90.3654}
        }
      ],
      "requires_medical": true,
      "partial_allocation": false,
      "notes": "Manual test"
    }
    """
    fake_envelope = AgentMessage(
        sender_agent="manual_trigger",
        receiver_agent="agent_4_dispatch",
        message_type="dispatch_order",
        payload=payload,
        priority=5,
    )
    await handle_dispatch_order(fake_envelope)
    return {"status": "triggered"}


@app.get("/status")
async def get_status():
    return {
        "agent": "agent_4_dispatch",
        "connected": agent_state["connected"],
        "started_at": agent_state["started_at"],
        "routes_computed": agent_state["routes_computed"],
        "last_action": agent_state["last_action"],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AGENT_PORT", 8000))
    uvicorn.run("src.agents.agent_4_dispatch.main:app", host="0.0.0.0", port=port, reload=False)
