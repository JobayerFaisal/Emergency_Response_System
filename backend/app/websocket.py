"""
backend/app/websocket.py
=========================
WebSocket endpoint: ws://localhost:8005/ws

The frontend connects here.  Every message published on ANY of the
6 Redis channels (flood_alert, distress_queue, dispatch_order,
route_assignment, inventory_update, agent_status) is forwarded here
in real-time via the ConnectionManager in redis_bridge.py.

Protocol (server → client):
  {
    "channel":  "flood_alert",          ← which Redis channel
    "event":    "flood_alert",          ← alias (same value)
    "data":     { ...AgentMessage... }  ← parsed JSON payload
  }

Protocol (client → server):
  { "type": "ping" }  → { "type": "pong" }
  { "type": "subscribe", "channels": ["flood_alert"] }  → acknowledged
  (no-op — all channels are always broadcast; subscription is just cosmetic)
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from app.services.redis_bridge import manager

logger = logging.getLogger("dashboard.websocket")

router = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """
    Main WebSocket endpoint.

    1. Accepts the connection and registers it with ConnectionManager.
    2. Starts a receive loop to handle client pings / subscription messages.
    3. The redis_bridge broadcasts to all registered connections — so this
       endpoint doesn't need to do anything special for outbound messages.
    """

    await manager.connect(ws)

    # Send a welcome / handshake frame so the client knows it's live
    await ws.send_json({
        "channel": "system",
        "event":   "connected",
        "data": {
            "message":             "Connected to Emergency Response System",
            "active_connections":  manager.active_connections,
        },
    })

    try:
        while True:
            # Keep the connection alive; handle any client messages
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a keepalive ping so the browser doesn't time out
                await ws.send_json({"channel": "system", "event": "keepalive"})
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "ping":
                await ws.send_json({"type": "pong"})

            elif msg_type == "subscribe":
                # Acknowledge subscription (we broadcast everything regardless)
                channels = msg.get("channels", [])
                await ws.send_json({
                    "channel": "system",
                    "event":   "subscribed",
                    "data":    {"channels": channels},
                })

            elif msg_type == "get_stats":
                await ws.send_json({
                    "channel": "system",
                    "event":   "stats",
                    "data":    {"active_connections": manager.active_connections},
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("WS error: %s", exc)
    finally:
        manager.disconnect(ws)