# dashboard/callbacks.py
"""
Callbacks
=========
All reactive logic lives here.  One function per concern:

  1. poll_agent_output   — fetches /output every N seconds → dcc.Store
  2. update_kpi_bar      — reads store → updates KPI numbers + clock
  3. update_map_layers   — reads store → builds zone circles, distress pins, routes
  4. update_tab_content  — reads store + active tab → renders agent panel
  5. update_feed         — simulates/reads Redis feed events
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from dash import Input, Output, State, callback, html, no_update
import dash_leaflet as dl

import theme as T
from api_client import get_output
from components.agent1_panel import build_agent1_panel
from components.agent2_panel import build_agent2_panel
from components.agent3_panel import build_agent3_panel
from components.agent4_panel import build_agent4_panel


# ── Feed state (module-level; survives between callback calls) ────────────────

_FEED_EVENTS: list[dict] = [
    {"time": "10:42", "agent": 1,
     "msg": "Detected 37% flooding in Sylhet via SAR. Published FLOOD_RISK alert to Redis."},
    {"time": "10:43", "agent": 2,
     "msg": "Received flood alert. Cross-referenced 5 reports in Sylhet. Escalated 3 to LIFE-THREATENING."},
    {"time": "10:43", "agent": 3,
     "msg": "Allocated 2 rescue boats + 1 medical team for Sylhet zone. Inventory updated."},
    {"time": "10:44", "agent": 4,
     "msg": "Computed routes for 3 teams. ETA: 25 min (boat), 40 min (medical). Routes live on map."},
    {"time": "10:42", "agent": 1,
     "msg": "Mirpur zone: MINIMAL risk. 0% flood. All clear."},
    {"time": "10:45", "agent": 1,
     "msg": "Sunamganj Sadar: risk=0.82. River 26.4 m³/s (ratio=1.03x, stable). SAR confirmed."},
    {"time": "10:45", "agent": 2,
     "msg": "New distress: Sunamganj — 'বন্যায় রাস্তা ডুবে গেছে'. Escalated. ~30 people at risk."},
    {"time": "10:46", "agent": 3,
     "msg": "3 rescue teams deployed to Sunamganj. Food packs: 120/200 remaining."},
    {"time": "10:46", "agent": 4,
     "msg": "Rescue T2 dispatched to Sunamganj. ETA 15 min. Route computed."},
]

_NEW_FEED_EVENTS = [
    {"agent": 1, "msg": "Netrokona Sadar: Satellite MINIMAL risk, 0.0% flood. River 32.2 m³/s (ratio=0.95x, falling)."},
    {"agent": 2, "msg": "Sylhet report cross-ref: satellite CONFIRMED. Priority escalated to CRITICAL."},
    {"agent": 3, "msg": "Medical kits restocked: +10 units. Total 45/60."},
    {"agent": 4, "msg": "Boat Unit A ETA updated: 22 min to Sylhet. Route recalculated."},
    {"agent": 1, "msg": "Sirajganj Sadar: River 0.3 m³/s (ratio=0.76x, trend=falling). Risk reduced to MINIMAL."},
    {"agent": 2, "msg": "New signal: 'Water rising fast in Netrokona'. Trilingual NLP: Bengali confirmed. Urgency: URGENT."},
    {"agent": 3, "msg": "Rescue boats: 12/20 remaining. 1 unit returned from Sirajganj."},
    {"agent": 4, "msg": "Medical T1 now En Route to Sylhet. ETA revised: 35 min."},
]

_feed_idx = 0


# ── Agent color/name helpers ──────────────────────────────────────────────────

_AGENT_NAME = {
    1: "Agent 1 (Environmental)",
    2: "Agent 2 (Distress)",
    3: "Agent 3 (Resource)",
    4: "Agent 4 (Dispatch)",
}
_AGENT_CLASS = {1: T.AGENT1, 2: T.AGENT2, 3: T.AGENT3, 4: T.AGENT4}


def _feed_event_component(ev: dict) -> html.Div:
    color = _AGENT_CLASS[ev["agent"]]
    return html.Div([
        html.P(ev["time"], style={
            "fontSize": "9px", "color": T.TEXT_MUTE, "margin": "0 0 2px",
        }),
        html.P(_AGENT_NAME[ev["agent"]], style={
            "fontFamily": T.FONT_UI, "fontSize": "10px", "fontWeight": "700",
            "color": color, "margin": "0 0 3px",
        }),
        html.P(ev["msg"], style={
            "fontSize": "10.5px", "color": T.TEXT, "lineHeight": "1.5", "margin": 0,
        }),
    ], style={
        "backgroundColor": T.BG3,
        "borderLeft": f"3px solid {color}",
        "borderRadius": "5px",
        "padding": "10px 12px",
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _zone_circle(pred: dict) -> list:
    """Return [pulse ring, filled circle, label marker] for one prediction."""
    zone      = pred["zone"]
    severity  = pred["severity_level"]
    risk_score= pred["risk_score"]
    color     = T.SEVERITY_COLOR.get(severity, T.TEXT_DIM)
    lat       = zone["center"]["latitude"]
    lng       = zone["center"]["longitude"]
    radius    = T.SEVERITY_RADIUS.get(severity, 10000)

    rf        = pred.get("risk_factors", {})
    sat_pct   = rf.get("satellite_flood_detection", 0) * 100
    has_river = rf.get("has_river_data", False)
    confirmed = rf.get("satellite_confirmed_flooding", False)
    pop       = zone.get("population_density", 0)
    elevation = zone.get("elevation", "?")
    drainage  = zone.get("drainage_capacity", "?")

    tooltip_html = f"""
        <div style="font-family:monospace;font-size:11px;background:{T.BG2};
                    border:1px solid {T.BORDER2};color:{T.TEXT};
                    padding:10px 14px;border-radius:5px;min-width:200px">
            <b style="font-size:13px;color:{color}">{zone['name']}</b><br/><br/>
            <span style="color:{T.TEXT_DIM}">Risk Score: </span>
            <b style="color:{color}">{risk_score*100:.1f}%</b>
            &nbsp;&nbsp;
            <span style="color:{T.TEXT_DIM}">Severity: </span>
            <b style="color:{color}">{severity.upper()}</b><br/>
            <span style="color:{T.TEXT_DIM}">SAR Flood: </span>{sat_pct:.0f}%
            {'&nbsp;🛰️ <b style="color:'+T.CRITICAL+'">CONFIRMED</b>' if confirmed else ''}<br/>
            <span style="color:{T.TEXT_DIM}">Population: </span>{pop:,}/km²<br/>
            <span style="color:{T.TEXT_DIM}">Elevation: </span>{elevation}m
            &nbsp;&nbsp;
            <span style="color:{T.TEXT_DIM}">Drainage: </span>{drainage}<br/>
            {'<span style="color:'+T.GREEN+'">● River data available</span>' if has_river else ''}
        </div>
    """

    return [
        # Outer dashed pulse ring
        dl.Circle(
            center=[lat, lng],
            radius=int(radius * 1.5),
            color=color, fillColor=color,
            fillOpacity=0.04, opacity=0.25, weight=1,
            dashArray="4 6",
        ),
        # Main filled circle
        dl.Circle(
            center=[lat, lng],
            radius=radius,
            color=color, fillColor=color,
            fillOpacity=0.18, opacity=0.9, weight=2,
            children=dl.Tooltip(
                html.Div(
                    dangerouslySetInnerHTML={"__html": tooltip_html},
                ),
                sticky=True,
            ),
        ),
        # Zone name label
        dl.Marker(
            position=[lat, lng],
            icon={
                "iconUrl": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "iconSize": [1, 1],
            },
            children=dl.Tooltip(
                zone["name"],
                permanent=True,
                direction="bottom",
                className="zone-label-tooltip",
            ),
        ),
    ]


_DISTRESS_POINTS = [
    {"lat": 23.800, "lng": 90.365,
     "msg": "Mirpur — bari dube jacche", "urgency": "LIFE-THREAT"},
    {"lat": 24.8975, "lng": 91.872,
     "msg": "Sylhet station — water entering homes", "urgency": "URGENT"},
    {"lat": 23.747, "lng": 90.415,
     "msg": "Kawran Bazar — road flooded", "urgency": "MODERATE"},
    {"lat": 24.866, "lng": 91.399,
     "msg": "Sunamganj — রাস্তা ডুবে গেছে", "urgency": "LIFE-THREAT"},
    {"lat": 24.870, "lng": 90.728,
     "msg": "Netrokona — water rising fast", "urgency": "URGENT"},
]

_URGENCY_COLOR = {
    "LIFE-THREAT": T.CRITICAL,
    "URGENT": T.HIGH,
    "MODERATE": T.MODERATE,
}

_TEAM_ROUTES = [
    {"from": [23.810, 90.412], "to": [24.8975, 91.872], "color": T.HIGH,    "label": "Boat Unit A → Sylhet"},
    {"from": [23.850, 90.430], "to": [24.8975, 91.872], "color": "#ffaa44", "label": "Boat Unit B → Sylhet"},
    {"from": [24.449, 89.700], "to": [24.449, 89.700],  "color": T.GREEN,   "label": "Boat Unit C — On site"},
    {"from": [24.000, 90.200], "to": [24.866, 91.399],  "color": T.BLUE,    "label": "Rescue T2 → Sunamganj"},
]


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER ALL CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def register_callbacks(app):

    # ── 1. Poll Agent 1 and write to store ───────────────────────────────────
    @app.callback(
        Output("store-agent-output", "data"),
        Input("interval-data", "n_intervals"),
    )
    def poll_agent_output(n):
        return get_output()

    # ── 2. Update KPI bar ────────────────────────────────────────────────────
    @app.callback(
        Output("kpi-zones",   "children"),
        Output("kpi-alerts",  "children"),
        Output("kpi-reports", "children"),
        Output("kpi-teams",   "children"),
        Output("kpi-people",  "children"),
        Input("store-agent-output", "data"),
    )
    def update_kpi(data):
        if not data:
            return "—", "—", "—", "—", "—"

        predictions = data.get("predictions", [])
        alerts      = data.get("alerts", [])

        zones_count  = len(predictions)
        alerts_count = len([p for p in predictions
                            if p.get("severity_level") in ("high", "critical")])
        # Static demo values for agents 2–4 (replace with real API calls later)
        reports = 42
        teams   = "5/8"
        people  = "12,400"

        return zones_count, alerts_count, reports, teams, people

    # ── 3. Live clock ────────────────────────────────────────────────────────
    @app.callback(
        Output("kpi-clock", "children"),
        Input("interval-clock", "n_intervals"),
    )
    def update_clock(n):
        now = datetime.now(timezone.utc)
        # Bangladesh Standard Time = UTC+6
        from datetime import timedelta
        bst = now + timedelta(hours=6)
        return bst.strftime("%H:%M:%S") + " BST"

    # ── 4. Map — zone circles ─────────────────────────────────────────────────
    @app.callback(
        Output("layer-zones", "children"),
        Input("store-agent-output", "data"),
    )
    def update_zone_circles(data):
        if not data:
            return []
        children = []
        for pred in data.get("predictions", []):
            children.extend(_zone_circle(pred))
        return children

    # ── 5. Map — distress pins (toggle-aware) ─────────────────────────────────
    @app.callback(
        Output("layer-distress", "children"),
        Input("btn-distress", "n_clicks"),
        Input("store-agent-output", "data"),
    )
    def update_distress_layer(n_clicks, data):
        # Odd clicks = show, even = hide
        if n_clicks % 2 == 0:
            return []
        markers = []
        for d in _DISTRESS_POINTS:
            color = _URGENCY_COLOR.get(d["urgency"], T.TEXT_DIM)
            markers.append(
                dl.CircleMarker(
                    center=[d["lat"], d["lng"]],
                    radius=7,
                    color=color, fillColor=color, fillOpacity=0.9, weight=2,
                    children=dl.Tooltip(
                        f"{d['urgency']} — {d['msg']}",
                        sticky=True,
                    ),
                )
            )
        return markers

    # ── 6. Map — team routes (toggle-aware) ───────────────────────────────────
    @app.callback(
        Output("layer-routes", "children"),
        Input("btn-routes", "n_clicks"),
    )
    def update_routes_layer(n_clicks):
        if n_clicks % 2 == 0:
            return []
        lines = []
        for r in _TEAM_ROUTES:
            lines.append(
                dl.Polyline(
                    positions=[r["from"], r["to"]],
                    color=r["color"], weight=2, opacity=0.75,
                    dashArray="6 4",
                    children=dl.Tooltip(r["label"]),
                )
            )
        return lines

    # ── 7. Tab content ────────────────────────────────────────────────────────
    @app.callback(
        Output("tab-content", "children"),
        Input("agent-tabs", "value"),
        Input("store-agent-output", "data"),
    )
    def update_tab(active_tab, data):
        if not data:
            return html.P("Loading...", style={"color": T.TEXT_DIM, "padding": "20px"})

        if active_tab == "tab-agent1":
            return build_agent1_panel(data)
        elif active_tab == "tab-agent2":
            return build_agent2_panel(data)
        elif active_tab == "tab-agent3":
            return build_agent3_panel(data)
        elif active_tab == "tab-agent4":
            return build_agent4_panel(data)
        return no_update

    # ── 8. Live feed ──────────────────────────────────────────────────────────
    @app.callback(
        Output("feed-list", "children"),
        Input("interval-feed", "n_intervals"),
        Input("store-agent-output", "data"),
    )
    def update_feed(n, data):
        global _feed_idx, _FEED_EVENTS

        # Inject one new synthetic event per tick
        if n > 0:
            ev = _NEW_FEED_EVENTS[_feed_idx % len(_NEW_FEED_EVENTS)]
            _feed_idx += 1
            now = datetime.now(timezone.utc)
            from datetime import timedelta
            bst_str = (now + timedelta(hours=6)).strftime("%H:%M")
            _FEED_EVENTS.insert(0, {"time": bst_str, "agent": ev["agent"], "msg": ev["msg"]})
            if len(_FEED_EVENTS) > 25:
                _FEED_EVENTS.pop()

        return [_feed_event_component(ev) for ev in _FEED_EVENTS[:15]]
