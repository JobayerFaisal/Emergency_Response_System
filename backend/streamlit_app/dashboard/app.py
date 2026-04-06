"""
Emergency Response System — Streamlit Dashboard
================================================
3-screen dashboard:
  Screen 1 (tab 0): Environmental Monitor — MSN-style flood map
  Screen 2 (tab 1): Distress & Dispatch   — incidents + route map
  Screen 3 (tab 2): Simulation Control    — Sylhet 2022 replay

Backend: reads directly from PostgreSQL + Redis
Auto-refreshes every 30 seconds
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg
import folium
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import AntPath, MarkerCluster
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emergency Response System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit default header/footer */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1rem; padding-bottom: 0rem;}

/* KPI card styling */
.kpi-card {
    background: linear-gradient(135deg, #0f1923 0%, #1a2535 100%);
    border: 1px solid rgba(56, 139, 253, 0.3);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}
.kpi-label {font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em;}
.kpi-value {font-size: 24px; font-weight: 700; color: #f0f6fc; margin: 4px 0 2px;}
.kpi-sub {font-size: 11px; color: #58a6ff;}

/* Status badge */
.badge-critical {background:#3d1a1a; color:#f85149; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600;}
.badge-high {background:#3d2d1a; color:#e3b341; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600;}
.badge-moderate {background:#1a2d3d; color:#58a6ff; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600;}
.badge-minimal {background:#1a3d2d; color:#3fb950; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600;}

/* Agent feed entry */
.feed-entry {
    border-left: 3px solid;
    padding: 6px 10px;
    margin-bottom: 6px;
    border-radius: 0 6px 6px 0;
    background: rgba(255,255,255,0.03);
    font-size: 12px;
}
.feed-agent1 {border-color: #3fb950;}
.feed-agent2 {border-color: #e3b341;}
.feed-agent3 {border-color: #58a6ff;}
.feed-agent4 {border-color: #f85149;}

/* Dashboard header */
.dash-header {
    background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid rgba(56,139,253,0.2);
    padding: 8px 0 12px;
    margin-bottom: 12px;
}
.system-title {
    font-size: 18px; font-weight: 700; color: #f0f6fc;
    display: inline-flex; align-items: center; gap: 8px;
}
.live-dot {
    width: 8px; height: 8px; background: #3fb950;
    border-radius: 50%; display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% {opacity:1; transform:scale(1);}
    50% {opacity:0.5; transform:scale(1.3);}
}

/* Metric panel */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid rgba(56,139,253,0.15);
    border-radius: 8px;
    padding: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Config ──────────────────────────────────────────────────────────────────────
DB_URL     = os.getenv("DATABASE_URL_ASYNC", "postgresql://postgres:postgres@localhost:5432/disaster_response")
REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379")
AGENT1_URL = os.getenv("AGENT1_URL", "http://localhost:8001")
AGENT2_URL = os.getenv("AGENT2_URL", "http://localhost:8002")
AGENT3_URL = os.getenv("AGENT3_URL", "http://localhost:8003")
AGENT4_URL = os.getenv("AGENT4_URL", "http://localhost:8004")

# Auto-refresh every 30s
st_autorefresh(interval=30000, key="autorefresh")

# ── DB helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=25)
def fetch_data(query: str, params: tuple = ()) -> List[Dict]:
    """Fetch data from PostgreSQL."""
    async def _fetch():
        try:
            conn = await asyncpg.connect(DB_URL)
            rows = await conn.fetch(query, *params)
            await conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            return []
    return asyncio.run(_fetch())


def fetch_zones() -> List[Dict]:
    return fetch_data("""
        SELECT id::text, name, risk_level,
               ST_Y(center::geometry) AS lat,
               ST_X(center::geometry) AS lon,
               radius_km, population_density, elevation,
               drainage_capacity, last_monitored
        FROM sentinel_zones ORDER BY name
    """)


def fetch_predictions() -> List[Dict]:
    return fetch_data("""
        SELECT p.id::text, p.zone_id::text, z.name AS zone_name,
               p.risk_score, p.severity_level, p.confidence,
               p.affected_area_km2, p.risk_factors, p.timestamp
        FROM flood_predictions p
        JOIN sentinel_zones z ON z.id = p.zone_id
        WHERE p.timestamp > NOW() - INTERVAL '2 hours'
        ORDER BY p.timestamp DESC
    """)


def fetch_weather() -> List[Dict]:
    return fetch_data("""
        SELECT w.zone_id::text, z.name AS zone_name,
               w.temperature, w.humidity, w.wind_speed,
               w.precipitation_1h, w.precipitation_24h,
               w.condition, w.timestamp
        FROM weather_data w
        JOIN sentinel_zones z ON z.id = w.zone_id
        WHERE w.timestamp > NOW() - INTERVAL '3 hours'
        ORDER BY w.timestamp DESC
    """)


def fetch_allocations() -> List[Dict]:
    return fetch_data("""
        SELECT id::text, incident_id, zone_id, zone_name,
               priority, urgency, num_people_affected,
               allocated_units, partial_allocation, timestamp
        FROM resource_allocations
        ORDER BY timestamp DESC LIMIT 20
    """)


def fetch_dispatch_routes() -> List[Dict]:
    return fetch_data("""
        SELECT dr.id::text, dr.incident_id, dr.zone_name,
               dr.priority, dr.total_eta_minutes,
               dr.route_safety_score, dr.status,
               COUNT(tr.id) AS team_count
        FROM dispatch_routes dr
        LEFT JOIN team_routes tr ON tr.dispatch_id = dr.id
        WHERE dr.status = 'active'
        GROUP BY dr.id ORDER BY dr.timestamp DESC LIMIT 10
    """)


def fetch_team_routes() -> List[Dict]:
    return fetch_data("""
        SELECT tr.id::text, tr.unit_name, tr.resource_type,
               tr.transport_mode, tr.distance_km, tr.eta_minutes,
               tr.status, tr.route_geometry,
               ST_Y(tr.origin::geometry) AS origin_lat,
               ST_X(tr.origin::geometry) AS origin_lon,
               ST_Y(tr.destination::geometry) AS dest_lat,
               ST_X(tr.destination::geometry) AS dest_lon
        FROM team_routes tr
        JOIN dispatch_routes dr ON dr.id = tr.dispatch_id
        WHERE dr.status = 'active'
        ORDER BY tr.eta_minutes ASC LIMIT 30
    """)


def fetch_agent_messages() -> List[Dict]:
    return fetch_data("""
        SELECT sender_agent, receiver_agent, message_type,
               zone_id, priority, timestamp, payload
        FROM agent_messages
        ORDER BY timestamp DESC LIMIT 50
    """)


def fetch_inventory() -> List[Dict]:
    return fetch_data("""
        SELECT resource_type,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE status='available') AS available,
               COUNT(*) FILTER (WHERE status='deployed') AS deployed,
               COUNT(*) FILTER (WHERE status='returning') AS returning
        FROM resource_units
        GROUP BY resource_type ORDER BY resource_type
    """)


# ── Agent API helpers ──────────────────────────────────────────────────────────
@st.cache_data(ttl=20)
def call_agent(url: str, path: str) -> Optional[Dict]:
    try:
        r = httpx.get(f"{url}{path}", timeout=5.0)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ── Risk color helpers ─────────────────────────────────────────────────────────
RISK_COLORS = {
    "critical": "#f85149",
    "high":     "#e3b341",
    "moderate": "#58a6ff",
    "low":      "#79c0ff",
    "minimal":  "#3fb950",
}
RISK_FILL = {
    "critical": "#f8514940",
    "high":     "#e3b34130",
    "moderate": "#58a6ff25",
    "low":      "#79c0ff15",
    "minimal":  "#3fb95010",
}


# ── Build flood map ────────────────────────────────────────────────────────────
def build_flood_map(zones, predictions, weather_data, show_rain=True, show_rivers=True):
    """Build MSN-style flood risk map with Folium."""
    center_lat = sum(z["lat"] for z in zones) / len(zones) if zones else 24.5
    center_lon = sum(z["lon"] for z in zones) / len(zones) if zones else 90.5

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=None,
        prefer_canvas=True,
    )

    # Base tile layers
    folium.TileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap contributors © CARTO",
        name="Dark (default)",
        max_zoom=19,
    ).add_to(m)

    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Satellite",
    ).add_to(m)

    # Build prediction lookup
    pred_by_zone = {}
    for p in predictions:
        zid = p.get("zone_id")
        if zid not in pred_by_zone or p["timestamp"] > pred_by_zone[zid]["timestamp"]:
            pred_by_zone[zid] = p

    weather_by_zone = {}
    for w in weather_data:
        zid = w.get("zone_id")
        if zid not in weather_by_zone:
            weather_by_zone[zid] = w

    # Flood zone layer group
    flood_fg = folium.FeatureGroup(name="Flood risk zones", show=True)

    for zone in zones:
        zid = zone["id"]
        pred = pred_by_zone.get(zid)
        risk = pred["severity_level"] if pred else zone.get("risk_level", "minimal")
        risk_score = pred["risk_score"] if pred else 0.0
        color = RISK_COLORS.get(risk, "#58a6ff")
        fill  = RISK_FILL.get(risk, "#58a6ff20")

        radius_m = zone.get("radius_km", 5) * 1000

        # Flood fill circle (MSN-style blue overlay)
        folium.Circle(
            location=[zone["lat"], zone["lon"]],
            radius=radius_m,
            color=color,
            fill=True,
            fill_color=fill,
            fill_opacity=min(0.7, 0.1 + risk_score * 0.8),
            weight=2,
            opacity=0.9,
            tooltip=folium.Tooltip(f"""
                <b>{zone['name']}</b><br>
                Risk: <b style='color:{color}'>{risk.upper()}</b><br>
                Score: {risk_score:.2f}<br>
                Pop: {zone.get('population_density', 'N/A'):,}
            """),
            popup=folium.Popup(f"""
                <div style='min-width:200px;font-family:sans-serif'>
                <h4 style='margin:0 0 8px;color:{color}'>{zone['name']}</h4>
                <table style='width:100%;font-size:12px'>
                <tr><td>Risk Level</td><td><b>{risk.upper()}</b></td></tr>
                <tr><td>Risk Score</td><td>{risk_score:.3f}</td></tr>
                <tr><td>Elevation</td><td>{zone.get('elevation','?')} m</td></tr>
                <tr><td>Drainage</td><td>{zone.get('drainage_capacity','?')}</td></tr>
                <tr><td>Population</td><td>{zone.get('population_density', 0):,}/km²</td></tr>
                </table>
                </div>
            """, max_width=250),
        ).add_to(flood_fg)

        # Zone center marker
        icon_color = "red" if risk == "critical" else "orange" if risk == "high" else "blue" if risk == "moderate" else "green"
        folium.Marker(
            location=[zone["lat"], zone["lon"]],
            icon=folium.Icon(color=icon_color, icon="tint", prefix="fa"),
            tooltip=zone["name"],
        ).add_to(flood_fg)

    flood_fg.add_to(m)

    # Rain overlay layer
    if show_rain:
        rain_fg = folium.FeatureGroup(name="Precipitation overlay", show=True)
        for zone in zones:
            w = weather_by_zone.get(zone["id"], {})
            precip = w.get("precipitation_1h") or 0
            humidity = w.get("humidity") or 0

            if humidity > 70 or precip > 0:
                # Outer rain halo
                folium.Circle(
                    location=[zone["lat"], zone["lon"]],
                    radius=zone.get("radius_km", 5) * 1500,
                    color="#4da9ff",
                    fill=True,
                    fill_color="#4da9ff",
                    fill_opacity=0.04 + min(0.15, humidity / 1000),
                    weight=0,
                ).add_to(rain_fg)

                # Rain intensity rings
                if precip > 5:
                    for i in range(3):
                        folium.Circle(
                            location=[zone["lat"], zone["lon"]],
                            radius=zone.get("radius_km", 5) * 600 * (i + 1),
                            color="#4da9ff",
                            fill=False,
                            weight=1,
                            opacity=0.15 - i * 0.04,
                            dash_array="5,8",
                        ).add_to(rain_fg)
        rain_fg.add_to(m)

    # Layer control
    folium.LayerControl(position="topright").add_to(m)

    # Scale bar
    folium.plugins.MeasureControl(position="bottomleft").add_to(m)

    return m


# ── Build route map ────────────────────────────────────────────────────────────
def build_route_map(team_routes, zones):
    """Build dispatch route map with OSRM polylines."""
    center = [24.5, 90.5]
    if zones:
        center = [zones[0]["lat"], zones[0]["lon"]]

    m = folium.Map(
        location=center,
        zoom_start=8,
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="© CARTO",
    )

    # Zone markers
    for zone in zones:
        folium.CircleMarker(
            location=[zone["lat"], zone["lon"]],
            radius=8,
            color="#f85149",
            fill=True,
            fill_color="#f8514920",
            fill_opacity=0.4,
            tooltip=f"⚠ {zone['name']}",
        ).add_to(m)

    # Team routes
    RESOURCE_COLORS = {
        "rescue_boat":  "#58a6ff",
        "medical_team": "#f85149",
        "medical_kit":  "#e3b341",
        "food_supply":  "#3fb950",
        "water_supply": "#79c0ff",
    }
    RESOURCE_ICONS = {
        "rescue_boat":  "anchor",
        "medical_team": "plus-square",
        "medical_kit":  "first-aid",
        "food_supply":  "shopping-basket",
        "water_supply": "tint",
    }

    for tr in team_routes:
        color = RESOURCE_COLORS.get(tr.get("resource_type", ""), "#58a6ff")
        icon  = RESOURCE_ICONS.get(tr.get("resource_type", ""), "map-marker")

        origin = [tr.get("origin_lat", 0), tr.get("origin_lon", 0)]
        dest   = [tr.get("dest_lat", 0), tr.get("dest_lon", 0)]

        if not all(origin + dest):
            continue

        # Route line (animated)
        route_geom = tr.get("route_geometry")
        if route_geom and isinstance(route_geom, dict):
            coords = route_geom.get("coordinates", [])
            if coords:
                # GeoJSON is [lon, lat] — flip to [lat, lon] for folium
                latlon = [[c[1], c[0]] for c in coords]
                AntPath(
                    locations=latlon,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    delay=800,
                    tooltip=f"{tr['unit_name']} — ETA {tr.get('eta_minutes', '?'):.0f} min",
                ).add_to(m)
        else:
            # Straight line fallback
            AntPath(
                locations=[origin, dest],
                color=color,
                weight=2,
                opacity=0.6,
                delay=1000,
                dash_array=[10, 20],
            ).add_to(m)

        # Origin marker (resource base)
        folium.Marker(
            location=origin,
            icon=folium.Icon(color="gray", icon=icon, prefix="fa"),
            tooltip=f"{tr['unit_name']} (origin)",
        ).add_to(m)

        # Destination marker
        folium.CircleMarker(
            location=dest,
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=f"Destination — {tr.get('eta_minutes', '?'):.0f} min ETA",
        ).add_to(m)

    return m


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-header'>
  <span class='system-title'>
    <span class='live-dot'></span>
    Emergency Response System
  </span>
  &nbsp;&nbsp;
  <span style='font-size:13px;color:#8b949e'>Bangladesh Flood Intelligence</span>
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading..."):
    zones       = fetch_zones()
    predictions = fetch_predictions()
    weather     = fetch_weather()
    allocations = fetch_allocations()
    routes      = fetch_dispatch_routes()
    teams       = fetch_team_routes()
    messages    = fetch_agent_messages()
    inventory   = fetch_inventory()

# ── KPI Bar ────────────────────────────────────────────────────────────────────
latest_preds = {}
for p in predictions:
    zid = p.get("zone_id")
    if zid not in latest_preds:
        latest_preds[zid] = p

total_zones    = len(zones)
active_alerts  = sum(1 for p in latest_preds.values() if p.get("severity_level") not in ("minimal", "low"))
teams_deployed = sum(r.get("team_count", 0) for r in routes)
people_at_risk = sum(a.get("num_people_affected", 0) for a in allocations[:5])
max_risk       = max((p.get("risk_score", 0) for p in latest_preds.values()), default=0)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Zones Monitored", total_zones, help="Active sentinel zones")
with k2:
    st.metric("Active Alerts", active_alerts, delta=None if active_alerts == 0 else f"+{active_alerts}")
with k3:
    st.metric("Teams Deployed", teams_deployed)
with k4:
    st.metric("People at Risk", f"{people_at_risk:,}" if people_at_risk else "—")
with k5:
    st.metric("Max Risk Score", f"{max_risk:.2f}")
with k6:
    agent_status = call_agent(AGENT1_URL, "/health")
    status_text = "All Systems OK" if agent_status and agent_status.get("status") == "ok" else "Degraded"
    st.metric("System Status", status_text)

st.divider()

# ── Main Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌊 Environmental Monitor",
    "🚁 Distress & Dispatch",
    "🎮 Simulation Control",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ENVIRONMENTAL MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_map, col_feed = st.columns([3, 1])

    with col_map:
        # Map controls
        mc1, mc2, mc3 = st.columns(3)
        show_rain    = mc1.toggle("Rain overlay", value=True)
        show_rivers  = mc2.toggle("River levels", value=True)
        _base_layer  = mc3.selectbox("Base", ["Dark", "Satellite"], label_visibility="collapsed")

        if not zones:
            st.warning("No sentinel zones found in database.")
        else:
            flood_map = build_flood_map(zones, predictions, weather, show_rain, show_rivers)
            map_data  = st_folium(flood_map, width="100%", height=520, returned_objects=["last_object_clicked"])

    with col_feed:
        st.markdown("**Live Agent Feed**")
        agent_colors = {
            "agent_1_environmental": "feed-agent1",
            "agent_2_distress":      "feed-agent2",
            "agent_3_resource":      "feed-agent3",
            "agent_4_dispatch":      "feed-agent4",
        }
        agent_labels = {
            "agent_1_environmental": "Agent 1",
            "agent_2_distress":      "Agent 2",
            "agent_3_resource":      "Agent 3",
            "agent_4_dispatch":      "Agent 4",
        }

        feed_html = ""
        for msg in messages[:20]:
            sender  = msg.get("sender_agent", "")
            css_cls = agent_colors.get(sender, "feed-agent1")
            label   = agent_labels.get(sender, sender)
            mtype   = msg.get("message_type", "")
            ts      = msg.get("timestamp")
            ts_str  = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)[:19]
            zone    = msg.get("zone_id", "")
            prio    = msg.get("priority", 3)
            prio_color = "#f85149" if prio >= 5 else "#e3b341" if prio >= 4 else "#58a6ff"

            feed_html += f"""
            <div class='feed-entry {css_cls}'>
              <span style='font-size:10px;color:#8b949e'>{ts_str}</span>
              <span style='font-weight:600;font-size:11px'> {label}</span>
              <span style='color:{prio_color};font-size:10px'> P{prio}</span><br>
              <span style='color:#c9d1d9'>{mtype}</span>
              {f'<span style="color:#8b949e;font-size:10px"> · {zone[:20]}</span>' if zone else ''}
            </div>"""

        if not feed_html:
            feed_html = "<p style='color:#8b949e;font-size:12px'>No messages yet. Start agents to see live feed.</p>"

        st.markdown(f"<div style='height:480px;overflow-y:auto'>{feed_html}</div>", unsafe_allow_html=True)

    # Zone predictions table
    st.markdown("### Zone Risk Dashboard")
    if predictions:
        df_zones = []
        for zone in zones:
            pred = latest_preds.get(zone["id"])
            w    = next((x for x in weather if x.get("zone_id") == zone["id"]), {})
            df_zones.append({
                "Zone":         zone["name"],
                "Risk Score":   round(pred["risk_score"], 3) if pred else 0,
                "Severity":     pred["severity_level"].upper() if pred else "—",
                "Confidence":   f"{pred['confidence']:.0%}" if pred else "—",
                "Temp (°C)":    round(w.get("temperature", 0), 1),
                "Humidity (%)": round(w.get("humidity", 0)),
                "Rain 1h (mm)": round(w.get("precipitation_1h") or 0, 1),
                "Condition":    w.get("condition", "—").title(),
                "Elevation (m)": zone.get("elevation", "—"),
                "Drainage":     zone.get("drainage_capacity", "—").title(),
            })

        df = pd.DataFrame(df_zones).sort_values("Risk Score", ascending=False)

        def color_severity(val):
            colors = {"CRITICAL":"background-color:#3d1a1a;color:#f85149",
                      "HIGH":"background-color:#3d2d1a;color:#e3b341",
                      "MODERATE":"background-color:#1a2d3d;color:#58a6ff",
                      "LOW":"background-color:#1a3d2d;color:#3fb950",
                      "MINIMAL":"background-color:#1a1a1a;color:#8b949e"}
            return colors.get(val, "")

        st.dataframe(
            df.style.applymap(color_severity, subset=["Severity"]),
            use_container_width=True,
            hide_index=True,
        )

    # Risk score chart
    if latest_preds:
        st.markdown("### Risk Score Timeline")
        df_chart = pd.DataFrame([
            {"Zone": p["zone_name"], "Risk": p["risk_score"], "Severity": p["severity_level"]}
            for p in latest_preds.values()
        ])
        fig = px.bar(
            df_chart.sort_values("Risk", ascending=True),
            x="Risk", y="Zone", orientation="h",
            color="Severity",
            color_discrete_map={
                "critical": "#f85149", "high": "#e3b341",
                "moderate": "#58a6ff", "low": "#79c0ff", "minimal": "#3fb950"
            },
            template="plotly_dark",
        )
        fig.update_layout(
            height=280, margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="#e3b341", opacity=0.5,
                      annotation_text="Alert threshold")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRESS & DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Dispatch Route Map")
        if teams or zones:
            route_map = build_route_map(teams, zones)
            st_folium(route_map, width="100%", height=480)
        else:
            st.info("No active dispatch routes. Trigger a simulation to see routes.")

        # Active routes table
        if routes:
            st.markdown("### Active Dispatches")
            df_routes = pd.DataFrame([{
                "Incident":    r["incident_id"][:12],
                "Zone":        r["zone_name"],
                "Teams":       r["team_count"],
                "ETA (min)":   round(r["total_eta_minutes"] or 0),
                "Safety":      f"{r['route_safety_score']:.0%}" if r.get("route_safety_score") else "—",
                "Priority":    r["priority"],
                "Status":      r["status"].title(),
            } for r in routes])
            st.dataframe(df_routes, use_container_width=True, hide_index=True)

    with col_right:
        # Incident queue
        st.markdown("### Incident Queue")
        if allocations:
            for alloc in allocations[:8]:
                urgency = alloc.get("urgency", "MODERATE")
                color   = "#f85149" if "LIFE" in urgency else "#e3b341" if "URGENT" in urgency else "#58a6ff"
                people  = alloc.get("num_people_affected") or 0
                st.markdown(f"""
                <div style='border-left:3px solid {color};padding:8px 12px;
                     margin-bottom:8px;background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0'>
                  <span style='font-size:10px;color:#8b949e'>{alloc.get('incident_id','')[:14]}</span><br>
                  <span style='font-weight:600;font-size:13px'>{alloc.get('zone_name','Unknown')}</span><br>
                  <span style='color:{color};font-size:11px'>{urgency}</span>
                  <span style='color:#8b949e;font-size:11px'> · {people:,} people</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No active incidents.")

        st.markdown("### Resource Inventory")
        if inventory:
            for inv in inventory:
                rtype    = inv["resource_type"].replace("_", " ").title()
                total    = inv["total"]
                avail    = inv["available"]
                deployed = inv["deployed"]
                pct      = avail / total if total > 0 else 0
                bar_color = "#3fb950" if pct > 0.5 else "#e3b341" if pct > 0.2 else "#f85149"
                st.markdown(f"""
                <div style='margin-bottom:10px'>
                  <div style='display:flex;justify-content:space-between;font-size:12px'>
                    <span>{rtype}</span>
                    <span style='color:#8b949e'>{avail}/{total} available</span>
                  </div>
                  <div style='background:#21262d;border-radius:3px;height:6px;margin-top:4px'>
                    <div style='background:{bar_color};width:{pct*100:.0f}%;height:6px;border-radius:3px'></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        # Manual trigger
        st.markdown("### Manual Trigger")
        trigger_zone = st.selectbox("Zone", [z["name"] for z in zones] if zones else ["—"])
        trigger_risk = st.slider("Risk Score", 0.0, 1.0, 0.75, 0.05)
        trigger_sev  = st.select_slider("Severity", ["minimal","low","moderate","high","critical"], value="high")

        if st.button("🚨 Trigger Flood Alert → Agent 2", use_container_width=True):
            selected_zone = next((z for z in zones if z["name"] == trigger_zone), None)
            if selected_zone:
                payload = {
                    "zone_id":       selected_zone["id"],
                    "zone_name":     selected_zone["name"],
                    "risk_score":    trigger_risk,
                    "severity_level": trigger_sev,
                    "confidence":    0.85,
                    "risk_factors":  {},
                }
                try:
                    r = httpx.post(f"{AGENT2_URL}/trigger/flood-alert",
                                   json=payload, timeout=10.0)
                    if r.status_code == 200:
                        st.success(f"Alert triggered for {trigger_zone}!")
                    else:
                        st.error(f"Agent 2 returned {r.status_code}")
                except Exception as e:
                    st.error(f"Could not reach Agent 2: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIMULATION CONTROL
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Sylhet 2022 Flood Simulation")
    st.caption("Replay the worst flood in Bangladesh in 122 years — data sourced from OCHA, BDRCS, FFWC.")

    col_sim1, col_sim2 = st.columns([1, 1])

    with col_sim1:
        # Scenario cards
        scenarios = {
            "peak": {
                "name": "Peak Flood — June 18, 2022",
                "desc": "5 zones · 7.2M people affected · 84% of Sylhet submerged",
                "zones": 5, "people": "7.2M", "rainfall": "303.6 mm/day",
                "color": "#f85149",
            },
            "early": {
                "name": "Early Stage — June 9-10, 2022",
                "desc": "3 zones · 2M people · Rivers rising",
                "zones": 3, "people": "2M", "rainfall": "145 mm/day",
                "color": "#e3b341",
            },
            "single_zone": {
                "name": "Single Zone — Sylhet City",
                "desc": "Quick demo · Surma river 100cm above danger",
                "zones": 1, "people": "2M", "rainfall": "303.6 mm/day",
                "color": "#58a6ff",
            },
        }

        selected = st.radio(
            "Select scenario",
            list(scenarios.keys()),
            format_func=lambda k: scenarios[k]["name"],
        )
        sc = scenarios[selected]

        st.markdown(f"""
        <div style='background:#161b22;border:1px solid {sc["color"]}40;
             border-radius:10px;padding:16px;margin:12px 0'>
          <h4 style='color:{sc["color"]};margin:0 0 8px'>{sc["name"]}</h4>
          <p style='color:#8b949e;font-size:13px;margin:0 0 12px'>{sc["desc"]}</p>
          <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px'>
            <div style='text-align:center'>
              <div style='font-size:20px;font-weight:700;color:#f0f6fc'>{sc["zones"]}</div>
              <div style='font-size:11px;color:#8b949e'>Zones</div>
            </div>
            <div style='text-align:center'>
              <div style='font-size:20px;font-weight:700;color:#f0f6fc'>{sc["people"]}</div>
              <div style='font-size:11px;color:#8b949e'>People</div>
            </div>
            <div style='text-align:center'>
              <div style='font-size:20px;font-weight:700;color:#f0f6fc'>{sc["rainfall"]}</div>
              <div style='font-size:11px;color:#8b949e'>Rainfall</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        delay = st.slider("Delay between zone alerts (seconds)", 0.0, 5.0, 1.0, 0.5,
                          help="Stagger alerts for dramatic effect during presentation")

        if st.button("▶ Run Simulation", type="primary", use_container_width=True):
            with st.spinner(f"Running {sc['name']}..."):
                try:
                    r = httpx.post(
                        f"{AGENT1_URL}/simulate",
                        json={"scenario": selected, "delay_seconds": delay},
                        timeout=30.0,
                    )
                    if r.status_code == 200:
                        result = r.json()
                        st.success(f"✅ {result.get('message', 'Simulation complete!')}")
                        st.json(result)
                    else:
                        # Fallback: inject directly into Agent 2
                        st.warning("Agent 1 /simulate not available — injecting via Agent 2...")
                        with open("simulation_payload.json") as f:
                            payload = json.load(f)
                        r2 = httpx.post(
                            f"{AGENT2_URL}/trigger/distress-queue",
                            json=payload, timeout=15.0,
                        )
                        if r2.status_code == 200:
                            st.success("Injected via Agent 2 successfully!")
                        else:
                            st.error(f"Agent 2 returned {r2.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col_sim2:
        # Pipeline flow visualization
        st.markdown("### Agent Pipeline Status")

        agent_urls = {
            "Agent 1\nEnvironmental": AGENT1_URL,
            "Agent 2\nDistress":      AGENT2_URL,
            "Agent 3\nResource":      AGENT3_URL,
            "Agent 4\nDispatch":      AGENT4_URL,
        }

        for name, url in agent_urls.items():
            status = call_agent(url, "/health")
            is_ok  = status is not None and (
                status.get("status") in ("ok", "healthy")
            )
            dot_color  = "#3fb950" if is_ok else "#f85149"
            status_txt = "Online" if is_ok else "Offline"
            agent_short = name.split("\n")[0]
            agent_role  = name.split("\n")[1]

            msgs_for_agent = sum(
                1 for m in messages
                if agent_role.lower() in m.get("sender_agent", "").lower()
                or agent_role.lower() in m.get("receiver_agent", "").lower()
            )

            st.markdown(f"""
            <div style='background:#161b22;border:1px solid rgba(56,139,253,0.2);
                 border-radius:8px;padding:12px;margin-bottom:8px;
                 display:flex;align-items:center;justify-content:space-between'>
              <div>
                <span style='font-weight:600'>{agent_short}</span>
                <span style='color:#8b949e;font-size:12px'> — {agent_role}</span>
              </div>
              <div style='display:flex;align-items:center;gap:10px'>
                <span style='color:#8b949e;font-size:11px'>{msgs_for_agent} msgs</span>
                <span style='background:{dot_color}20;color:{dot_color};
                      padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600'>
                  {status_txt}
                </span>
              </div>
            </div>""", unsafe_allow_html=True)

        # Historical flood data reference
        st.markdown("### 2022 Sylhet Flood — Historical Reference")
        hist_data = {
            "Date": ["May 17", "Jun 9", "Jun 15", "Jun 18", "Jun 22", "Jul 3"],
            "Event": ["Initial flooding", "Flash flood begins", "Rivers burst banks",
                      "PEAK — record rainfall", "Water receding", "Relief operations"],
            "Affected (M)": [2.0, 2.5, 5.0, 7.2, 6.0, 4.0],
            "Submerged (%)": [20, 35, 55, 84, 70, 40],
        }
        fig_hist = px.area(
            pd.DataFrame(hist_data),
            x="Date", y="Submerged (%)",
            template="plotly_dark",
            color_discrete_sequence=["#58a6ff"],
            title="Sylhet division submerged area progression",
        )
        fig_hist.update_layout(
            height=220, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_hist.add_hline(y=50, line_dash="dash", line_color="#e3b341",
                           annotation_text="50% threshold")
        st.plotly_chart(fig_hist, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:12px;color:#8b949e;font-size:11px;
     border-top:1px solid rgba(56,139,253,0.1);margin-top:1rem'>
  Emergency Response System · Bangladesh Flood Intelligence ·
  Data: OpenWeatherMap · GloFAS · Google Earth Engine · OSRM
</div>
""", unsafe_allow_html=True)