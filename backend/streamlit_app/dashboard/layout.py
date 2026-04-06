# dashboard/layout.py
"""
Layout
======
Builds the full Dash component tree once at startup.
All dynamic content is populated by callbacks — layout only
defines the skeleton (IDs, containers, static chrome).

Structure:
  ┌─────────────────────────────────────────────┐
  │  KPI BAR (always visible)                   │
  ├───────────────────────────┬─────────────────┤
  │  MAP (dash-leaflet)       │  AGENT FEED     │
  │                           │  (Redis events) │
  ├───────────────────────────┴─────────────────┤
  │  AGENT TABS                                  │
  │  [Agent1:Env] [Agent2:Distress]              │
  │  [Agent3:Resource] [Agent4:Dispatch]         │
  └─────────────────────────────────────────────┘
"""

from dash import html, dcc
import dash_leaflet as dl

import theme as T


# ── Google Fonts injection ────────────────────────────────────────────────────

_FONT_LINK = html.Link(
    rel="stylesheet",
    href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@400;600;700;800&display=swap",
)

# ── Interval timers ───────────────────────────────────────────────────────────

_INTERVALS = html.Div([
    # Main data poll — matches Agent 1 monitoring interval (default 180 s)
    # You can reduce to 30 s for snappier updates without hammering the API
    dcc.Interval(id="interval-data",   interval=30_000, n_intervals=0),
    # Fast tick for the live clock in the KPI bar
    dcc.Interval(id="interval-clock",  interval=1_000,  n_intervals=0),
    # Feed animation tick (simulates streaming when Redis is unavailable)
    dcc.Interval(id="interval-feed",   interval=8_000,  n_intervals=0),
])

# ── Data store ────────────────────────────────────────────────────────────────
# Single source of truth: callbacks write here, panels read from here.
# This avoids multiple parallel HTTP requests on every update.

_STORE = dcc.Store(id="store-agent-output", storage_type="memory")


# ─────────────────────────────────────────────────────────────────────────────
# KPI BAR
# ─────────────────────────────────────────────────────────────────────────────

def _kpi_item(item_id: str, label: str, accent: str = T.TEXT) -> html.Div:
    return html.Div([
        html.P(id=item_id, children="—", style={**T.KPI_NUM, "color": accent}),
        html.P(label, style={**T.LABEL_SMALL, "marginTop": "2px"}),
    ], style={"minWidth": "90px"})


def _kpi_divider() -> html.Div:
    return html.Div(style={
        "width": "1px", "height": "36px",
        "background": T.BORDER2, "margin": "0 20px",
    })


def _kpi_bar() -> html.Div:
    return html.Div([
        # Brand
        html.Div([
            html.P("Emergency Response System", style={
                "fontFamily": T.FONT_UI, "fontSize": "10px", "fontWeight": "700",
                "letterSpacing": ".12em", "color": T.TEXT_DIM,
                "textTransform": "uppercase", "margin": 0,
            }),
            html.P("Bangladesh · Multi-Agent Platform", style={
                "fontFamily": T.FONT_MONO, "fontSize": "9px",
                "color": T.TEXT_MUTE, "letterSpacing": ".06em", "margin": 0,
            }),
        ], style={"marginRight": "24px"}),

        _kpi_divider(),
        _kpi_item("kpi-zones",   "Zones Monitored"),
        _kpi_divider(),
        _kpi_item("kpi-alerts",  "Active Alerts",     T.CRITICAL),
        _kpi_divider(),
        _kpi_item("kpi-reports", "Distress Reports"),
        _kpi_divider(),
        _kpi_item("kpi-teams",   "Teams Deployed",    T.YELLOW),
        _kpi_divider(),
        _kpi_item("kpi-people",  "People Affected",   T.BLUE),

        # Spacer
        html.Div(style={"flex": "1"}),

        # Clock
        html.P(id="kpi-clock", style={
            "fontFamily": T.FONT_MONO, "fontSize": "9px",
            "color": T.TEXT_MUTE, "marginRight": "16px",
        }),

        # Live badge
        html.Div([
            html.Div(className="pulse-dot", style={
                "width": "7px", "height": "7px", "borderRadius": "50%",
                "background": T.GREEN, "boxShadow": f"0 0 8px {T.GREEN}",
            }),
            html.Span("SYSTEM LIVE", style={
                "fontFamily": T.FONT_UI, "fontSize": "11px",
                "fontWeight": "700", "color": T.GREEN, "letterSpacing": ".1em",
            }),
        ], style={
            "display": "flex", "alignItems": "center", "gap": "8px",
            "background": f"{T.GREEN}14",
            "border": f"1px solid {T.GREEN}40",
            "padding": "6px 14px", "borderRadius": "4px",
        }),
    ], style={
        "background": T.BG2,
        "borderBottom": f"1px solid {T.BORDER}",
        "padding": "0 20px",
        "height": "60px",
        "display": "flex",
        "alignItems": "center",
        "gap": "0",
        "flexShrink": "0",
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAP PANEL  (dash-leaflet)
# ─────────────────────────────────────────────────────────────────────────────

def _map_panel() -> html.Div:
    base_map = dl.TileLayer(
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        opacity=0.55,
        id="tile-layer",
    )

    return html.Div([
        # Header
        html.Div([
            html.P("Zone Status Map", style=T.PANEL_TITLE),
            # Toggle buttons — callbacks will add/remove overlay layers
            html.Div([
                html.Button("Satellite Overlay", id="btn-satellite", n_clicks=0,
                            className="map-toggle active"),
                html.Button("Team Routes",       id="btn-routes",    n_clicks=0,
                            className="map-toggle"),
                html.Button("Distress Pins",     id="btn-distress",  n_clicks=1,
                            className="map-toggle active"),
            ], style={"display": "flex", "gap": "6px", "marginLeft": "auto"}),
        ], style={
            **T.PANEL_HEADER,
            "position": "absolute", "top": 0, "left": 0, "right": 0,
            "zIndex": 1000, "backdropFilter": "blur(8px)",
        }),

        # Leaflet map
        dl.Map(
            id="main-map",
            center=[24.3, 90.5],
            zoom=7,
            style={"width": "100%", "height": "100%", "marginTop": "38px",
                   "background": "#080c14"},
            children=[
                base_map,
                dl.LayerGroup(id="layer-zones"),     # risk circles (always on)
                dl.LayerGroup(id="layer-distress"),   # distress pins
                dl.LayerGroup(id="layer-routes"),     # team route polylines
            ],
        ),

        # Legend
        html.Div([
            *[html.Div([
                html.Div(style={
                    "width": "8px", "height": "8px", "borderRadius": "50%",
                    "background": color, "flexShrink": 0,
                }),
                html.Span(label, style={
                    "fontSize": "9px", "color": T.TEXT_DIM,
                    "textTransform": "uppercase", "letterSpacing": ".08em",
                }),
            ], style={"display": "flex", "alignItems": "center", "gap": "7px"})
            for label, color in [
                ("Critical", T.CRITICAL), ("High", T.HIGH),
                ("Moderate", T.MODERATE), ("Minimal", T.MINIMAL),
                ("Team Route", T.BLUE),   ("Distress", T.YELLOW),
            ]],
        ], style={
            "position": "absolute", "bottom": "16px", "left": "16px",
            "background": "rgba(10,13,19,.88)", "backdropFilter": "blur(8px)",
            "border": f"1px solid {T.BORDER2}", "padding": "10px 14px",
            "borderRadius": "6px", "zIndex": 1000,
            "display": "flex", "flexDirection": "column", "gap": "5px",
        }),
    ], style={
        "position": "relative", "overflow": "hidden",
        "borderRight": f"1px solid {T.BORDER}",
        "borderBottom": f"1px solid {T.BORDER}",
    })


# ─────────────────────────────────────────────────────────────────────────────
# AGENT FEED PANEL
# ─────────────────────────────────────────────────────────────────────────────

def _feed_panel() -> html.Div:
    return html.Div([
        html.Div([
            html.P("Live Agent Feed", style=T.PANEL_TITLE),
            html.Span("Redis pub/sub · logged to PostgreSQL", style={
                **T.LABEL_SMALL, "marginLeft": "auto",
            }),
        ], style={**T.PANEL_HEADER, "flexShrink": 0}),

        html.Div(id="feed-list", style={
            "flex": 1, "overflowY": "auto", "padding": "8px",
            "display": "flex", "flexDirection": "column", "gap": "6px",
        }),

        html.P(
            "Every message = one Redis pub/sub event → logged to PostgreSQL",
            style={
                "padding": "8px 14px", "borderTop": f"1px solid {T.BORDER}",
                "fontSize": "9px", "color": T.TEXT_MUTE, "flexShrink": 0,
            },
        ),
    ], style={
        "background": T.BG2,
        "borderBottom": f"1px solid {T.BORDER}",
        "display": "flex", "flexDirection": "column", "overflow": "hidden",
    })


# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM TABS
# ─────────────────────────────────────────────────────────────────────────────

def _tab(label: str, agent_num: int, tab_id: str) -> dcc.Tab:
    dot_color = [T.AGENT1, T.AGENT2, T.AGENT3, T.AGENT4][agent_num - 1]
    return dcc.Tab(
        label=label,
        value=tab_id,
        style={
            "fontFamily": T.FONT_MONO, "fontSize": "10px",
            "color": T.TEXT_DIM, "backgroundColor": T.BG,
            "border": f"1px solid {T.BORDER}",
            "borderBottom": "none", "borderRadius": "4px 4px 0 0",
            "padding": "7px 16px",
        },
        selected_style={
            "fontFamily": T.FONT_MONO, "fontSize": "10px",
            "color": T.TEXT, "backgroundColor": T.BG2,
            "border": f"1px solid {T.BORDER}",
            "borderBottom": f"2px solid {dot_color}",
            "borderRadius": "4px 4px 0 0",
            "padding": "7px 16px",
        },
    )


def _bottom_tabs() -> html.Div:
    return html.Div([
        dcc.Tabs(
            id="agent-tabs",
            value="tab-agent1",
            style={"background": T.BG, "borderBottom": f"1px solid {T.BORDER}"},
            children=[
                _tab("Agent 1: Environmental", 1, "tab-agent1"),
                _tab("Agent 2: Distress",      2, "tab-agent2"),
                _tab("Agent 3: Resource",       3, "tab-agent3"),
                _tab("Agent 4: Dispatch",       4, "tab-agent4"),
            ],
        ),
        html.Div(id="tab-content", style={
            "flex": 1, "overflow": "hidden",
            "background": T.BG2,
        }),
    ], style={
        "gridColumn": "1 / -1",
        "background": T.BG2,
        "borderTop": f"1px solid {T.BORDER}",
        "display": "flex", "flexDirection": "column",
        "height": "280px", "flexShrink": 0,
    })


# ─────────────────────────────────────────────────────────────────────────────
# ROOT LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def build_layout() -> html.Div:
    return html.Div([
        _FONT_LINK,
        _STORE,
        _INTERVALS,

        # KPI bar
        _kpi_bar(),

        # Main grid: map + feed + tabs
        html.Div([
            _map_panel(),
            _feed_panel(),
            _bottom_tabs(),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "1fr 380px",
            "gridTemplateRows": "1fr 280px",
            "flex": 1,
            "overflow": "hidden",
            "minHeight": 0,
        }),

    ], style={
        "background": T.BG,
        "color": T.TEXT,
        "fontFamily": T.FONT_MONO,
        "fontSize": "12px",
        "height": "100vh",
        "display": "flex",
        "flexDirection": "column",
        "overflow": "hidden",
    })
