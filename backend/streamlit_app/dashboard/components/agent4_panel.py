# dashboard/components/agent4_panel.py
"""
Agent 4 Panel — Dispatch Optimization
========================================
Team assignments table + route summary.
Replace DEMO_TEAMS with a call to Agent 4's API endpoint.
"""

from dash import html
import theme as T

_DEMO_TEAMS = [
    {"team": "Boat Unit A",  "destination": "Sylhet",    "eta": "25 min",  "status": "enroute"},
    {"team": "Boat Unit B",  "destination": "Sylhet",    "eta": "30 min",  "status": "enroute"},
    {"team": "Medical T1",   "destination": "Sylhet",    "eta": "40 min",  "status": "preparing"},
    {"team": "Boat Unit C",  "destination": "Sirajganj", "eta": "On site", "status": "active"},
    {"team": "Rescue T2",    "destination": "Sunamganj", "eta": "15 min",  "status": "enroute"},
    {"team": "Medical T2",   "destination": "Netrokona", "eta": "55 min",  "status": "standby"},
]

_STATUS_STYLES = {
    "enroute":   (T.HIGH,     "En Route"),
    "preparing": (T.BLUE,     "Preparing"),
    "active":    (T.GREEN,    "Active"),
    "standby":   (T.TEXT_DIM, "Standby"),
}


def _status_badge(status: str) -> html.Span:
    color, label = _STATUS_STYLES.get(status, (T.TEXT_DIM, status.title()))
    return html.Span(label, style={
        "display": "inline-block",
        "padding": "2px 8px", "borderRadius": "3px",
        "fontSize": "9px", "fontWeight": "700",
        "letterSpacing": ".06em", "textTransform": "uppercase",
        "color": color,
        "backgroundColor": f"{color}20",
        "border": f"1px solid {color}50",
    })


def build_agent4_panel(data: dict) -> html.Div:
    # Count by status
    counts = {"enroute": 0, "active": 0, "preparing": 0, "standby": 0}
    for t in _DEMO_TEAMS:
        counts[t["status"]] = counts.get(t["status"], 0) + 1

    rows = []
    for t in _DEMO_TEAMS:
        eta_color = T.GREEN if t["eta"] == "On site" else (
            T.YELLOW if int(t["eta"].split()[0]) > 30 else T.GREEN
        ) if t["eta"] != "On site" and t["eta"] != "—" else T.TEXT_DIM
        rows.append(html.Tr([
            html.Td(t["team"],        style={**T.TABLE_CELL, "fontWeight": "500"}),
            html.Td(t["destination"], style=T.TABLE_CELL),
            html.Td(t["eta"],         style={**T.TABLE_CELL, "color": eta_color}),
            html.Td(_status_badge(t["status"]), style=T.TABLE_CELL),
        ]))

    h = T.TABLE_HEADER
    return html.Div([
        html.P(
            "Dispatch Optimization — Team assignments + route computation",
            style={**T.LABEL_SMALL, "marginBottom": "10px"},
        ),

        html.Div([
            # Table
            html.Div([
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Team",        style=h),
                        html.Th("Destination", style=h),
                        html.Th("ETA",         style=h),
                        html.Th("Status",      style=h),
                    ])),
                    html.Tbody(rows),
                ], style={"width": "100%", "borderCollapse": "collapse"}),
            ], style={"flex": 1, "overflowY": "auto"}),

            html.Div(style={"width": "1px", "background": T.BORDER, "alignSelf": "stretch"}),

            # Summary + route note
            html.Div([
                html.P("Deployment Summary", style={**T.LABEL_SMALL, "marginBottom": "10px"}),
                *[html.Div([
                    html.Div(style={
                        "width": "8px", "height": "8px", "borderRadius": "50%",
                        "background": _STATUS_STYLES[k][0], "flexShrink": 0,
                    }),
                    html.Span(f"{v} {_STATUS_STYLES[k][1]}", style={
                        "fontSize": "10px", "color": T.TEXT,
                    }),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"})
                for k, v in counts.items() if v > 0],

                html.Div(style={"height": "1px", "background": T.BORDER, "margin": "10px 0"}),

                html.Div([
                    html.P("Route Map", style={**T.LABEL_SMALL, "marginBottom": "4px"}),
                    html.P(
                        "Toggle "Team Routes" on the map above to show active route polylines "
                        "(same Leaflet layer). Colored lines: team position → destination.",
                        style={"fontSize": "9px", "color": T.TEXT_MUTE, "lineHeight": "1.6"},
                    ),
                ], style={
                    "background": T.BG3,
                    "border": f"1px dashed {T.BORDER2}",
                    "borderRadius": "5px",
                    "padding": "10px 12px",
                }),
            ], style={"minWidth": "180px", "maxWidth": "220px"}),
        ], style={
            "display": "flex", "gap": "20px",
            "flex": 1, "overflow": "hidden",
        }),
    ], style={
        "padding": "14px 20px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "0",
        "height": "100%",
        "overflow": "hidden",
    })
