# dashboard/components/agent3_panel.py
"""
Agent 3 Panel — Resource Management
=====================================
Inventory bars and transaction log.
Replace DEMO_RESOURCES with a call to Agent 3's API endpoint.
"""

from dash import html
import theme as T

_DEMO_RESOURCES = [
    {"name": "Rescue Boats",       "current": 12, "total": 20},
    {"name": "Medical Kits",       "current": 45, "total": 60},
    {"name": "Rescue Teams",       "current":  3, "total":  8},
    {"name": "Emergency Food Packs","current": 120, "total": 200},
    {"name": "Life Vests",         "current": 85,  "total": 120},
]

_DEMO_TRANSACTIONS = [
    ("10:44", "2 boats allocated to Sylhet (Agent 4 dispatch)"),
    ("10:44", "1 medical team allocated to Sylhet"),
    ("10:30", "5 medical kits restocked (manual)"),
    ("10:15", "1 boat allocated to Sirajganj"),
    ("09:55", "3 rescue teams deployed to Sunamganj"),
    ("09:30", "20 life vests restocked (manual)"),
]


def _resource_bar(name: str, current: int, total: int) -> html.Div:
    ratio = current / total if total > 0 else 0
    pct   = f"{ratio * 100:.0f}%"
    color = T.resource_bar_color(ratio)
    count_color = (
        T.CRITICAL if ratio <= 0.35 else
        T.HIGH     if ratio <= 0.60 else
        T.GREEN
    )
    return html.Div([
        html.Div([
            html.Span(name, style={"fontSize": "11px", "fontWeight": "500", "color": T.TEXT}),
            html.Span(f"{current} / {total}", style={
                "fontFamily": T.FONT_UI, "fontSize": "12px",
                "fontWeight": "700", "color": count_color,
            }),
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
        html.Div(
            html.Div(style={
                "width": pct, "height": "100%",
                "borderRadius": "4px", "background": color,
            }),
            style={
                "height": "7px", "background": T.BG3,
                "borderRadius": "4px", "overflow": "hidden",
            },
        ),
    ], style={"display": "flex", "flexDirection": "column"})


def build_agent3_panel(data: dict) -> html.Div:
    resource_bars = [_resource_bar(r["name"], r["current"], r["total"])
                     for r in _DEMO_RESOURCES]

    tx_rows = [
        html.Div([
            html.Span(t, style={"color": T.TEXT_MUTE, "marginRight": "8px", "fontWeight": "500"}),
            html.Span(msg, style={"color": T.TEXT_DIM}),
        ], style={
            "fontSize": "9.5px", "lineHeight": "1.5",
            "padding": "5px 0", "borderBottom": f"1px solid {T.BORDER}",
        })
        for t, msg in _DEMO_TRANSACTIONS
    ]

    return html.Div([
        html.P(
            "Resource Management — Auto-deduct on dispatch, manual restock",
            style={**T.LABEL_SMALL, "marginBottom": "10px"},
        ),
        html.Div([
            # Inventory bars
            html.Div([
                html.P("Inventory", style={**T.LABEL_SMALL, "marginBottom": "8px"}),
                html.Div(resource_bars, style={
                    "display": "flex", "flexDirection": "column", "gap": "10px",
                }),
            ], style={"flex": 1}),

            html.Div(style={"width": "1px", "background": T.BORDER, "alignSelf": "stretch"}),

            # Transaction log
            html.Div([
                html.P("Transaction Log", style={**T.LABEL_SMALL, "marginBottom": "8px"}),
                html.Div(tx_rows, style={"overflowY": "auto"}),
            ], style={"flex": 1}),
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
