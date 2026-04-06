# dashboard/components/agent2_panel.py
"""
Agent 2 Panel — Distress Intelligence
======================================
Trilingual NLP distress reports table with urgency badges
and satellite cross-reference status.
In production: populated from Agent 2's own database table.
Demo data used when Agent 2 API is not yet connected.
"""

from dash import html
import theme as T

# Demo distress data — replace with Agent 2 API call
_DEMO_REPORTS = [
    {
        "message":  '"Mirpur e pani utheche, bari dube jacche"',
        "language": "Banglish",
        "location": "Mirpur, Dhaka",
        "urgency":  "LIFE-THREAT",
        "people":   "~15",
        "xref":     "CONFIRMED",
    },
    {
        "message":  '"Water entering homes near Sylhet station"',
        "language": "English",
        "location": "Sylhet City",
        "urgency":  "URGENT",
        "people":   "~8",
        "xref":     "CONFIRMED",
    },
    {
        "message":  '"Road flooded near Kawran Bazar"',
        "language": "English",
        "location": "Kawran Bazar",
        "urgency":  "MODERATE",
        "people":   "?",
        "xref":     "UNVERIFIED",
    },
    {
        "message":  '"বন্যায় রাস্তা ডুবে গেছে, সাহায্য করুন"',
        "language": "Bengali",
        "location": "Sunamganj Sadar",
        "urgency":  "LIFE-THREAT",
        "people":   "~30",
        "xref":     "CONFIRMED",
    },
    {
        "message":  '"Flood water rising fast in Netrokona"',
        "language": "English",
        "location": "Netrokona Sadar",
        "urgency":  "URGENT",
        "people":   "~12",
        "xref":     "CONFIRMED",
    },
]

_URGENCY_COLORS = {
    "LIFE-THREAT": T.CRITICAL,
    "URGENT":      T.HIGH,
    "MODERATE":    T.MODERATE,
}


def _urgency_badge(urgency: str) -> html.Span:
    color = _URGENCY_COLORS.get(urgency, T.TEXT_DIM)
    return html.Span(urgency, style={
        "display": "inline-block",
        "padding": "2px 8px", "borderRadius": "3px",
        "fontSize": "9px", "fontWeight": "700",
        "letterSpacing": ".08em", "textTransform": "uppercase",
        "color": color,
        "backgroundColor": f"{color}20",
        "border": f"1px solid {color}50",
    })


def _xref_badge(status: str) -> html.Span:
    if status == "CONFIRMED":
        return html.Span("✓ Confirmed", style={
            "display": "inline-block",
            "padding": "2px 8px", "borderRadius": "3px",
            "fontSize": "9px", "fontWeight": "700",
            "color": T.GREEN,
            "backgroundColor": f"{T.GREEN}18",
            "border": f"1px solid {T.GREEN}40",
        })
    return html.Span("Unverified", style={
        "display": "inline-block",
        "padding": "2px 8px", "borderRadius": "3px",
        "fontSize": "9px", "fontWeight": "700",
        "color": T.TEXT_DIM,
        "backgroundColor": T.BG3,
        "border": f"1px solid {T.BORDER2}",
    })


def build_agent2_panel(data: dict) -> html.Div:
    # Build table rows
    rows = []
    for r in _DEMO_REPORTS:
        rows.append(html.Tr([
            html.Td(html.I(r["message"]), style={**T.TABLE_CELL, "color": T.TEXT, "maxWidth": "280px"}),
            html.Td(r["language"], style={**T.TABLE_CELL, "color": T.TEXT_DIM, "fontSize": "9px"}),
            html.Td(r["location"], style={T.TABLE_CELL["padding"]: "8px 12px", **T.TABLE_CELL}),
            html.Td(_urgency_badge(r["urgency"]), style={T.TABLE_CELL["padding"]: "8px 12px", **T.TABLE_CELL}),
            html.Td(r["people"], style={**T.TABLE_CELL, "textAlign": "center"}),
            html.Td(_xref_badge(r["xref"]), style={T.TABLE_CELL["padding"]: "8px 12px", **T.TABLE_CELL}),
        ]))

    header_style = {
        **T.TABLE_HEADER,
        "padding": "0 12px 8px",
        "borderBottom": f"1px solid {T.BORDER2}",
    }

    return html.Div([
        # Subtitle
        html.P(
            "Trilingual NLP (Bengali / English / Banglish) · Cross-referenced with Agent 1 satellite data",
            style={**T.LABEL_SMALL, "marginBottom": "10px"},
        ),

        # Table
        html.Div(
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Message",    style=header_style),
                    html.Th("Language",   style=header_style),
                    html.Th("Location",   style=header_style),
                    html.Th("Urgency",    style=header_style),
                    html.Th("People",     style=header_style),
                    html.Th("Cross-ref",  style=header_style),
                ])),
                html.Tbody(rows),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
            style={"overflowX": "auto", "flex": 1},
        ),

        # Footnote
        html.P(
            'Cross-ref = Agent 2 asked Agent 1 via Redis: "Is this zone actually flooded?" '
            "CONFIRMED = satellite agrees. Proves inter-agent coordination.",
            style={
                "fontSize": "9px", "color": T.TEXT_MUTE, "lineHeight": "1.6",
                "paddingTop": "8px", "borderTop": f"1px solid {T.BORDER}",
            },
        ),
    ], style={
        "padding": "14px 20px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "10px",
        "height": "100%",
        "overflow": "hidden",
    })
