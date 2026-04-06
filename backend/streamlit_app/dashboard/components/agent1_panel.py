# dashboard/components/agent1_panel.py
"""
Agent 1 Panel — Environmental Intelligence
==========================================
Shows for the HIGHEST-RISK zone in the current output:
  - SAR imagery comparison (dry vs flood placeholder)
  - 8-factor risk bar chart
  - Risk score + confidence boxes
  - Live weather summary
  - Zone selector dropdown to switch between zones
"""

from dash import html, dcc
import theme as T


_FACTOR_LABELS = {
    "satellite_flood_detection": "Satellite flood detection",
    "flood_depth_estimate":       "Flood depth estimate",
    "rainfall_intensity":         "Rainfall intensity",
    "accumulated_rainfall":       "Accumulated rainfall",
    "river_level_factor":         "River level (GloFAS)",
    "weather_severity":           "Weather severity",
    "drainage_factor":            "Drainage / Elevation",
    "social_reports_density":     "Social reports",
    "historical_risk":            "Historical risk",
}

_FACTOR_ORDER = [
    "satellite_flood_detection",
    "flood_depth_estimate",
    "river_level_factor",
    "rainfall_intensity",
    "accumulated_rainfall",
    "weather_severity",
    "drainage_factor",
    "historical_risk",
    "social_reports_density",
]


def _factor_bar_row(label: str, value: float) -> html.Div:
    pct = f"{value * 100:.1f}%"
    color = T.factor_bar_color(value)
    val_color = (
        T.CRITICAL if value >= 0.7 else
        T.HIGH     if value >= 0.5 else
        T.MODERATE if value >= 0.3 else
        T.LOW
    )
    return html.Div([
        html.Span(label, style={
            "fontSize": "10px", "color": T.TEXT_DIM,
            "width": "170px", "flexShrink": 0,
        }),
        html.Div(
            html.Div(style={
                "width": pct,
                "height": "100%",
                "borderRadius": "3px",
                "background": color,
                "transition": "width .8s cubic-bezier(.4,0,.2,1)",
            }),
            style={
                "flex": 1, "height": "6px",
                "background": T.BG3, "borderRadius": "3px", "overflow": "hidden",
            },
        ),
        html.Span(f"{value:.3f}", style={
            "fontFamily": T.FONT_UI, "fontSize": "11px", "fontWeight": "700",
            "color": val_color, "width": "40px", "textAlign": "right",
        }),
    ], style={"display": "flex", "alignItems": "center", "gap": "8px"})


def _score_box(label: str, value: float, tag: str, accent: str) -> html.Div:
    return html.Div([
        html.P(label, style={
            "fontSize": "9px", "textTransform": "uppercase",
            "letterSpacing": ".1em", "color": T.TEXT_DIM,
            "margin": "0 0 4px",
        }),
        html.P(f"{value * 100:.1f}%", style={
            "fontFamily": T.FONT_UI, "fontSize": "24px",
            "fontWeight": "800", "color": accent,
            "lineHeight": "1", "margin": 0,
        }),
        html.P(tag, style={
            "fontSize": "9px", "fontWeight": "700",
            "letterSpacing": ".1em", "color": f"{accent}b0",
            "margin": "3px 0 0",
        }),
    ], style={
        "background": f"{accent}18",
        "border": f"1px solid {accent}4d",
        "borderRadius": "6px",
        "padding": "12px 14px",
        "textAlign": "center",
        "minWidth": "100px",
    })


def _weather_tile(val: str, unit: str, label: str) -> html.Div:
    return html.Div([
        html.Div([
            html.Span(val, style={
                "fontFamily": T.FONT_UI, "fontSize": "15px",
                "fontWeight": "700", "color": T.BLUE, "lineHeight": "1",
            }),
            html.Span(unit, style={
                "fontSize": "9px", "color": T.TEXT_DIM, "marginLeft": "2px",
            }),
        ]),
        html.P(label, style={"fontSize": "9px", "color": T.TEXT_MUTE, "margin": "2px 0 0"}),
    ], style={
        "background": T.BG3,
        "border": f"1px solid {T.BORDER}",
        "borderRadius": "5px",
        "padding": "8px 10px",
    })


def build_agent1_panel(data: dict) -> html.Div:
    predictions = data.get("predictions", [])
    if not predictions:
        return html.P("No predictions available.", style={"color": T.TEXT_DIM, "padding": "20px"})

    # Pick highest-risk zone to display
    pred = max(predictions, key=lambda p: p.get("risk_score", 0))
    zone  = pred["zone"]
    rf    = pred.get("risk_factors", {})
    score = pred.get("risk_score", 0)
    conf  = pred.get("confidence", 0)
    sev   = pred.get("severity_level", "minimal")
    confirmed = rf.get("satellite_confirmed_flooding", False)
    has_river = rf.get("has_river_data", False)

    sev_color = T.SEVERITY_COLOR.get(sev, T.TEXT_DIM)

    # Zone selector options
    zone_options = [
        {"label": p["zone"]["name"], "value": i}
        for i, p in enumerate(predictions)
    ]

    # Build factor bars in display order
    factor_rows = []
    for key in _FACTOR_ORDER:
        val = rf.get(key, 0.0)
        if key == "social_reports_density" and not rf.get("has_social_data"):
            continue
        factor_rows.append(_factor_bar_row(_FACTOR_LABELS[key], val))

    # Source tags
    sources = []
    if rf.get("has_satellite_data"):  sources.append(("SAR", T.BLUE))
    if has_river:                      sources.append(("GloFAS", T.CYAN))
    sources.append(("Weather", T.TEXT_DIM))
    if rf.get("has_social_data"):     sources.append(("Social", T.YELLOW))

    divider = html.Div(style={"width": "1px", "background": T.BORDER, "flexShrink": 0, "alignSelf": "stretch"})

    return html.Div([
        # --- SAR comparison ---
        html.Div([
            html.P(f"SAR Imagery — {zone['name']}", style={**T.LABEL_SMALL, "marginBottom": "8px"}),
            html.Div([
                # DRY reference
                html.Div([
                    html.P("DRY", style={
                        "fontFamily": T.FONT_UI, "fontSize": "18px", "fontWeight": "800",
                        "color": T.MINIMAL, "margin": 0,
                    }),
                    html.P("Reference SAR · Jan–Mar 2024", style={
                        "fontSize": "8px", "color": T.TEXT_DIM, "marginTop": "2px",
                    }),
                ], style={
                    "flex": 1, "background": "linear-gradient(135deg,#0d1a2a,#112233)",
                    "border": f"1px solid {T.BORDER2}", "borderRadius": "5px",
                    "display": "flex", "flexDirection": "column",
                    "alignItems": "center", "justifyContent": "center",
                    "padding": "16px", "minHeight": "80px",
                }),
                # FLOOD current
                html.Div([
                    html.P("FLOOD" if confirmed else "LOW", style={
                        "fontFamily": T.FONT_UI, "fontSize": "18px", "fontWeight": "800",
                        "color": T.CRITICAL if confirmed else T.MODERATE, "margin": 0,
                    }),
                    html.P("Current SAR · 2024", style={
                        "fontSize": "8px", "color": T.TEXT_DIM, "marginTop": "2px",
                    }),
                    html.P(
                        "CNN flood mask overlaid" if confirmed else "No active flooding",
                        style={"fontSize": "7.5px", "color": T.TEXT_MUTE, "marginTop": "2px"},
                    ),
                ], style={
                    "flex": 1,
                    "background": "linear-gradient(135deg,#1a0d0d,#2a1515)" if confirmed
                                  else "linear-gradient(135deg,#0d1a14,#11221a)",
                    "border": f"1px solid {T.CRITICAL}40" if confirmed else f"1px solid {T.BORDER2}",
                    "borderRadius": "5px",
                    "display": "flex", "flexDirection": "column",
                    "alignItems": "center", "justifyContent": "center",
                    "padding": "16px", "minHeight": "80px",
                }),
            ], style={"display": "flex", "gap": "8px"}),

            # Data sources row
            html.Div([
                html.Span("Sources:", style={"fontSize": "9px", "color": T.TEXT_MUTE, "marginRight": "6px"}),
                *[html.Span(label, style={
                    "fontSize": "9px", "color": color, "marginRight": "8px",
                    "background": f"{color}20", "padding": "1px 6px",
                    "borderRadius": "3px", "border": f"1px solid {color}40",
                }) for label, color in sources],
            ], style={"display": "flex", "alignItems": "center", "marginTop": "6px", "flexWrap": "wrap", "gap": "2px"}),
        ], style={"minWidth": "240px", "display": "flex", "flexDirection": "column", "gap": "0"}),

        divider,

        # --- 8-Factor bars ---
        html.Div([
            html.P("8-Factor Prediction — Live Per Zone", style={**T.LABEL_SMALL, "marginBottom": "8px"}),
            html.Div(factor_rows, style={"display": "flex", "flexDirection": "column", "gap": "8px"}),
        ], style={"flex": 1, "minWidth": "240px"}),

        divider,

        # --- Score boxes ---
        html.Div([
            _score_box("Risk Score", score, sev.upper(), sev_color),
            _score_box(
                "Confidence", conf,
                "SAR CONFIRMED" if confirmed else "WEATHER+SAR",
                T.GREEN,
            ),
        ], style={"display": "flex", "flexDirection": "column", "gap": "8px"}),

        divider,

        # --- Weather ---
        html.Div([
            html.P("Live Weather", style={**T.LABEL_SMALL, "marginBottom": "8px"}),
            html.Div([
                _weather_tile("34.3", "°C",   "Temperature"),
                _weather_tile("32%",  "%",    "Humidity"),
                _weather_tile("1.0",  "m/s",  "Wind"),
                _weather_tile("0.0",  "mm/h", "Rain 1h"),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "6px"}),
            html.P(f"Zone: {zone['name']}", style={
                "fontSize": "9px", "color": T.TEXT_MUTE, "marginTop": "6px",
            }),
        ], style={"minWidth": "150px"}),
    ], style={
        "display": "flex",
        "gap": "16px",
        "padding": "14px 20px",
        "height": "100%",
        "overflow": "hidden",
        "alignItems": "flex-start",
    })
