# dashboard/theme.py
"""
Design tokens and inline-style helpers.
Single source of truth for colors, fonts, spacing.
Matches the dark terminal aesthetic from the HTML prototype.
"""

# ── Colors ───────────────────────────────────────────────────────────────────

BG       = "#0a0d13"
BG2      = "#10141d"
BG3      = "#161b27"
BORDER   = "#1e2535"
BORDER2  = "#252d40"
TEXT     = "#c8d4e8"
TEXT_DIM = "#5a6a88"
TEXT_MUTE= "#3a4558"

# Severity
CRITICAL = "#ff4444"
HIGH     = "#ff8c00"
MODERATE = "#f5c518"
LOW      = "#4caf50"
MINIMAL  = "#4db8ff"

# General accents
GREEN  = "#3ddc84"
RED    = "#ff4d4d"
YELLOW = "#ffd166"
BLUE   = "#4db8ff"
CYAN   = "#00e5cc"

# Agent identity colors
AGENT1 = "#e84040"
AGENT2 = "#e89040"
AGENT3 = "#4080e8"
AGENT4 = "#e8d8a0"

# ── Severity helpers ──────────────────────────────────────────────────────────

SEVERITY_COLOR = {
    "critical": CRITICAL,
    "high":     HIGH,
    "moderate": MODERATE,
    "low":      LOW,
    "minimal":  MINIMAL,
}

SEVERITY_LABEL = {
    "critical": "CRITICAL",
    "high":     "HIGH",
    "moderate": "MODERATE",
    "low":      "LOW",
    "minimal":  "MINIMAL",
}

# Map severity → Leaflet circle radius (meters)
SEVERITY_RADIUS = {
    "critical": 22000,
    "high":     16000,
    "moderate": 12000,
    "low":      10000,
    "minimal":  9000,
}

# ── Font stack ────────────────────────────────────────────────────────────────

FONT_MONO = "'JetBrains Mono', 'Fira Code', monospace"
FONT_UI   = "'Syne', 'DM Sans', sans-serif"

# ── Shared style dicts ────────────────────────────────────────────────────────

PANEL = {
    "backgroundColor": BG2,
    "border": f"1px solid {BORDER}",
    "borderRadius": "6px",
    "overflow": "hidden",
}

PANEL_HEADER = {
    "backgroundColor": BG,
    "borderBottom": f"1px solid {BORDER}",
    "padding": "8px 14px",
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
}

PANEL_TITLE = {
    "fontFamily": FONT_UI,
    "fontSize": "11px",
    "fontWeight": "700",
    "color": TEXT,
    "letterSpacing": ".06em",
    "textTransform": "uppercase",
    "margin": 0,
}

LABEL_SMALL = {
    "fontFamily": FONT_MONO,
    "fontSize": "9px",
    "color": TEXT_DIM,
    "textTransform": "uppercase",
    "letterSpacing": ".1em",
}

KPI_NUM = {
    "fontFamily": FONT_UI,
    "fontSize": "22px",
    "fontWeight": "800",
    "lineHeight": "1",
    "letterSpacing": "-.02em",
    "margin": 0,
}

TABLE_CELL = {
    "fontFamily": FONT_MONO,
    "fontSize": "11px",
    "color": TEXT,
    "padding": "8px 12px",
    "borderBottom": f"1px solid {BORDER}",
    "verticalAlign": "middle",
}

TABLE_HEADER = {
    **TABLE_CELL,
    "fontSize": "9px",
    "color": TEXT_MUTE,
    "fontWeight": "700",
    "textTransform": "uppercase",
    "letterSpacing": ".1em",
    "borderBottom": f"1px solid {BORDER2}",
    "paddingBottom": "8px",
}


def severity_badge_style(severity: str) -> dict:
    """Inline style for a severity pill badge."""
    color = SEVERITY_COLOR.get(severity, TEXT_DIM)
    return {
        "display": "inline-block",
        "padding": "2px 8px",
        "borderRadius": "3px",
        "fontSize": "9px",
        "fontWeight": "700",
        "letterSpacing": ".08em",
        "textTransform": "uppercase",
        "color": color,
        "backgroundColor": f"{color}20",
        "border": f"1px solid {color}50",
    }


def factor_bar_color(value: float) -> str:
    """Return a gradient CSS string based on 0–1 factor value."""
    if value >= 0.7:
        return f"linear-gradient(90deg, {CRITICAL}, #ff6b6b)"
    elif value >= 0.5:
        return f"linear-gradient(90deg, {HIGH}, #ffaa44)"
    elif value >= 0.3:
        return f"linear-gradient(90deg, {MODERATE}, #ffe066)"
    else:
        return f"linear-gradient(90deg, {LOW}, #6fcf97)"


def resource_bar_color(ratio: float) -> str:
    """Color for resource inventory bars (ratio = current/max)."""
    if ratio <= 0.35:
        return f"linear-gradient(90deg, {CRITICAL}, #ff6b6b)"
    elif ratio <= 0.6:
        return f"linear-gradient(90deg, {HIGH}, #ffaa44)"
    else:
        return f"linear-gradient(90deg, #2db868, {GREEN})"
