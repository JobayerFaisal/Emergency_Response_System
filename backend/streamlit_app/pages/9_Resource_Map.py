import streamlit as st
import pandas as pd
import pydeck as pdk

from utils.db_resource_map import get_incidents, get_teams

from pathlib import Path
import os

ICON_PATH = Path(__file__).parent.parent / "assets" / "icons"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #f8f9fa;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# # Serve icons as static assets
# st.static_directory = str(ICON_PATH)



st.set_page_config(page_title="🚨 Resource Allocation Map", layout="wide")

st.title("🚨 Real-Time Resource & Incident Map")

# Fetch data
incidents = get_incidents()
teams = get_teams()

# Icon assignments (future use)
ICON_TYPE = {
    "rescue": "🚁",
    "medical": "🚑",
    "fire": "🔥",
    "delivery": "📦",
}

# View settings (MISSING BEFORE — NOW FIXED)
view_state = pdk.ViewState(
    latitude=23.8103,
    longitude=90.4125,
    zoom=10,
    pitch=45
)

# TEAM LAYER
ICON_URL = "http://localhost:8501/icons/"  # Streamlit static files (we'll map this next)

# TEAM ICONS
team_layer = pdk.Layer(
    "IconLayer",
    data=teams,
    pickable=True,
    get_icon="icon_data",
    get_size=4,
    size_scale=15,
    get_position=["lon", "lat"],
    get_tooltip="_tooltip"
)

incident_layer = pdk.Layer(
    "IconLayer",
    data=incidents,
    pickable=True,
    get_icon="icon_data",
    get_size=5,
    size_scale=20,
    get_position=["lon", "lat"],
    get_tooltip="_tooltip"
)





deck = pdk.Deck(
    initial_view_state=view_state,
    layers=[team_layer, incident_layer]
)

st.pydeck_chart(deck)



