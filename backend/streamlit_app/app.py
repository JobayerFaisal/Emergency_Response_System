# backend/streamlit_app/app.py

import streamlit as st

st.set_page_config(page_title="Weather Dashboard", layout="wide")


# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🌦️ Dashboard Navigation")
st.sidebar.markdown("Use the page list below to navigate.")
st.sidebar.markdown("---")

st.sidebar.info("This dashboard visualizes weather data from your PostgreSQL database.")

# -------------------------
# Main Page
# -------------------------
st.title("🌍 Weather Monitoring Dashboard")

st.markdown("""
Welcome to your weather monitoring system.

### Available Pages:
- **Weather Data Table**
- **Charts**
- **Map View**

Use the **sidebar or page navigator** to explore.
""")
