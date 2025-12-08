import streamlit as st
import pandas as pd
from db import get_weather_full

st.set_page_config(page_title="Weather Monitoring Dashboard", layout="wide")
st.title("🌤️ Environmental Weather Data Overview")

# -------------------------------------------------------------
# Filters
# -------------------------------------------------------------
limit = st.slider("Number of records to load:", 20, 1000, 200)

df = get_weather_full(limit)

if df is None or df.empty:
    st.warning("No weather data found in database yet.")
    st.stop()

# -------------------------------------------------------------
# Formatting
# -------------------------------------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Order columns nicely
preferred_columns = [
    "timestamp",
    "zone_name",
    "latitude",
    "longitude",
    "temperature",
    "feels_like",
    "humidity",
    "pressure",
    "wind_speed",
    "precip_1h",
    "precip_3h",
    "precip_24h",
    "weather_condition",
]

# Add additional columns if available
cols = [c for c in preferred_columns if c in df.columns] + [
    c for c in df.columns if c not in preferred_columns
]

df = df[cols]

# -------------------------------------------------------------
# Display Table
# -------------------------------------------------------------
st.subheader("📊 Latest Collected Weather Observations")
st.dataframe(df, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# Summary stats
# -------------------------------------------------------------
st.subheader("📈 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Temperature", f"{df['temperature'].mean():.2f} °C")
col2.metric("Avg Humidity", f"{df['humidity'].mean():.1f}%")
col3.metric("Max Rain (1h)", f"{df['precip_1h'].max()}")
col4.metric("Data Rows", len(df))

# -------------------------------------------------------------
# Optional: Map visualization
# -------------------------------------------------------------
st.subheader("🗺️ Weather Stations Map")

map_df = df[["latitude", "longitude"]].dropna()
st.map(map_df)
