import streamlit as st
from db import get_weather

st.title("🗺️ Weather Map View")

df = get_weather(200)

# Only keep rows with coordinates
df = df.dropna(subset=["latitude", "longitude"])

# Streamlit expects columns: lat, lon
df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

st.map(df)
