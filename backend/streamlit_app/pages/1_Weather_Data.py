import streamlit as st
from db import get_weather

st.title("📋 Weather Data Table")

limit = st.slider("Number of rows:", 20, 500, 100)

df = get_weather(limit)
st.dataframe(df, use_container_width=True)
