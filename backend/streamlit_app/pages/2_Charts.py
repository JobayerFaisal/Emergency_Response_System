import streamlit as st
from db import get_weather

st.title("📊 Weather Charts")

limit = st.slider("Records to visualize:", 20, 300, 100)

df = get_weather(limit)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Temperature Over Time")
    st.line_chart(df.set_index("timestamp")["temperature"])

with col2:
    st.subheader("Humidity Over Time")
    st.line_chart(df.set_index("timestamp")["humidity"])

st.subheader("Pressure")
st.line_chart(df.set_index("timestamp")["pressure"])
