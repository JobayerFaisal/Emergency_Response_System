import streamlit as st
import requests
import pandas as pd

st.title("📦 Extracted Emergency Incidents")

API_URL = "http://disaster_backend:8000/api/v1/emergency-reports/"

st.info("Fetching structured emergency reports extracted from responder conversations...")

try:
    response = requests.get(API_URL)
    response.raise_for_status()
    data = response.json()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if not data:
    st.warning("No incidents found yet.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(data)

# Expand nested JSON fields for visibility
df["people"] = df["people"].apply(lambda x: str(x))
df["needs"] = df["needs"].apply(lambda x: str(x))
df["hazards"] = df["hazards"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")

st.dataframe(df, use_container_width=True)

# Show individual items
st.subheader("Detailed View")

selected = st.selectbox("Select Incident ID", df["id"])

incident = df[df["id"] == selected].iloc[0]

st.write("### Incident Details")
st.json(incident.to_dict())
