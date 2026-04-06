# import streamlit as st
# from db import get_weather
# import pandas as pd
# from datetime import datetime, timedelta

# st.title("🗺️ Dynamic Weather Map View")

# # ------------------- Sidebar Filters -------------------
# st.sidebar.header("Filters")

# record_limit = st.sidebar.slider("Number of Records", 50, 1000, 200)

# hours_back = st.sidebar.slider("Show Data from Last (Hours)", 1, 720, 24)

# temp_range = st.sidebar.slider("Temperature Range (°C)", -10, 50, (-10, 50))
# humidity_range = st.sidebar.slider("Humidity Range (%)", 0, 100, (0, 100))

# refresh = st.sidebar.button("🔄 Refresh Data")

# # ------------------- Fetch Data -------------------
# df = get_weather(record_limit)

# # Remove rows without coordinates
# df = df.dropna(subset=["latitude", "longitude"])

# # Rename for streamlit map
# df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

# # Convert timestamp if needed
# if isinstance(df["timestamp"].iloc[0], str):
#     df["timestamp"] = pd.to_datetime(df["timestamp"])

# # ------------------- Dynamic Filters -------------------
# time_threshold = datetime.utcnow() - timedelta(hours=hours_back)
# df = df[df["timestamp"] >= time_threshold]

# df = df[(df["temperature"].between(*temp_range))]
# df = df[(df["humidity"].between(*humidity_range))]

# # ------------------- Show Summary -------------------
# st.subheader("📊 Filtered Data Summary")
# st.write(f"**Total points shown:** {len(df)}")

# if not df.empty:
#     st.map(df)
# else:
#     st.warning("No data matches the filter conditions.")

# # ------------------- Show Table -------------------
# with st.expander("📄 View Table Data"):
#     st.dataframe(df)

# # ------------------- Timestamp -------------------
# st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
##########


# ---------------------------
# import streamlit as st
# import pydeck as pdk
# from db import get_weather

# st.title("🌡️ Weather Heatmap + Scatter Visualization")

# df = get_weather(300)
# df = df.dropna(subset=["latitude", "longitude"])

# df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

# heatmap = pdk.Layer(
#     "HeatmapLayer",
#     data=df,
#     get_position=['lon', 'lat'],
#     get_weight="temperature",
#     radiusPixels=60,
# )

# scatter = pdk.Layer(
#     "ScatterplotLayer",
#     data=df,
#     get_position=['lon', 'lat'],
#     get_color='[200, 30, 0, 160]',
#     get_radius=80,
# )

# view_state = pdk.ViewState(
#     latitude=df["lat"].mean(),
#     longitude=df["lon"].mean(),
#     zoom=5,
#     pitch=40,
# )

# r = pdk.Deck(layers=[heatmap, scatter], initial_view_state=view_state)

# st.pydeck_chart(r)


# ---------------------------

import streamlit as st
from db import get_weather
import time

st.title("⏱️ Live Updating Weather Map")

interval = st.slider("Refresh interval (seconds)", 5, 60, 10)

placeholder = st.empty()

while True:
    df = get_weather(300)
    df = df.dropna(subset=["latitude", "longitude"])
    df.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

    with placeholder.container():
        st.map(df)
        st.caption(f"Updated at: {time.strftime('%H:%M:%S')}")

    time.sleep(interval)

