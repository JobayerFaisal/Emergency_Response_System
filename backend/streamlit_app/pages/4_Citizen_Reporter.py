import streamlit as st
import folium
from streamlit_folium import st_folium
import asyncpg
import asyncio
import os

DATABASE_URL = os.getenv(
    "ENV_DB_URL",
    # "postgresql://postgres:postgres@localhost:5432/disaster_db"
    "postgresql://postgres:postgres@db:5432/disaster_db"
)

# Save report to DB
async def save_report(data):
    query = """
        INSERT INTO citizen_reports (
            id, name, phone, latitude, longitude, category, message, status
        )
        VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, 'pending');
    """
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute(
        query,
        data["name"], data["phone"], data["lat"], data["lon"],
        data["category"], data["message"]
    )
    await conn.close()


def commit_report(data):
    asyncio.run(save_report(data))


# ------------------------------------
# STREAMLIT UI
# ------------------------------------
st.title("📣 Citizen Reporter")
st.write("Report conditions, request help, or send updates.")

# ---------------------------- Map Picker ----------------------------
st.subheader("📍 Select Your Location")

m = folium.Map(location=[23.8103, 90.4125], zoom_start=12)

# Enable click handler to pick lat/lon
clicked_point = st_folium(m, height=400)

lat = None
lon = None

if clicked_point and clicked_point.get("last_clicked"):
    lat = clicked_point["last_clicked"]["lat"]
    lon = clicked_point["last_clicked"]["lng"]
    st.success(f"📌 Location Selected: {lat}, {lon}")
else:
    st.info("Click on the map to pick your location.")

# ---------------------------- Form Fields ----------------------------
st.subheader("📝 Report Details")

name = st.text_input("Your Name")
phone = st.text_input("Phone Number")
category = st.selectbox("Type of Emergency", [
    "Medical Emergency",
    "Flooding",
    "Fire",
    "Building Collapse",
    "Road Blockage",
    "Rescue Needed",
    "Other"
])
message = st.text_area("Describe the situation")


submitted = st.button("Submit Report", type="primary")

if submitted:
    if not lat or not lon:
        st.error("Please choose a location from the map.")
    elif not name or not phone or not message:
        st.error("Please fill all required fields.")
    else:
        try:
            report = {
                "name": name,
                "phone": phone,
                "lat": lat,
                "lon": lon,
                "category": category,
                "message": message,
            }
            commit_report(report)
            st.success("✅ Report submitted successfully!")
        except Exception as e:
            st.error(f"Database error: {e}")
