import streamlit as st
import asyncpg
import asyncio
import pandas as pd
import base64
import os

DATABASE_URL = os.getenv(
    "ENV_DB_URL",
    "postgresql://postgres:postgres@localhost:5432/disaster_db"
)

# ------------------------------
# DB Queries
# ------------------------------

async def fetch_reports():
    query = """
        SELECT id, name, phone, latitude, longitude, category, message, status, created_at
        FROM citizen_reports
        ORDER BY created_at DESC;
    """
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch(query)
    await conn.close()
    return [dict(r) for r in rows]


async def fetch_files(report_id):
    query = """
        SELECT id, file_name, file_data, file_type, uploaded_at
        FROM citizen_report_files
        WHERE report_id = $1;
    """
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch(query, report_id)
    await conn.close()
    return [dict(r) for r in rows]


# Sync wrappers
def get_reports():
    return asyncio.run(fetch_reports())

def get_files(report_id):
    return asyncio.run(fetch_files(report_id))


# ------------------------------
# Streamlit UI
# ------------------------------

st.title("📂 All Citizen Reports")
st.subheader("View details and attached documents")

reports = get_reports()

if len(reports) == 0:
    st.info("No reports found.")
    st.stop()

for rep in reports:
    with st.container(border=True):
        st.markdown(f"### 🧑 {rep['name']}")
        st.write(f"📞 {rep['phone']}")
        st.write(f"🕒 {rep['created_at']}")
        st.write(f"📌 **Category:** {rep['category']}")
        st.write(f"📝 **Status:** `{rep['status']}`")
        st.write(f"🌍 **Location:** {rep['latitude']}, {rep['longitude']}")
        st.write(f"🗒️ **Message:** {rep['message']}")

        # # load files for this report
        # files = get_files(rep["id"])

        # if len(files) > 0:
        #     st.markdown("#### 📎 Attached Documents")

        #     for f in files:
        #         file_name = f["file_name"]
        #         file_data = f["file_data"]
        #         file_type = f["file_type"]

        #         # display image
        #         if file_type.startswith("image"):
        #             st.image(file_data, caption=file_name, use_container_width=True)

        #         # display non-image files for download
        #         b64 = base64.b64encode(file_data).decode()
        #         href = (
        #             f'<a href="data:{file_type};base64,{b64}" '
        #             f'download="{file_name}">📥 Download {file_name}</a>'
        #         )
        #         st.markdown(href, unsafe_allow_html=True)

        # else:
        #     st.info("No documents uploaded for this report.")

        st.markdown("---")
