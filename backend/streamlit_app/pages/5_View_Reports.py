import streamlit as st
import asyncpg
import asyncio
import os
import sys

# -----------------------------
# FIX MODULE IMPORT ERROR
# -----------------------------
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


from utils.validation_checker import process_pending_validations

DATABASE_URL = os.getenv(
    "ENV_DB_URL",
    "postgresql://postgres:postgres@localhost:5432/disaster_db"
)

# Auto-refresh every 10 seconds
st.set_page_config(page_title="Citizen Reports", initial_sidebar_state="expanded")
st_autorefresh = st.rerun  # flexible approach


# --------------------------
# Run auto-validation first
# --------------------------
async def run_validation():
    count = await process_pending_validations()
    return count

validation_count = asyncio.run(run_validation())

if validation_count > 0:
    st.success(f"🔄 {validation_count} new reports validated automatically.")
else:
    st.info("✔ No pending validations.")


# --------------------------
# Fetch reports
# --------------------------
async def fetch_reports():
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch("SELECT * FROM citizen_reports ORDER BY created_at DESC")
    await conn.close()
    return [dict(r) for r in rows]

reports = asyncio.run(fetch_reports())


# --------------------------
# Display each report
# --------------------------
st.title("📋 All Citizen Reports (Real-Time Validation Enabled)")
st.caption("This page auto-validates reports as soon as they arrive.")

for rep in reports:
    with st.container(border=True):
        st.subheader(f"🧑 Name: {rep['name']}")
        st.write(f"📞 Phone: {rep['phone']}")
        st.write(f"📌 Category: {rep['category']}")
        st.write(f"📝 Message: {rep['message']}")
        st.write(f"🌍 Location: {rep['latitude']}, {rep['longitude']}")
        st.write(f"🕒 Reported At: {rep['created_at']}")

        # Fetch validation
        conn = asyncio.run(asyncpg.connect(DATABASE_URL))
        validation = asyncio.run(conn.fetchrow(
            """
            SELECT flood_risk_level, risk_score, claim_validity, validation_notes 
            FROM citizen_report_validations 
            WHERE report_id = $1
            """, rep["id"]
        ))
        asyncio.run(conn.close())

        if validation:
            st.markdown("### 🛡️ Validation")
            st.write(f"Flood Risk Level: **{validation['flood_risk_level']}**")
            st.write(f"Risk Score: **{validation['risk_score']}**")
            st.write(f"Claim Valid: **{validation['claim_validity']}**")
            st.write(f"Notes: {validation['validation_notes']}")
        else:
            st.warning("⏳ Waiting for validation...")

        st.markdown("---")
