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


from app.agents.c_environmental.environment_agent import validate_request

DATABASE_URL = os.getenv("ENV_DB_URL")

async def process_pending_validations():
    conn = await asyncpg.connect(DATABASE_URL)

    # Find reports with NO validation yet
    query = """
        SELECT *
        FROM citizen_reports
        WHERE id NOT IN (
            SELECT report_id FROM citizen_report_validations
        );
    """

    pending = await conn.fetch(query)

    if not pending:
        await conn.close()
        return 0  # no new reports

    for report in pending:
        validation_output = await validate_request(dict(report))

        # Insert into validation table
        await conn.execute("""
            INSERT INTO citizen_report_validations
            (report_id, flood_risk_level, risk_score, claim_validity, validation_notes)
            VALUES ($1, $2, $3, $4, $5)
        """,
        report["id"],
        validation_output["flood_risk_level"],
        validation_output["risk_score"],
        validation_output["claim_validity"],
        validation_output["validation_notes"]
        )

    await conn.close()
    return len(pending)
