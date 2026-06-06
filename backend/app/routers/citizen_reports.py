# backend/app/routers/citizen_reports.py
"""
GET  /api/citizen-reports        — list recent citizen reports (+ validation if available)
POST /api/citizen-reports        — submit a new citizen report
GET  /api/citizen-reports/{id}   — single report
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.services.db import get_db

logger = logging.getLogger("dashboard.routers.citizen_reports")

router = APIRouter(prefix="/api/citizen-reports", tags=["citizen-reports"])


# ── Schema ─────────────────────────────────────────────────────────────────────

class CitizenReportIn(BaseModel):
    name: str
    phone: Optional[str] = ""
    category: str = "Flooding"
    message: str
    latitude: float
    longitude: float


# ── GET /api/citizen-reports ──────────────────────────────────────────────────

@router.get("")
async def list_reports(
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None),
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Returns citizen reports joined with Agent 1 validation data (if available).
    The LEFT JOIN means reports without validation are still returned,
    with flood_risk_level / risk_score / validation_notes = null.
    """
    rows = await conn.fetch("""
        SELECT
            r.id::text,
            r.name,
            r.phone,
            r.category,
            r.message,
            r.latitude,
            r.longitude,
            r.status,
            r.created_at,
            v.flood_risk_level,
            v.risk_score,
            v.claim_validity,
            v.validation_notes
        FROM citizen_reports r
        LEFT JOIN citizen_report_validations v ON v.report_id = r.id
        WHERE ($1::text IS NULL OR r.status = $1)
        ORDER BY r.created_at DESC
        LIMIT $2
    """, status, limit)

    reports: List[Dict[str, Any]] = []
    for r in rows:
        reports.append({
            "id":               r["id"],
            "name":             r["name"],
            "phone":            r["phone"],
            "category":         r["category"],
            "message":          r["message"],
            "latitude":         float(r["latitude"] or 0),
            "longitude":        float(r["longitude"] or 0),
            "status":           r["status"],
            "created_at":       r["created_at"].isoformat() if r["created_at"] else None,
            "flood_risk_level": r["flood_risk_level"],
            "risk_score":       r["risk_score"],
            "claim_validity":   r["claim_validity"],
            "validation_notes": r["validation_notes"],
        })

    return {"reports": reports, "count": len(reports)}


# ── POST /api/citizen-reports ─────────────────────────────────────────────────

@router.post("", status_code=201)
async def create_report(
    body: CitizenReportIn,
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    """
    Saves a new citizen report submitted from the React Citizen Reporter page.
    """
    report_id = uuid4()

    try:
        await conn.execute("""
            INSERT INTO citizen_reports
                (id, name, phone, latitude, longitude, category, message, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'pending')
        """,
            report_id,
            body.name.strip(),
            (body.phone or "").strip(),
            body.latitude,
            body.longitude,
            body.category,
            body.message.strip(),
        )
    except asyncpg.UndefinedTableError:
        raise HTTPException(
            status_code=503,
            detail="citizen_reports table does not exist yet. Run the database migration first.",
        )
    except Exception as e:
        logger.exception("Failed to insert citizen report: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("Citizen report saved: %s (%s) at %.4f,%.4f", body.name, body.category, body.latitude, body.longitude)

    return {
        "id":      str(report_id),
        "status":  "pending",
        "message": "Report submitted successfully.",
    }


# ── GET /api/citizen-reports/{id} ─────────────────────────────────────────────

@router.get("/{report_id}")
async def get_report(
    report_id: str,
    conn: asyncpg.Connection = Depends(get_db),
) -> Dict[str, Any]:
    row = await conn.fetchrow("""
        SELECT
            r.id::text, r.name, r.phone, r.category, r.message,
            r.latitude, r.longitude, r.status, r.created_at,
            v.flood_risk_level, v.risk_score, v.claim_validity, v.validation_notes
        FROM citizen_reports r
        LEFT JOIN citizen_report_validations v ON v.report_id = r.id
        WHERE r.id = $1::uuid
    """, report_id)

    if not row:
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "id":               row["id"],
        "name":             row["name"],
        "phone":            row["phone"],
        "category":         row["category"],
        "message":          row["message"],
        "latitude":         float(row["latitude"] or 0),
        "longitude":        float(row["longitude"] or 0),
        "status":           row["status"],
        "created_at":       row["created_at"].isoformat() if row["created_at"] else None,
        "flood_risk_level": row["flood_risk_level"],
        "risk_score":       row["risk_score"],
        "claim_validity":   row["claim_validity"],
        "validation_notes": row["validation_notes"],
    }
