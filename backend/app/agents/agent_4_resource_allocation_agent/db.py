from backend.app.core.db import SessionLocal
from models import EmergencyReport, TeamResource, SupplyInventory, DispatchLog


def get_pending_reports():
    db = SessionLocal()
    reports = db.query(EmergencyReport).order_by(EmergencyReport.timestamp.desc()).all()
    db.close()
    return reports


def get_available_teams():
    db = SessionLocal()
    teams = db.query(TeamResource).filter(TeamResource.status == "available").all()
    db.close()
    return teams


def get_supplies():
    db = SessionLocal()
    supplies = db.query(SupplyInventory).all()
    db.close()
    return supplies


def save_dispatch(report_id, team_id, supplies, eta, reasoning):
    db = SessionLocal()
    log = DispatchLog(
        report_id=report_id,
        team_id=team_id,
        supplies=supplies,
        eta_minutes=eta,
        status="dispatched",
        reasoning=reasoning
    )
    db.add(log)

    # Update team status
    db.query(TeamResource).filter(
        TeamResource.id == team_id
    ).update({"status": "busy"})

    db.commit()
    db.close()
