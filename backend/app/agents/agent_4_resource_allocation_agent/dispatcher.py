from distance import haversine
from db import get_available_teams, get_supplies, save_dispatch
import json


def allocate_resources(report):
    # ---------------------------
    # 1️⃣ Normalize all report fields
    # ---------------------------
    try:
        hazards = report.hazards
        hazards = hazards if isinstance(hazards, str) else str(hazards or "")
        hazards = hazards.lower()
    except:
        hazards = ""

    try:
        urgency = report.urgency
        urgency = urgency if isinstance(urgency, str) else str(urgency or "")
        urgency = urgency.lower()
    except:
        urgency = ""

    try:
        needs_raw = report.needs
        if isinstance(needs_raw, str):
            needs_obj = json.loads(needs_raw)
        else:
            needs_obj = needs_raw or {}
        needed_items = needs_obj.get("items", [])
    except:
        needed_items = []

    # ---------------------------
    # 2️⃣ Incident location (placeholder)
    # ---------------------------
    incident_lat = 23.8103
    incident_lng = 90.4125

    # ---------------------------
    # 3️⃣ Load available teams + supplies
    # ---------------------------
    teams = get_available_teams()
    supplies_inventory = get_supplies()

    if not teams:
        return None, None, "No available teams"

    # ---------------------------
    # 4️⃣ Score teams by distance, hazard match, urgency
    # ---------------------------
    scored = []

    hazards = str(report.hazards or "").lower()
    urgency = str(report.urgency or "").lower()

    scored = []

    for t in teams:
        # normalize team_type
        team_type = str(t.team_type or "").lower()

        distance = haversine(incident_lat, incident_lng, t.latitude, t.longitude)
        score = 100 - distance

        if "flood" in hazards and team_type == "rescue":
            score += 20

        if urgency == "high":
            score += 15

        scored.append((t, score))


    best_team = max(scored, key=lambda x: x[1])[0]

    # ---------------------------
    # 5️⃣ Allocate supplies
    # ---------------------------
    allocated = {}

    for item in needed_items:
        for s in supplies_inventory:
            # These must now be clean Python values
            supply_name = str(getattr(s, "item_name", "") or "").lower()

            # Safe integer fallback
            quantity_raw = getattr(s, "quantity", 0)

            try:
                quantity = int(quantity_raw)
            except:
                quantity = 0

            if supply_name == item.lower() and quantity > 0:
                allocated[item] = 1
                break


    # ---------------------------
    # 6️⃣ ETA estimation (km / 0.5 km/min ≈ 30 km/h)
    # ---------------------------
    eta_km = haversine(best_team.latitude, best_team.longitude, incident_lat, incident_lng)
    eta_minutes = int(eta_km / 0.5)

    reasoning = f"Selected team {best_team.team_name} due to proximity and urgency."

    return best_team, allocated, reasoning
