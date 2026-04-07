# backend/src/agents/agent_3_resource/seed_data.py

"""
src/agents/agent_3_resource/seed_data.py
Pre-populate resource_units with realistic Bangladesh disaster response assets.
Skips seeding if data already exists (idempotent).
"""

import logging
from .models import ResourceType
from shared.geo_utils import postgis_point_wkt

logger = logging.getLogger("agent3.seed")

# (resource_type, name, capacity, lat, lon)
SEED_RESOURCES = [

    # ─── RESCUE BOATS ──────────────────────────────────────────────────────
    # BIWTA stations, river ghats
    (ResourceType.RESCUE_BOAT, "Boat Mirpur-1",       15, 23.8041,  90.3654),
    (ResourceType.RESCUE_BOAT, "Boat Mirpur-2",       15, 23.8041,  90.3654),
    (ResourceType.RESCUE_BOAT, "Boat Mirpur-3",       12, 23.8170,  90.3660),  # Near Mirpur-12
    (ResourceType.RESCUE_BOAT, "Boat Sadarghat-1",    20, 23.7104,  90.4074),
    (ResourceType.RESCUE_BOAT, "Boat Sadarghat-2",    20, 23.7104,  90.4074),
    (ResourceType.RESCUE_BOAT, "Boat Demra-1",        15, 23.7133,  90.4700),
    (ResourceType.RESCUE_BOAT, "Boat Keraniganj-1",   12, 23.6922,  90.3795),
    (ResourceType.RESCUE_BOAT, "Boat Sylhet-1",       12, 24.8949,  91.8687),
    (ResourceType.RESCUE_BOAT, "Boat Sylhet-2",       12, 24.8949,  91.8687),
    (ResourceType.RESCUE_BOAT, "Boat Sunamganj-1",    10, 25.0715,  91.3953),
    (ResourceType.RESCUE_BOAT, "Boat Sunamganj-2",    10, 25.0715,  91.3953),
    (ResourceType.RESCUE_BOAT, "Boat Sirajganj-1",    18, 24.4534,  89.7007),
    (ResourceType.RESCUE_BOAT, "Boat Sirajganj-2",    18, 24.4490,  89.6950),
    (ResourceType.RESCUE_BOAT, "Boat Jamalpur-1",     14, 24.9000,  89.9378),
    (ResourceType.RESCUE_BOAT, "Boat Gaibandha-1",    10, 25.3286,  89.5284),
    (ResourceType.RESCUE_BOAT, "Boat Kurigram-1",     10, 25.8074,  89.6357),
    (ResourceType.RESCUE_BOAT, "Boat Barishal-1",     16, 22.7010,  90.3535),
    (ResourceType.RESCUE_BOAT, "Boat Narayanganj-1",  15, 23.6238,  90.4986),
    (ResourceType.RESCUE_BOAT, "Boat Chittagong-1",   18, 22.3384,  91.8317),
    (ResourceType.RESCUE_BOAT, "Boat Noakhali-1",     12, 22.8696,  91.0996),

    # ─── MEDICAL TEAMS ─────────────────────────────────────────────────────
    # Based at major hospitals
    (ResourceType.MEDICAL_TEAM, "MedTeam Dhaka-1",       50, 23.7229,  90.3952),  # Dhaka Medical
    (ResourceType.MEDICAL_TEAM, "MedTeam Dhaka-2",       40, 23.7170,  90.3981),  # Mitford
    (ResourceType.MEDICAL_TEAM, "MedTeam Mirpur-1",      35, 23.8058,  90.3689),  # Mirpur-10
    (ResourceType.MEDICAL_TEAM, "MedTeam Sylhet-1",      30, 24.8998,  91.8710),  # Osmani Hospital
    (ResourceType.MEDICAL_TEAM, "MedTeam Sylhet-2",      25, 24.8998,  91.8710),
    (ResourceType.MEDICAL_TEAM, "MedTeam Sirajganj-1",   25, 24.4600,  89.7100),
    (ResourceType.MEDICAL_TEAM, "MedTeam Barishal-1",    30, 22.7010,  90.3535),
    (ResourceType.MEDICAL_TEAM, "MedTeam Chittagong-1",  40, 22.3384,  91.8317),

    # ─── MEDICAL KITS ──────────────────────────────────────────────────────
    # Stored at district relief depots
    (ResourceType.MEDICAL_KIT, "Kit-Depot-Dhaka-1",      100, 23.8103,  90.4125),
    (ResourceType.MEDICAL_KIT, "Kit-Depot-Dhaka-2",      100, 23.7781,  90.4070),  # Mohakhali
    (ResourceType.MEDICAL_KIT, "Kit-Depot-Mirpur",        80, 23.8058,  90.3689),
    (ResourceType.MEDICAL_KIT, "Kit-Depot-Sylhet",        60, 24.8949,  91.8687),
    (ResourceType.MEDICAL_KIT, "Kit-Depot-Sirajganj",     50, 24.4534,  89.7007),
    (ResourceType.MEDICAL_KIT, "Kit-Depot-Sunamganj",     40, 25.0715,  91.3953),
    (ResourceType.MEDICAL_KIT, "Kit-Depot-Barishal",      60, 22.7010,  90.3535),
    (ResourceType.MEDICAL_KIT, "Kit-Depot-Chittagong",    80, 22.3384,  91.8317),
    (ResourceType.MEDICAL_KIT, "Kit-Depot-Narayanganj",   50, 23.6238,  90.4986),
    (ResourceType.MEDICAL_KIT, "Kit-Depot-Jamalpur",      40, 24.9000,  89.9378),
    (ResourceType.MEDICAL_KIT, "Kit-Depot-Kurigram",      30, 25.8074,  89.6357),

    # ─── FOOD SUPPLY ───────────────────────────────────────────────────────
    (ResourceType.FOOD_SUPPLY, "Food-Mohakhali",         500, 23.7781,  90.4070),
    (ResourceType.FOOD_SUPPLY, "Food-Mirpur-Depot",      400, 23.8100,  90.3680),
    (ResourceType.FOOD_SUPPLY, "Food-Sylhet-Depot",      300, 24.8949,  91.8687),
    (ResourceType.FOOD_SUPPLY, "Food-Sirajganj-Depot",   350, 24.4534,  89.7007),
    (ResourceType.FOOD_SUPPLY, "Food-Sunamganj-Depot",   200, 25.0715,  91.3953),
    (ResourceType.FOOD_SUPPLY, "Food-Barishal-Depot",    300, 22.7010,  90.3535),
    (ResourceType.FOOD_SUPPLY, "Food-Chittagong-Depot",  400, 22.3384,  91.8317),
    (ResourceType.FOOD_SUPPLY, "Food-Kurigram-Depot",    200, 25.8074,  89.6357),

    # ─── WATER SUPPLY ──────────────────────────────────────────────────────
    (ResourceType.WATER_SUPPLY, "Water-Mohakhali",        1000, 23.7781,  90.4070),
    (ResourceType.WATER_SUPPLY, "Water-Mirpur-Depot",      800, 23.8100,  90.3680),
    (ResourceType.WATER_SUPPLY, "Water-Sylhet-Depot",      600, 24.8949,  91.8687),
    (ResourceType.WATER_SUPPLY, "Water-Sirajganj-Depot",   700, 24.4534,  89.7007),
    (ResourceType.WATER_SUPPLY, "Water-Sunamganj-Depot",   400, 25.0715,  91.3953),
    (ResourceType.WATER_SUPPLY, "Water-Barishal-Depot",    600, 22.7010,  90.3535),
    (ResourceType.WATER_SUPPLY, "Water-Chittagong-Depot",  800, 22.3384,  91.8317),
    (ResourceType.WATER_SUPPLY, "Water-Kurigram-Depot",    400, 25.8074,  89.6357),
]


async def seed_initial_resources(inv_manager) -> None:
    """Insert seed data only if resource_units table is empty."""
    existing = await inv_manager.get_all_units()
    if existing:
        logger.info("Seed skipped — %d resources already in DB", len(existing))
        return

    logger.info("Seeding %d initial resources…", len(SEED_RESOURCES))
    async with inv_manager.pool.acquire() as conn:
        for rtype, name, capacity, lat, lon in SEED_RESOURCES:
            wkt = postgis_point_wkt(lat, lon)
            await conn.execute("""
                INSERT INTO resource_units
                    (resource_type, name, status, capacity,
                     current_location, base_location)
                VALUES ($1, $2, 'available', $3,
                        $4::geography, $4::geography)
                ON CONFLICT DO NOTHING
            """, rtype.value, name, capacity, wkt)

    logger.info("Seeding complete.")
