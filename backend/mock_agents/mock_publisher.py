"""
mock_agents/mock_publisher.py
═══════════════════════════════════════════════════════════════════
BYPASS TOOL — Run this instead of Agent 1 & 2 during development.

Publishes realistic flood_alert + distress_queue messages to Redis
so that Agent 3 and Agent 4 can be tested completely standalone.

When your teammate merges Agent 1 & 2, this file is no longer needed.
Agent 3 & 4 will receive the EXACT same message format from the real agents.

Usage:
    python -m mock_agents.mock_publisher --scenario mirpur_flood
    python -m mock_agents.mock_publisher --scenario sylhet_flood
    python -m mock_agents.mock_publisher --scenario multi_zone
    python -m mock_agents.mock_publisher --scenario all

═══════════════════════════════════════════════════════════════════
"""

import asyncio
import json
import argparse
import sys
import os
from uuid import uuid4
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from redis import asyncio as aioredis
from shared.message_protocol import AgentMessage

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS
# Each scenario mimics what Agent 2 would publish to distress_queue
# after receiving Agent 1's flood_alert and running trilingual NLP.
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {

    # ── Scenario 1: Mirpur flood ──────────────────────────────────────────
    # Corner case: vague "Mirpur" location resolved by location_resolver
    "mirpur_flood": {
        "description": "Flooding in Mirpur — tests location resolution corner case",
        "incidents": [
            {
                "incident_id":   f"INC-MIR-{str(uuid4())[:8]}",
                "zone_id":       "mirpur-dhaka",
                "zone_name":     "Mirpur, Dhaka",
                # ↓ Raw message in Banglish — Agent 2's NLP extracted this
                "raw_message":   "Mirpur e pani utheche, bari dube jacche, mirpur 10 er kache",
                # ↓ Coarse location — LocationResolver will refine this
                "raw_location":  "Mirpur, Dhaka",
                # ↓ Agent 2's best-guess coord (coarse — centre of Mirpur)
                "latitude":      23.8041,
                "longitude":     90.3654,
                "urgency":       "LIFE_THREATENING",
                "num_people":    15,
                "medical_need":  True,
                "priority":      5,
                "source":        "social_media_banglish",
            },
            {
                "incident_id":   f"INC-MIR-{str(uuid4())[:8]}",
                "zone_id":       "mirpur-12-dhaka",
                "zone_name":     "Mirpur-12, Dhaka",
                "raw_message":   "Section 12 te khub beshi pani, road block",
                "raw_location":  "Mirpur Section 12, Dhaka",
                "latitude":      23.8249,
                "longitude":     90.3710,
                "urgency":       "URGENT",
                "num_people":    8,
                "medical_need":  False,
                "priority":      4,
                "source":        "social_media_banglish",
            },
        ]
    },

    # ── Scenario 2: Sylhet flood ──────────────────────────────────────────
    "sylhet_flood": {
        "description": "Flooding near Sylhet Station — high confidence location",
        "incidents": [
            {
                "incident_id":   f"INC-SYL-{str(uuid4())[:8]}",
                "zone_id":       "sylhet-city",
                "zone_name":     "Sylhet City",
                "raw_message":   "Water entering homes near Sylhet station, 8 families trapped",
                "raw_location":  "Sylhet City",
                "latitude":      24.8975,
                "longitude":     91.8720,
                "urgency":       "URGENT",
                "num_people":    8,
                "medical_need":  False,
                "priority":      4,
                "source":        "social_media_english",
            },
        ]
    },

    # ── Scenario 3: Multi-zone simultaneous flood ─────────────────────────
    "multi_zone": {
        "description": "Simultaneous incidents across 3 zones — tests resource contention",
        "incidents": [
            {
                "incident_id":   f"INC-MIR-{str(uuid4())[:8]}",
                "zone_id":       "mirpur-10-dhaka",
                "zone_name":     "Mirpur-10, Dhaka",
                "raw_message":   "Mirpur 10 circle er kache pani utheche, ekta bari dube gese",
                "raw_location":  "Mirpur 10, Dhaka",
                "latitude":      23.8058,
                "longitude":     90.3689,
                "urgency":       "LIFE_THREATENING",
                "num_people":    20,
                "medical_need":  True,
                "priority":      5,
                "source":        "social_media_banglish",
            },
            {
                "incident_id":   f"INC-SYL-{str(uuid4())[:8]}",
                "zone_id":       "sylhet-ambarkhana",
                "zone_name":     "Ambarkhana, Sylhet",
                "raw_message":   "Ambarkhana area flooded, people need food and water",
                "raw_location":  "Ambarkhana, Sylhet",
                "latitude":      24.8960,
                "longitude":     91.8784,
                "urgency":       "MODERATE",
                "num_people":    50,
                "medical_need":  False,
                "priority":      3,
                "source":        "social_media_english",
            },
            {
                "incident_id":   f"INC-SIR-{str(uuid4())[:8]}",
                "zone_id":       "sirajganj-river",
                "zone_name":     "Sirajganj River Bank",
                "raw_message":   "Sirajganj river overflowing, 3 boats needed urgently",
                "raw_location":  "Sirajganj",
                "latitude":      24.4490,
                "longitude":     89.6950,
                "urgency":       "URGENT",
                "num_people":    30,
                "medical_need":  True,
                "priority":      4,
                "source":        "official_report",
            },
        ]
    },

    # ── Scenario 4: Kawran Bazar (unverified/moderate) ────────────────────
    "kawran_bazar": {
        "description": "Road flooding — moderate urgency, unverified",
        "incidents": [
            {
                "incident_id":   f"INC-KAW-{str(uuid4())[:8]}",
                "zone_id":       "kawran-bazar-dhaka",
                "zone_name":     "Kawran Bazar, Dhaka",
                "raw_message":   "Road flooded near Kawran Bazar, traffic stuck",
                "raw_location":  "Kawran Bazar, Dhaka",
                "latitude":      23.7514,
                "longitude":     90.3930,
                "urgency":       "MODERATE",
                "num_people":    0,
                "medical_need":  False,
                "priority":      2,
                "source":        "social_media_english",
            },
        ]
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLISHER
# ─────────────────────────────────────────────────────────────────────────────

async def publish_flood_alert(redis: aioredis.Redis, zone_name: str, risk_level: str = "high"):
    """Simulate what Agent 1 publishes to flood_alert channel."""
    msg = AgentMessage(
        sender_agent="agent_1_environmental",
        receiver_agent="agent_2_distress",
        message_type="flood_alert",
        zone_id=zone_name.lower().replace(" ", "-"),
        priority=5,
        payload={
            "zone_name":        zone_name,
            "risk_level":       risk_level,
            "risk_score":       0.785,
            "confidence":       0.852,
            "satellite_source": "Sentinel-1 SAR (mock)",
            "weather": {
                "rain_mm_h": 42.5,
                "humidity":  93,
                "temp_c":    27.2,
                "wind_ms":   7.3,
            },
        },
    )
    await redis.publish("flood_alert", msg.model_dump_json())
    print(f"  ✓ Published flood_alert for {zone_name}")


async def publish_distress_queue(redis: aioredis.Redis, incidents: list):
    """Simulate what Agent 2 publishes to distress_queue channel."""
    msg = AgentMessage(
        sender_agent="agent_2_distress",
        receiver_agent="agent_3_resource",
        message_type="distress_queue",
        priority=5,
        payload={"incidents": incidents},
    )
    await redis.publish("distress_queue", msg.model_dump_json())
    print(f"  ✓ Published distress_queue with {len(incidents)} incident(s)")
    for inc in incidents:
        print(
            f"    → [{inc['urgency']}] {inc['zone_name']} "
            f"| ~{inc['num_people']} people | priority={inc['priority']}"
        )


async def run_scenario(scenario_name: str):
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)

    try:
        await redis.ping()
    except Exception as e:
        print(f"❌ Cannot connect to Redis at {REDIS_URL}: {e}")
        return

    scenarios_to_run = (
        list(SCENARIOS.keys()) if scenario_name == "all" else [scenario_name]
    )

    for name in scenarios_to_run:
        if name not in SCENARIOS:
            print(f"❌ Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
            continue

        scenario = SCENARIOS[name]
        print(f"\n{'='*60}")
        print(f"🌊 Running scenario: {name}")
        print(f"   {scenario['description']}")
        print(f"{'='*60}")

        # Step 1: Publish flood_alert (what Agent 1 would send)
        for inc in scenario["incidents"]:
            await publish_flood_alert(redis, inc["zone_name"])
            await asyncio.sleep(0.2)

        # Step 2: Small delay (simulate Agent 2 processing)
        print("  ⏳ Simulating Agent 2 NLP processing (1s)…")
        await asyncio.sleep(1.0)

        # Step 3: Publish distress_queue (what Agent 2 would send)
        await publish_distress_queue(redis, scenario["incidents"])

        print(f"\n  ✅ Scenario '{name}' published to Redis.")
        print(f"     Agent 3 should now allocate resources…")
        print(f"     Agent 4 should compute routes within 2–3 seconds.")

        if scenario_name == "all":
            await asyncio.sleep(3.0)  # Gap between scenarios

    await redis.aclose()
    print("\n🎉 Done. Check Agent 3 logs and GET /inventory + GET /allocations")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock Agent 1 & 2 publisher")
    parser.add_argument(
        "--scenario",
        default="mirpur_flood",
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Which scenario to publish (default: mirpur_flood)",
    )
    args = parser.parse_args()
    asyncio.run(run_scenario(args.scenario))
