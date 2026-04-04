"""
src/agents/agent_3_resource/tests/test_allocator.py
Unit tests for the allocation algorithm — no DB or Redis needed.
Run with: pytest src/agents/agent_3_resource/tests/ -v
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from src.agents.agent_3_resource.allocator import ResourceAllocator
from src.agents.agent_3_resource.models import ResourceType


# ── Mock inventory manager ─────────────────────────────────────────────────

def make_unit(rtype: str, lat: float, lon: float, name: str = None):
    return {
        "id":            str(uuid4()),
        "resource_type": rtype,
        "name":          name or f"{rtype}-mock",
        "status":        "available",
        "capacity":      15,
        "lat":           lat,
        "lon":           lon,
    }


def mock_inventory(available_by_type: dict):
    """
    Returns an AsyncMock InventoryManager with configurable available units.
    available_by_type = {"rescue_boat": [...], "medical_team": [...], ...}
    """
    inv = AsyncMock()
    inv.get_available_units = AsyncMock(
        side_effect=lambda rtype: available_by_type.get(rtype.value, [])
    )
    inv.mark_deployed           = AsyncMock()
    inv.save_allocation         = AsyncMock()
    inv.log_allocation_transaction = AsyncMock()
    return inv


# ── Test cases ────────────────────────────────────────────────────────────

class TestLifeThreateningAllocation:

    @pytest.mark.asyncio
    async def test_gets_boats_and_medical(self):
        """LIFE_THREATENING should allocate 2 boats + 1 medical team + 2 kits."""
        inv = mock_inventory({
            "rescue_boat":  [make_unit("rescue_boat",  23.80, 90.36, "Boat-1"),
                             make_unit("rescue_boat",  23.81, 90.37, "Boat-2")],
            "medical_team": [make_unit("medical_team", 23.75, 90.40, "Med-1")],
            "medical_kit":  [make_unit("medical_kit",  23.78, 90.41, "Kit-1"),
                             make_unit("medical_kit",  23.79, 90.40, "Kit-2")],
        })
        allocator = ResourceAllocator(inv)
        incident = {
            "incident_id":  "TEST-001",
            "zone_id":      "mirpur-10",
            "zone_name":    "Mirpur-10, Dhaka",
            "raw_message":  "pani utheche mirpur 10 circle",
            "raw_location": "Mirpur 10, Dhaka",
            "latitude":     23.8058,
            "longitude":    90.3689,
            "urgency":      "LIFE_THREATENING",
            "num_people":   15,
            "medical_need": True,
            "priority":     5,
        }
        allocation = await allocator.allocate_for_incident(incident)

        assert allocation is not None
        types = [r["resource_type"] for r in allocation.allocated_resources]
        assert "rescue_boat"  in types
        assert "medical_team" in types
        assert "medical_kit"  in types
        assert allocation.requires_medical is True
        assert allocation.partial_allocation is False

    @pytest.mark.asyncio
    async def test_partial_when_boats_exhausted(self):
        """If only 1 boat available but 2 needed → partial_allocation=True."""
        inv = mock_inventory({
            "rescue_boat":  [make_unit("rescue_boat", 23.80, 90.36)],  # only 1!
            "medical_team": [make_unit("medical_team", 23.75, 90.40)],
            "medical_kit":  [make_unit("medical_kit",  23.78, 90.41),
                             make_unit("medical_kit",  23.79, 90.40)],
        })
        allocator = ResourceAllocator(inv)
        incident = {
            "incident_id": "TEST-002", "zone_id": "sylhet-1",
            "zone_name": "Sylhet", "raw_message": "flood", "raw_location": "Sylhet",
            "latitude": 24.89, "longitude": 91.87,
            "urgency": "LIFE_THREATENING", "num_people": 20,
            "medical_need": True, "priority": 5,
        }
        allocation = await allocator.allocate_for_incident(incident)
        assert allocation.partial_allocation is True

    @pytest.mark.asyncio
    async def test_returns_none_when_completely_empty(self):
        """Zero resources → allocate_for_incident returns None."""
        inv = mock_inventory({})
        allocator = ResourceAllocator(inv)
        incident = {
            "incident_id": "TEST-003", "zone_id": "z1",
            "zone_name": "Zone 1", "raw_message": "help", "raw_location": "Dhaka",
            "latitude": 23.78, "longitude": 90.41,
            "urgency": "LIFE_THREATENING", "num_people": 10,
            "medical_need": True, "priority": 5,
        }
        result = await allocator.allocate_for_incident(incident)
        assert result is None


class TestLocationResolution:

    @pytest.mark.asyncio
    async def test_mirpur_10_resolved_precisely(self):
        """'mirpur 10 circle' in message → should resolve to exact Mirpur-10 coords."""
        inv = mock_inventory({
            "rescue_boat": [make_unit("rescue_boat", 23.80, 90.36)],
        })
        allocator = ResourceAllocator(inv)
        incident = {
            "incident_id": "TEST-LOC-001", "zone_id": "mirpur",
            "zone_name": "Mirpur, Dhaka",
            "raw_message": "Mirpur 10 circle er pashe pani utheche",
            "raw_location": "Mirpur, Dhaka",
            "latitude": 23.8041, "longitude": 90.3654,   # coarse
            "urgency": "URGENT", "num_people": 5,
            "medical_need": False, "priority": 4,
        }
        allocation = await allocator.allocate_for_incident(incident)
        assert allocation is not None
        # Destination should be refined towards Mirpur-10 coords
        assert abs(allocation.destination.latitude - 23.8058) < 0.01
        assert "landmark" in allocation.notes.lower() or "subzone" in allocation.notes.lower()

    @pytest.mark.asyncio
    async def test_vague_mirpur_flags_followup(self):
        """Plain 'Mirpur' with no sector → needs_followup flag in notes."""
        inv = mock_inventory({
            "food_supply":  [make_unit("food_supply",  23.80, 90.36)],
            "water_supply": [make_unit("water_supply", 23.81, 90.37)],
        })
        allocator = ResourceAllocator(inv)
        incident = {
            "incident_id": "TEST-LOC-002", "zone_id": "mirpur",
            "zone_name": "Mirpur", "raw_message": "pani utheche mirpur",
            "raw_location": "Mirpur",
            "latitude": 23.8041, "longitude": 90.3654,
            "urgency": "MODERATE", "num_people": 0,
            "medical_need": False, "priority": 2,
        }
        allocation = await allocator.allocate_for_incident(incident)
        assert allocation is not None
        # Notes should mention low confidence or follow-up
        assert "confidence" in allocation.notes.lower() or "low" in allocation.notes.lower()


class TestClosestResourcePreferred:

    @pytest.mark.asyncio
    async def test_closest_boat_selected_first(self):
        """
        Two boats: one near Mirpur (closer), one near Sadarghat (far).
        Incident in Mirpur-10. Should pick the Mirpur boat.
        """
        mirpur_boat   = make_unit("rescue_boat", 23.8041, 90.3654, "Boat-Mirpur")
        sadarghat_boat = make_unit("rescue_boat", 23.7104, 90.4074, "Boat-Sadarghat")

        inv = mock_inventory({
            "rescue_boat": [mirpur_boat, sadarghat_boat],
        })
        allocator = ResourceAllocator(inv)
        incident = {
            "incident_id": "TEST-DIST-001", "zone_id": "mirpur-10",
            "zone_name": "Mirpur-10", "raw_message": "mirpur 10 flood",
            "raw_location": "Mirpur 10, Dhaka",
            "latitude": 23.8058, "longitude": 90.3689,
            "urgency": "URGENT", "num_people": 8,
            "medical_need": False, "priority": 4,
        }
        allocation = await allocator.allocate_for_incident(incident)
        assert allocation is not None
        allocated_names = [r["name"] for r in allocation.allocated_resources]
        assert "Boat-Mirpur" in allocated_names


class TestPriorityOrdering:

    @pytest.mark.asyncio
    async def test_higher_priority_gets_resources_first(self):
        """
        Two incidents: one LIFE_THREATENING (priority 5), one MODERATE (priority 2).
        Only 2 boats available. LIFE_THREATENING should get both.
        """
        inv = mock_inventory({
            "rescue_boat":  [make_unit("rescue_boat", 23.80, 90.36, "B1"),
                             make_unit("rescue_boat", 23.81, 90.37, "B2")],
            "medical_team": [make_unit("medical_team", 23.75, 90.40)],
            "medical_kit":  [make_unit("medical_kit",  23.78, 90.41),
                             make_unit("medical_kit",  23.79, 90.40)],
            "food_supply":  [],   # None available
            "water_supply": [],
        })
        allocator = ResourceAllocator(inv)
        incidents = [
            {   # Low priority — comes first in list but should be processed second
                "incident_id": "MODERATE-001", "zone_id": "zone-a",
                "zone_name": "Zone A", "raw_message": "flooding",
                "raw_location": "Dhaka", "latitude": 23.78, "longitude": 90.41,
                "urgency": "MODERATE", "num_people": 5,
                "medical_need": False, "priority": 2,
            },
            {   # High priority
                "incident_id": "CRITICAL-001", "zone_id": "zone-b",
                "zone_name": "Zone B", "raw_message": "life threatening flood",
                "raw_location": "Sylhet", "latitude": 24.89, "longitude": 91.87,
                "urgency": "LIFE_THREATENING", "num_people": 20,
                "medical_need": True, "priority": 5,
            },
        ]
        allocations = await allocator.process_distress_queue(incidents)
        # CRITICAL should be first allocation with boats
        critical_alloc = next(
            (a for a in allocations if a.incident_id == "CRITICAL-001"), None
        )
        assert critical_alloc is not None
        boat_count = sum(
            1 for r in critical_alloc.allocated_resources
            if r["resource_type"] == "rescue_boat"
        )
        assert boat_count >= 1   # Got at least 1 boat (priority served first)
