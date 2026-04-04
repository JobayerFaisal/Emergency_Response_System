"""
src/agents/agent_4_dispatch/tests/test_route_computer.py
Unit tests for RouteComputer — OSRM is mocked so no Docker needed.
Run with: pytest src/agents/agent_4_dispatch/tests/ -v
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from src.agents.agent_4_dispatch.route_computer import RouteComputer
from src.agents.agent_4_dispatch.models import TransportMode
from shared.severity import GeoPoint


MIRPUR_10  = GeoPoint(latitude=23.8058, longitude=90.3689)
SYLHET_STN = GeoPoint(latitude=24.8975, longitude=91.8720)
SIRAJGANJ  = GeoPoint(latitude=24.4490, longitude=89.6950)


# ── OSRM mock response ────────────────────────────────────────────────────

MOCK_OSRM_ROUTE = {
    "code": "Ok",
    "routes": [{
        "distance": 15000,   # 15 km
        "duration": 1800,    # 30 min
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [90.3689, 23.8058],
                [91.2000, 24.3000],
                [91.8720, 24.8975],
            ],
        },
    }]
}

MOCK_OSRM_FAIL = {"code": "NoRoute", "message": "No route found"}


class TestRoadRoute:

    @pytest.mark.asyncio
    async def test_uses_osrm_when_available(self):
        """Road route should call OSRM and return real distance/duration."""
        router = RouteComputer(osrm_url="http://mock-osrm")

        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_OSRM_ROUTE

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await router.compute_route(
                origin=MIRPUR_10,
                destination=SYLHET_STN,
                transport_mode=TransportMode.ROAD,
            )

        assert result["distance_km"]  == 15.0
        assert result["eta_minutes"]  == 30.0
        assert result["route_geometry"] is not None
        assert result["source"]        == "osrm"

    @pytest.mark.asyncio
    async def test_fallback_when_osrm_down(self):
        """When OSRM is unreachable, should return haversine-based fallback."""
        router = RouteComputer(osrm_url="http://nonexistent-host")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await router.compute_route(
                origin=MIRPUR_10,
                destination=SYLHET_STN,
                transport_mode=TransportMode.ROAD,
            )

        assert result["source"]       == "fallback_haversine"
        assert result["distance_km"]  > 0
        assert result["eta_minutes"]  > 0
        assert result["route_geometry"]["type"] == "LineString"

    @pytest.mark.asyncio
    async def test_road_route_geometry_is_geojson(self):
        """Route geometry must be a valid GeoJSON LineString."""
        router = RouteComputer(osrm_url="http://mock-osrm")

        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_OSRM_ROUTE

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__  = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await router.compute_route(
                origin=MIRPUR_10,
                destination=SYLHET_STN,
                transport_mode=TransportMode.ROAD,
            )

        geom = result["route_geometry"]
        assert geom["type"] == "LineString"
        assert len(geom["coordinates"]) >= 2
        for coord in geom["coordinates"]:
            lon, lat = coord
            assert -180 <= lon <= 180
            assert  -90 <= lat <=  90


class TestBoatRoute:

    @pytest.mark.asyncio
    async def test_boat_does_not_call_osrm(self):
        """Boat route must use haversine only — never calls OSRM."""
        router = RouteComputer(osrm_url="http://mock-osrm")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value = mock_client

            result = await router.compute_route(
                origin=MIRPUR_10,
                destination=SYLHET_STN,
                transport_mode=TransportMode.WATERWAY,
            )

            # OSRM should NOT have been called
            mock_client.__aenter__.assert_not_called()

        assert result["source"]       == "boat_haversine"
        assert result["distance_km"]  > 0
        assert result["eta_minutes"]  > 0

    @pytest.mark.asyncio
    async def test_boat_speed_varies_by_flood_condition(self):
        """Severe flood should give longer ETA than shallow flood."""
        router = RouteComputer()
        origin = MIRPUR_10
        dest   = SYLHET_STN

        result_shallow = await router.compute_route(
            origin, dest, TransportMode.WATERWAY, flood_condition="flood_shallow"
        )
        result_severe = await router.compute_route(
            origin, dest, TransportMode.WATERWAY, flood_condition="flood_severe"
        )

        # Same distance, slower speed → longer ETA
        assert result_severe["eta_minutes"] > result_shallow["eta_minutes"]
        assert result_shallow["distance_km"] == result_severe["distance_km"]

    @pytest.mark.asyncio
    async def test_boat_route_is_straight_line(self):
        """Boat GeoJSON should be a 2-point straight LineString."""
        router = RouteComputer()
        result = await router.compute_route(
            MIRPUR_10, SYLHET_STN, TransportMode.WATERWAY
        )
        geom = result["route_geometry"]
        assert geom["type"] == "LineString"
        assert len(geom["coordinates"]) == 2   # straight line = exactly 2 points
        # Check [lon, lat] ordering (GeoJSON standard)
        start = geom["coordinates"][0]
        assert start[0] == pytest.approx(MIRPUR_10.longitude, abs=0.001)
        assert start[1] == pytest.approx(MIRPUR_10.latitude,  abs=0.001)


class TestLocationResolver:

    def test_mirpur_10_landmark_resolved(self):
        """'mirpur 10' in message → precise Mirpur-10 Circle coordinates."""
        from shared.location_resolver import resolver
        result = resolver.resolve(
            raw_message="Mirpur 10 circle te pani",
            coarse_location="Mirpur, Dhaka",
        )
        assert result.resolution_method == "landmark"
        assert result.confidence >= 0.90
        assert abs(result.latitude  - 23.8058) < 0.005
        assert abs(result.longitude - 90.3689) < 0.005
        assert result.needs_followup is False

    def test_vague_mirpur_flags_followup(self):
        """'Mirpur' alone with no sector → needs_followup=True."""
        from shared.location_resolver import resolver
        result = resolver.resolve(
            raw_message="Mirpur e bari dube jacche",
            coarse_location="Mirpur, Dhaka",
        )
        # Should resolve to SOME mirpur subzone, but flag follow-up due to ambiguity
        assert result.latitude > 23.7
        assert result.longitude > 90.3

    def test_sylhet_station_resolved(self):
        """'Sylhet station' → high-confidence station coordinates."""
        from shared.location_resolver import resolver
        result = resolver.resolve(
            raw_message="Water entering homes near Sylhet station",
            coarse_location="Sylhet City",
        )
        assert result.resolution_method in ("landmark", "subzone_keyword")
        assert result.confidence >= 0.70

    def test_unknown_location_fallback(self):
        """Completely unknown area → default fallback, confidence < 0.5."""
        from shared.location_resolver import resolver
        result = resolver.resolve(
            raw_message="some unknown place xyz",
            coarse_location="XYZ Unknown Area",
        )
        assert result.confidence < 0.5
        assert result.needs_followup is True
