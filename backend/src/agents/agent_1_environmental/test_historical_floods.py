"""
Test: Historical Flood Detection with Unified Pipeline
=======================================================
Tests Weather + Satellite against KNOWN flood events in Bangladesh.
Override dates to monsoon periods where flooding actually occurred.

Known major flood events:
  - Aug 2024: Severe floods in Sylhet, Feni, Noakhali
  - Jun-Jul 2022: Historic Sylhet floods (worst in 122 years)
  - Aug 2017: One-third of Bangladesh submerged

Run from: disaster-response-system/src/agents/agent_1_environmental/
    python test_historical_floods.py

No Docker/PostgreSQL/Redis needed.
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dotenv import load_dotenv
load_dotenv()

from backend.src.agents.agent_1_environmental.models import SentinelZone, GeoPoint, SeverityLevel
from backend.src.agents.agent_1_environmental.data_collectors import WeatherAPICollector, SocialMediaCollector, DataCollectionOrchestrator
from backend.src.agents.agent_1_environmental.services.satellite_service import SatelliteDataCollector
from backend.src.agents.agent_1_environmental.services.satellite_service import (
    _init_earth_engine, _download_sar_image,
    _preprocess_image, _detect_flood_change, _analyze_flood,
    SatelliteDataCollector, FloodDetectionResult
)

# ============================================================
# KNOWN FLOOD EVENTS + TEST ZONES
# ============================================================

FLOOD_SCENARIOS = {
    # ── 2024 Monsoon (Aug-Sep) ──────────────────────────────
    "sylhet_2024_monsoon": {
        "name": "Sylhet - Surma River (2024 Monsoon)",
        "description": "Major river basin, floods annually during monsoon",
        "center": (24.93, 91.90),
        "radius_km": 5.0,
        "ref_start": "2024-01-01",   # Dry season baseline
        "ref_end": "2024-03-31",
        "flood_start": "2024-06-01", # Monsoon period
        "flood_end": "2024-09-30",
    },
    "sunamganj_2024_monsoon": {
        "name": "Sunamganj Haor (2024 Monsoon)",
        "description": "Wetland area, extremely flood-prone haor region",
        "center": (25.06, 91.27),
        "radius_km": 8.0,
        "ref_start": "2024-01-01",
        "ref_end": "2024-03-31",
        "flood_start": "2024-06-01",
        "flood_end": "2024-09-30",
    },
    "sirajganj_2024_monsoon": {
        "name": "Sirajganj - Jamuna River (2024 Monsoon)",
        "description": "Major Jamuna river floodplain",
        "center": (24.47, 89.72),
        "radius_km": 7.0,
        "ref_start": "2024-01-01",
        "ref_end": "2024-03-31",
        "flood_start": "2024-06-01",
        "flood_end": "2024-09-30",
    },
    "kurigram_2024_monsoon": {
        "name": "Kurigram - Brahmaputra (2024 Monsoon)",
        "description": "Northern flood zone, Brahmaputra basin",
        "center": (25.77, 89.67),
        "radius_km": 7.0,
        "ref_start": "2024-01-01",
        "ref_end": "2024-03-31",
        "flood_start": "2024-06-01",
        "flood_end": "2024-09-30",
    },
    "chandpur_2024_monsoon": {
        "name": "Chandpur - Padma-Meghna Confluence (2024 Monsoon)",
        "description": "River confluence, major flood zone",
        "center": (23.27, 90.67),
        "radius_km": 7.0,
        "ref_start": "2024-01-01",
        "ref_end": "2024-03-31",
        "flood_start": "2024-06-01",
        "flood_end": "2024-09-30",
    },

    # ── 2022 Historic Sylhet Flood (Jun-Jul) ────────────────
    "sylhet_2022_historic": {
        "name": "Sylhet City (2022 Historic Flood)",
        "description": "Worst flood in 122 years, June-July 2022",
        "center": (24.90, 91.87),
        "radius_km": 6.0,
        "ref_start": "2022-01-01",   # Dry season 2022
        "ref_end": "2022-03-31",
        "flood_start": "2022-06-01", # Historic flood period
        "flood_end": "2022-07-31",
    },
    "sunamganj_2022_historic": {
        "name": "Sunamganj Haor (2022 Historic Flood)",
        "description": "Completely submerged during 2022 floods",
        "center": (25.06, 91.40),
        "radius_km": 10.0,
        "ref_start": "2022-01-01",
        "ref_end": "2022-03-31",
        "flood_start": "2022-06-01",
        "flood_end": "2022-07-31",
    },
}


# ============================================================
# SATELLITE DETECTION WITH CUSTOM DATES
# ============================================================

async def run_satellite_with_dates(
    zone: SentinelZone,
    ref_start: str, ref_end: str,
    flood_start: str, flood_end: str,
    zone_id: str = "test"
):
    """
    Run satellite flood detection with custom reference + flood dates.
    Overrides the 'last 30 days' default to test historical periods.
    """
    import numpy as np
    from services.satellite_service import (
        _init_earth_engine, _download_sar_image,
        _preprocess_image, _detect_flood_change, _analyze_flood,
        SatelliteDataCollector, FloodDetectionResult
    )

    collector = SatelliteDataCollector()
    collector._ensure_ee()
    collector._ensure_model()

    if collector._model is None:
        return None, "Model not loaded"

    bounds = collector._zone_to_bounds(zone)

    print(f"      Bounds: {bounds}")
    print(f"      Reference (dry): {ref_start} → {ref_end}")
    print(f"      Flood period:    {flood_start} → {flood_end}")

    # Download reference SAR (dry season)
    print(f"      📥 Downloading reference SAR...")
    ref_raw, ref_date = await asyncio.to_thread(
        _download_sar_image, bounds, ref_start, ref_end
    )
    if ref_raw is None:
        return None, "No reference SAR data"
    print(f"         Got reference from: {ref_date}")

    # Download flood period SAR
    print(f"      📥 Downloading flood period SAR...")
    flood_raw, flood_date = await asyncio.to_thread(
        _download_sar_image, bounds, flood_start, flood_end
    )
    if flood_raw is None:
        return None, "No flood period SAR data"
    print(f"         Got flood period from: {flood_date}")

    # Preprocess
    ref_proc = _preprocess_image(ref_raw, 64)
    flood_proc = _preprocess_image(flood_raw, 64)

    if ref_proc is None or flood_proc is None:
        return None, "Preprocessing failed"

    # Run change detection
    print(f"      🔍 Running change detection...")
    detection = _detect_flood_change(collector._model, ref_proc, flood_proc)
    analysis = _analyze_flood(detection)

    # Calculate area
    cos_lat = np.cos(np.radians((bounds[1] + bounds[3]) / 2))
    zone_w_km = (bounds[2] - bounds[0]) * 111.0 * cos_lat
    zone_h_km = (bounds[3] - bounds[1]) * 111.0
    total_area = zone_w_km * zone_h_km
    flood_area = total_area * (analysis['flood_pct'] / 100.0)

    result = FloodDetectionResult(
        zone_id=zone_id,
        zone_name=zone.name,
        bounds=bounds,
        timestamp=datetime.utcnow().isoformat(),
        reference_date=ref_date,
        current_date=flood_date,
        flood_detected=analysis['flood_pct'] > 3.0,
        flood_percentage=round(analysis['flood_pct'], 2),
        permanent_water_pct=round(analysis['perm_water_pct'], 2),
        current_water_pct=round(analysis['cur_water_pct'], 2),
        risk_level=analysis['risk'],
        status=analysis['status'],
        confidence=min(0.95, 0.6 + (analysis['flood_pct'] / 100.0)),
        flood_area_km2=round(flood_area, 3),
    )

    return result, None


# ============================================================
# WEATHER FOR LOCATION (current snapshot)
# ============================================================

async def fetch_weather_snapshot(lat: float, lon: float):
    """Fetch current weather for a location"""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return None

    collector = WeatherAPICollector(api_key=api_key)
    location = GeoPoint(latitude=lat, longitude=lon)
    return await collector.fetch_current_weather(location)


# ============================================================
# INTERACTIVE MENU
# ============================================================

def show_menu():
    """Show scenario selection menu"""
    print("\n" + "=" * 65)
    print("🌊 HISTORICAL FLOOD DETECTION TEST")
    print("=" * 65)
    print("\nAvailable scenarios:\n")

    scenarios = list(FLOOD_SCENARIOS.items())

    print("  ── 2024 Monsoon Season ──────────────────────────────────")
    for i, (key, s) in enumerate(scenarios):
        if "2024" in key:
            print(f"  [{i+1}] {s['name']}")
            print(f"      {s['description']}")

    print("\n  ── 2022 Historic Sylhet Flood ───────────────────────────")
    for i, (key, s) in enumerate(scenarios):
        if "2022" in key:
            print(f"  [{i+1}] {s['name']}")
            print(f"      {s['description']}")

    print(f"\n  [A] Run ALL scenarios")
    print(f"  [Q] Quit\n")

    return scenarios


async def run_single_scenario(key: str, scenario: dict):
    """Run one flood detection scenario"""
    print(f"\n{'─' * 65}")
    print(f"  📍 {scenario['name']}")
    print(f"     {scenario['description']}")
    print(f"{'─' * 65}")

    # Create zone
    zone = SentinelZone(
        name=scenario['name'],
        center=GeoPoint(
            latitude=scenario['center'][0],
            longitude=scenario['center'][1]
        ),
        radius_km=scenario['radius_km'],
        risk_level=SeverityLevel.HIGH,
        population_density=30000,
        elevation=5.0,
        drainage_capacity="poor"
    )

    start = datetime.utcnow()

    # Run satellite detection with historical dates
    result, error = await run_satellite_with_dates(
        zone=zone,
        ref_start=scenario['ref_start'],
        ref_end=scenario['ref_end'],
        flood_start=scenario['flood_start'],
        flood_end=scenario['flood_end'],
        zone_id=key,
    )

    elapsed = (datetime.utcnow() - start).total_seconds()

    if error:
        print(f"\n      ⚠️  Error: {error}")
        return None

    if result is None:
        return None

    # Display results
    risk_emoji = {
        "CRITICAL": "🔴", "HIGH": "🟠",
        "MEDIUM": "🟡", "LOW": "🟡", "MINIMAL": "🟢"
    }
    emoji = risk_emoji.get(result.risk_level, "⚪")

    print(f"\n      {emoji} {result.status}")
    print(f"      Risk level:       {result.risk_level}")
    print(f"      Flood percentage: {result.flood_percentage:.1f}%")
    print(f"      Flood area:       {result.flood_area_km2:.3f} km²")
    print(f"      Permanent water:  {result.permanent_water_pct:.1f}%")
    print(f"      Current water:    {result.current_water_pct:.1f}%")
    print(f"      Confidence:       {result.confidence:.2f}")
    print(f"      Reference SAR:    {result.reference_date}")
    print(f"      Flood SAR:        {result.current_date}")
    print(f"      Processing time:  {elapsed:.1f}s")

    # Also fetch current weather for context
    weather = await fetch_weather_snapshot(
        scenario['center'][0], scenario['center'][1]
    )
    if weather:
        print(f"\n      🌤️  Current weather at location:")
        print(f"         {weather.condition.value}, {weather.metrics.temperature}°C")
        print(f"         Humidity: {weather.metrics.humidity}%, Wind: {weather.metrics.wind_speed} m/s")

    return result


async def main():
    scenarios = show_menu()

    choice = input("  Select scenario number (or A/Q): ").strip().upper()

    if choice == 'Q':
        print("  Bye!")
        return

    results = []

    if choice == 'A':
        # Run all scenarios
        print(f"\n  Running all {len(scenarios)} scenarios...\n")
        for key, scenario in scenarios:
            try:
                result = await run_single_scenario(key, scenario)
                if result:
                    results.append({
                        'scenario': key,
                        'name': scenario['name'],
                        'risk_level': result.risk_level,
                        'flood_pct': result.flood_percentage,
                        'flood_area_km2': result.flood_area_km2,
                        'status': result.status,
                    })
            except Exception as e:
                print(f"      ❌ Failed: {e}")
            # Small delay between GEE calls
            await asyncio.sleep(0.5)
    else:
        # Run single scenario
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(scenarios):
                key, scenario = scenarios[idx]
                result = await run_single_scenario(key, scenario)
                if result:
                    results.append({
                        'scenario': key,
                        'name': scenario['name'],
                        'risk_level': result.risk_level,
                        'flood_pct': result.flood_percentage,
                        'flood_area_km2': result.flood_area_km2,
                        'status': result.status,
                    })
            else:
                print(f"  Invalid choice: {choice}")
                return
        except ValueError:
            print(f"  Invalid choice: {choice}")
            return

    # Summary
    if results:
        print(f"\n\n{'=' * 65}")
        print("📊 SUMMARY")
        print(f"{'=' * 65}")
        print(f"\n  {'Location':<45} {'Flood %':<10} {'Risk'}")
        print(f"  {'─' * 60}")
        for r in results:
            risk_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟡", "MINIMAL": "🟢"}
            emoji = risk_emoji.get(r['risk_level'], "⚪")
            print(f"  {r['name']:<45} {r['flood_pct']:<10.1f} {emoji} {r['risk_level']}")

        # Save results
        output_path = Path("output/historical_flood_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  📁 Results saved to: {output_path}")

    print(f"\n{'=' * 65}")
    print("🎉 Done!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    asyncio.run(main())
