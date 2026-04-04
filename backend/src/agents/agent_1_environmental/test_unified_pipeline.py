"""
Test: Unified Pipeline + 8-Factor Predictor (Historical Floods)
================================================================
Runs the FULL pipeline end-to-end using known historical flood events:
  1. Satellite SAR with custom historical dates (detects real flooding)
  2. Weather API (current snapshot at the location)
  3. 8-factor predictor with satellite override logic

Uses the same flood scenarios as test_historical_floods.py but feeds
results through the new predictor to show risk scores, severity levels,
satellite override, and recommended actions.

Run from: disaster-response-system/src/agents/agent_1_environmental/
    python test_unified_pipeline.py

No Docker/PostgreSQL/Redis needed.
"""

import asyncio
import sys
import os
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from models import SentinelZone, GeoPoint, SeverityLevel, SpatialAnalysisResult
from data_collectors import WeatherAPICollector, DataCollectionOrchestrator
from data_processors import WeatherDataNormalizer
from services.satellite_service import (
    SatelliteDataCollector, FloodDetectionResult,
    _download_sar_image, _preprocess_image,
    _detect_flood_change, _analyze_flood,
)
from predictor import FloodRiskPredictor, AlertGenerator, PredictionOrchestrator


# ============================================================
# HISTORICAL FLOOD SCENARIOS
# ============================================================
# Each scenario has a dry-season reference period and a known
# flood period where SAR change detection should find flooding.

FLOOD_SCENARIOS = {
    # ── 2024 Monsoon (Jun-Sep) ──────────────────────────────
    "sylhet_2024": {
        "name": "Sylhet - Surma River (2024 Monsoon)",
        "description": "Major river basin, floods annually during monsoon",
        "center": (24.93, 91.90),
        "radius_km": 5.0,
        "elevation": 5.0,
        "drainage": "poor",
        "population_density": 28000,
        "ref_start": "2024-01-01",
        "ref_end": "2024-03-31",
        "flood_start": "2024-06-01",
        "flood_end": "2024-09-30",
    },
    "sunamganj_2024": {
        "name": "Sunamganj Haor (2024 Monsoon)",
        "description": "Wetland haor region, extremely flood-prone",
        "center": (25.06, 91.27),
        "radius_km": 8.0,
        "elevation": 3.0,
        "drainage": "very_poor",
        "population_density": 15000,
        "ref_start": "2024-01-01",
        "ref_end": "2024-03-31",
        "flood_start": "2024-06-01",
        "flood_end": "2024-09-30",
    },
    "sirajganj_2024": {
        "name": "Sirajganj - Jamuna River (2024 Monsoon)",
        "description": "Major Jamuna river floodplain",
        "center": (24.47, 89.72),
        "radius_km": 7.0,
        "elevation": 7.0,
        "drainage": "poor",
        "population_density": 22000,
        "ref_start": "2024-01-01",
        "ref_end": "2024-03-31",
        "flood_start": "2024-06-01",
        "flood_end": "2024-09-30",
    },

    # ── 2022 Historic Sylhet Flood (Jun-Jul) ────────────────
    "sylhet_2022": {
        "name": "Sylhet City (2022 Historic Flood)",
        "description": "Worst flood in 122 years, June-July 2022",
        "center": (24.90, 91.87),
        "radius_km": 6.0,
        "elevation": 5.0,
        "drainage": "poor",
        "population_density": 35000,
        "ref_start": "2022-01-01",
        "ref_end": "2022-03-31",
        "flood_start": "2022-06-01",
        "flood_end": "2022-07-31",
    },
    "sunamganj_2022": {
        "name": "Sunamganj Haor (2022 Historic Flood)",
        "description": "Completely submerged during 2022 floods",
        "center": (25.06, 91.40),
        "radius_km": 10.0,
        "elevation": 3.0,
        "drainage": "very_poor",
        "population_density": 12000,
        "ref_start": "2022-01-01",
        "ref_end": "2022-03-31",
        "flood_start": "2022-06-01",
        "flood_end": "2022-07-31",
    },
}

# Quick-run default — change this to test a different scenario
DEFAULT_SCENARIO = "sylhet_2024"


# ============================================================
# SATELLITE WITH HISTORICAL DATES
# ============================================================

async def run_satellite_historical(
    collector: SatelliteDataCollector,
    zone: SentinelZone,
    scenario: dict,
    zone_id: str,
) -> FloodDetectionResult:
    """
    Run satellite flood detection with custom historical dates.
    Overrides the default 'last 30 days' to a known flood period.
    """
    collector._ensure_ee()
    collector._ensure_model()

    if collector._model is None:
        return FloodDetectionResult(
            zone_id=zone_id, zone_name=zone.name,
            bounds=[0, 0, 0, 0],
            timestamp=datetime.now(timezone.utc).isoformat(),
            error="Flood detection model not available",
        )

    bounds = collector._zone_to_bounds(zone)

    print(f"      Bounds: {bounds}")
    print(f"      Reference (dry): {scenario['ref_start']} -> {scenario['ref_end']}")
    print(f"      Flood period:    {scenario['flood_start']} -> {scenario['flood_end']}")

    # Download reference SAR (dry season baseline)
    print(f"      Downloading reference SAR...")
    ref_raw, ref_date = await asyncio.to_thread(
        _download_sar_image, bounds,
        scenario['ref_start'], scenario['ref_end']
    )
    if ref_raw is None:
        return FloodDetectionResult(
            zone_id=zone_id, zone_name=zone.name, bounds=bounds,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error="No reference SAR data",
        )
    print(f"         Reference SAR from: {ref_date}")

    # Download flood period SAR
    print(f"      Downloading flood period SAR...")
    flood_raw, flood_date = await asyncio.to_thread(
        _download_sar_image, bounds,
        scenario['flood_start'], scenario['flood_end']
    )
    if flood_raw is None:
        return FloodDetectionResult(
            zone_id=zone_id, zone_name=zone.name, bounds=bounds,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error="No flood period SAR data",
        )
    print(f"         Flood SAR from: {flood_date}")

    # Preprocess and detect
    ref_proc = _preprocess_image(ref_raw, 64)
    flood_proc = _preprocess_image(flood_raw, 64)
    if ref_proc is None or flood_proc is None:
        return FloodDetectionResult(
            zone_id=zone_id, zone_name=zone.name, bounds=bounds,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error="SAR preprocessing failed",
        )

    print(f"      Running CNN change detection...")
    detection = _detect_flood_change(collector._model, ref_proc, flood_proc)
    analysis = _analyze_flood(detection)

    # Calculate flood area in km²
    cos_lat = np.cos(np.radians((bounds[1] + bounds[3]) / 2))
    zone_w_km = (bounds[2] - bounds[0]) * 111.0 * cos_lat
    zone_h_km = (bounds[3] - bounds[1]) * 111.0
    total_area = zone_w_km * zone_h_km
    flood_area = total_area * (analysis['flood_pct'] / 100.0)

    return FloodDetectionResult(
        zone_id=zone_id,
        zone_name=zone.name,
        bounds=bounds,
        timestamp=datetime.now(timezone.utc).isoformat(),
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


# ============================================================
# WEATHER SNAPSHOT
# ============================================================

async def fetch_weather(lat: float, lon: float):
    """Fetch current weather for a location"""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        print("      OPENWEATHER_API_KEY not set — skipping weather")
        return None, None

    collector = WeatherAPICollector(api_key=api_key)
    normalizer = WeatherDataNormalizer()

    location = GeoPoint(latitude=lat, longitude=lon)
    weather = await collector.fetch_current_weather(location)

    if weather:
        normalized = normalizer.normalize_weather_data(weather)
        return weather, normalized
    return None, None


# ============================================================
# FULL PIPELINE: SATELLITE + WEATHER + PREDICTOR
# ============================================================

async def run_full_pipeline(scenario_key: str, scenario: dict):
    """
    Run the complete pipeline for one historical flood scenario:
      1. Satellite SAR detection (historical dates)
      2. Weather API (current conditions at location)
      3. 8-factor predictor with satellite override
    """
    print(f"\n{'=' * 65}")
    print(f"  {scenario['name']}")
    print(f"  {scenario['description']}")
    print(f"{'=' * 65}")

    # --- Create zone ---
    zone = SentinelZone(
        name=scenario['name'],
        center=GeoPoint(
            latitude=scenario['center'][0],
            longitude=scenario['center'][1]
        ),
        radius_km=scenario['radius_km'],
        risk_level=SeverityLevel.HIGH,
        population_density=scenario.get('population_density', 30000),
        elevation=scenario.get('elevation', 5.0),
        drainage_capacity=scenario.get('drainage', 'poor'),
    )

    start_time = datetime.now(timezone.utc)

    # --- Step 1: Satellite (historical dates) ---
    print(f"\n  STEP 1: Satellite Flood Detection")
    print(f"  {'─' * 55}")

    sat_collector = SatelliteDataCollector()
    fd = await run_satellite_historical(
        sat_collector, zone, scenario, scenario_key
    )

    if fd.error:
        print(f"\n      Error: {fd.error}")
        return None

    risk_emoji = {
        "CRITICAL": "🔴", "HIGH": "🟠",
        "MEDIUM": "🟡", "LOW": "🟡", "MINIMAL": "🟢"
    }
    emoji = risk_emoji.get(fd.risk_level, "⚪")

    print(f"\n      {emoji} {fd.status}")
    print(f"      Risk level:       {fd.risk_level}")
    print(f"      Flood percentage: {fd.flood_percentage:.1f}%")
    print(f"      Flood area:       {fd.flood_area_km2:.3f} km²")
    print(f"      Permanent water:  {fd.permanent_water_pct:.1f}%")
    print(f"      Current water:    {fd.current_water_pct:.1f}%")
    print(f"      Confidence:       {fd.confidence:.2f}")

    # --- Step 2: Weather ---
    print(f"\n  STEP 2: Weather Data")
    print(f"  {'─' * 55}")

    weather, normalized_weather = await fetch_weather(
        scenario['center'][0], scenario['center'][1]
    )

    if weather:
        print(f"      Condition:    {weather.condition.value}")
        print(f"      Temperature:  {weather.metrics.temperature}°C")
        print(f"      Humidity:     {weather.metrics.humidity}%")
        print(f"      Wind:         {weather.metrics.wind_speed} m/s")
        if weather.precipitation.rain_1h:
            print(f"      Rain (1h):    {weather.precipitation.rain_1h} mm")
        if normalized_weather:
            print(f"      Normalized:   rainfall={normalized_weather.get('rainfall_intensity', 0):.2f}, "
                  f"severity={normalized_weather.get('weather_severity', 0):.2f}")
    else:
        print(f"      Weather unavailable")

    # --- Step 3: Run 8-factor predictor ---
    print(f"\n  STEP 3: 8-Factor Flood Prediction")
    print(f"  {'─' * 55}")

    predictor = FloodRiskPredictor()
    alert_gen = AlertGenerator()
    orchestrator = PredictionOrchestrator(predictor, alert_gen)

    # Build the processed_data dict (same format main.py produces)
    processed_data = {
        'zone': zone,
        'weather': weather,
        'normalized_weather': normalized_weather or {
            'rainfall_intensity': 0.0,
            'accumulated_rainfall': 0.0,
            'weather_severity': 0.0,
        },
        'social_analysis': {},  # No social data — tests optional handling
        # Satellite data keys (same as main.py line 452-454)
        'satellite_risk': fd.risk_level,
        'satellite_flood_pct': fd.flood_percentage,
        'satellite_flood_area_km2': fd.flood_area_km2,
        'satellite_confidence': fd.confidence,
    }

    prediction, alert = await orchestrator.predict_for_zone(processed_data)

    # --- Display prediction results ---
    severity_emoji = {
        "minimal": "🟢", "low": "🟡",
        "moderate": "🟠", "high": "🔴", "critical": "🚨"
    }
    s_emoji = severity_emoji.get(prediction.severity_level.value, "⚪")

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    print(f"\n      {s_emoji} PREDICTION RESULT")
    print(f"      {'─' * 45}")
    print(f"      Risk Score:     {prediction.risk_score:.2%}")
    print(f"      Severity:       {prediction.severity_level.value.upper()}")
    print(f"      Confidence:     {prediction.confidence:.2%}")
    print(f"      Affected Area:  {prediction.affected_area_km2:.3f} km²")

    if prediction.time_to_impact_hours is not None:
        print(f"      Time to Impact: {prediction.time_to_impact_hours:.1f} hours")
    elif prediction.risk_factors.satellite_confirmed_flooding:
        print(f"      Time to Impact: NOW (satellite confirmed)")

    # Factor breakdown
    rf = prediction.risk_factors
    print(f"\n      8-FACTOR BREAKDOWN:")
    print(f"      {'─' * 45}")
    print(f"      Satellite flood detection: {rf.satellite_flood_detection:.3f}  {'← HIGHEST WEIGHT' if rf.has_satellite_data else ''}")
    print(f"      Flood depth estimate:      {rf.flood_depth_estimate:.3f}")
    print(f"      Rainfall intensity:        {rf.rainfall_intensity:.3f}")
    print(f"      Accumulated rainfall:      {rf.accumulated_rainfall:.3f}")
    print(f"      Weather severity:          {rf.weather_severity:.3f}")
    print(f"      Drainage capacity:         {rf.drainage_factor:.3f}")
    print(f"      Elevation risk:            {rf.elevation_factor:.3f}")
    print(f"      Social media reports:      {rf.social_reports_density:.3f}  {'(data available)' if rf.has_social_data else '(no data — weight redistributed)'}")
    print(f"\n      Flags:")
    print(f"      Satellite data:     {'YES' if rf.has_satellite_data else 'NO'}")
    print(f"      Social data:        {'YES' if rf.has_social_data else 'NO'}")
    print(f"      SAR confirmed flood: {'YES' if rf.satellite_confirmed_flooding else 'NO'}")
    print(f"      Override active:     {'YES (score floored at 0.65)' if rf.satellite_confirmed_flooding else 'NO'}")

    # Recommended actions
    if prediction.recommended_actions:
        print(f"\n      RECOMMENDED ACTIONS:")
        print(f"      {'─' * 45}")
        for i, action in enumerate(prediction.recommended_actions[:6], 1):
            print(f"      {i}. {action}")

    # Alert
    if alert:
        print(f"\n      ALERT (Priority {alert.priority}/5):")
        print(f"      {'─' * 45}")
        for line in alert.message.split('\n'):
            print(f"      {line}")

    print(f"\n      Total processing time: {elapsed:.1f}s")

    return {
        'scenario': scenario_key,
        'name': scenario['name'],
        'satellite_risk': fd.risk_level,
        'flood_pct': fd.flood_percentage,
        'flood_area_km2': fd.flood_area_km2,
        'prediction_risk': round(prediction.risk_score, 4),
        'severity': prediction.severity_level.value,
        'confidence': round(prediction.confidence, 4),
        'satellite_confirmed': rf.satellite_confirmed_flooding,
        'override_active': rf.satellite_confirmed_flooding,
        'elapsed_seconds': round(elapsed, 1),
    }


# ============================================================
# INTERACTIVE MENU
# ============================================================

def show_menu():
    """Show scenario selection menu"""
    print("\n" + "=" * 65)
    print("  UNIFIED PIPELINE + 8-FACTOR PREDICTOR TEST")
    print("  (Historical Flood Scenarios)")
    print("=" * 65)

    scenarios = list(FLOOD_SCENARIOS.items())

    print("\n  2024 Monsoon Season:")
    for i, (key, s) in enumerate(scenarios):
        if "2024" in key:
            print(f"    [{i+1}] {s['name']}")

    print("\n  2022 Historic Sylhet Flood:")
    for i, (key, s) in enumerate(scenarios):
        if "2022" in key:
            print(f"    [{i+1}] {s['name']}")

    print(f"\n  [A] Run ALL scenarios")
    print(f"  [Q] Quit")
    print()

    return scenarios


async def main():
    print(f"\n  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"  CWD:  {os.getcwd()}")
    print(f"  .env: OPENWEATHER_API_KEY={'set' if os.getenv('OPENWEATHER_API_KEY') else 'missing'}")
    print(f"        GEE_PROJECT_ID={os.getenv('GEE_PROJECT_ID', 'not set')}")

    scenarios = show_menu()

    choice = input("  Select scenario number (or A/Q): ").strip().upper()

    if choice == 'Q':
        print("  Bye!")
        return

    results = []

    if choice == 'A':
        print(f"\n  Running all {len(scenarios)} scenarios...\n")
        for key, scenario in scenarios:
            try:
                result = await run_full_pipeline(key, scenario)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"      Failed: {e}")
                import traceback
                traceback.print_exc()
            await asyncio.sleep(0.5)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(scenarios):
                key, scenario = scenarios[idx]
                result = await run_full_pipeline(key, scenario)
                if result:
                    results.append(result)
            else:
                print(f"  Invalid choice: {choice}")
                return
        except ValueError:
            print(f"  Invalid choice: {choice}")
            return

    # Summary table
    if results:
        print(f"\n\n{'=' * 65}")
        print("  SUMMARY")
        print(f"{'=' * 65}")
        print(f"\n  {'Location':<35} {'Flood%':<8} {'Risk':<8} {'Severity':<10} {'Override'}")
        print(f"  {'─' * 75}")
        for r in results:
            emoji = risk_emoji = {
                "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡",
                "LOW": "🟡", "MINIMAL": "🟢"
            }.get(r['satellite_risk'], "⚪")

            override_str = "YES" if r['override_active'] else "no"
            print(
                f"  {r['name']:<35} "
                f"{r['flood_pct']:<8.1f} "
                f"{emoji} {r['satellite_risk']:<6} "
                f"{r['severity'].upper():<10} "
                f"{override_str}"
            )

        # Save results
        output_path = Path("output/unified_pipeline_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {output_path}")

    print(f"\n{'=' * 65}")
    print("  Done!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    asyncio.run(main())
