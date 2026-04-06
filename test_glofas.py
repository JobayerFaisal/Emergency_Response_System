"""
test_glofas.py
==============
Tests the Open-Meteo Flood API (powered by GloFAS v4) for all 5
Bangladesh sentinel zones.

No API key required — completely free and open.
Endpoint: https://flood-api.open-meteo.com/v1/flood

FIX: river_discharge_mean / river_discharge_max are only available
when ensemble=true. For the seamless model, only "river_discharge"
is valid. The "models" param is also removed — it caused 400 errors.

Run from project root:
    python test_glofas.py
"""

import requests
from datetime import date

# ── Your 5 sentinel zones ─────────────────────────────────────────────
ZONES = [
    {"name": "Sunamganj Sadar", "lat": 24.8660, "lon": 91.3990},
    {"name": "Sylhet City",     "lat": 24.8975, "lon": 91.8720},
    {"name": "Netrokona Sadar", "lat": 24.8703, "lon": 90.7279},
    {"name": "Sirajganj Sadar", "lat": 24.4490, "lon": 89.7000},
    {"name": "Jamalpur Sadar",  "lat": 24.9000, "lon": 89.9333},
]

API_URL = "https://flood-api.open-meteo.com/v1/flood"

# If current discharge > 1.5x past-7-day average → flag as HIGH
HIGH_MULTIPLIER = 1.5


def fetch_zone(zone: dict) -> dict | None:
    """Fetch 7-day forecast + past 7 days for a single zone."""
    params = {
        "latitude":      zone["lat"],
        "longitude":     zone["lon"],
        "daily":         "river_discharge",   # FIX: only this is valid without ensemble
        "past_days":     7,
        "forecast_days": 7,
        # FIX: do NOT pass "models" — causes 400 on some coordinates
    }

    try:
        response = requests.get(API_URL, params=params, timeout=15)

        # Print raw error body to help debug future issues
        if response.status_code != 200:
            print(f"  ❌ HTTP {response.status_code}: {response.text[:200]}")
            return None

        return response.json()

    except requests.exceptions.ConnectionError:
        print("  ❌ Connection failed — check internet connection")
        return None
    except requests.exceptions.Timeout:
        print("  ❌ Request timed out after 15s")
        return None
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return None


def analyse_zone(zone: dict, data: dict) -> None:
    """Print discharge table and flag elevated levels."""
    daily     = data.get("daily", {})
    times     = daily.get("time", [])
    discharge = daily.get("river_discharge", [])

    if not times or not discharge:
        print("  ⚠️  No discharge data — river may be outside the 5 km grid cell.")
        print("      Try shifting lat/lon by ±0.1° to find the nearest river pixel.")
        return

    # Grid cell actually used (may differ slightly from requested coords)
    actual_lat = data.get("latitude",  zone["lat"])
    actual_lon = data.get("longitude", zone["lon"])
    print(f"  Grid cell: ({actual_lat:.4f}, {actual_lon:.4f})")

    today       = date.today().isoformat()
    past_vals   = []
    today_val   = None
    future_vals = []

    for t, d in zip(times, discharge):
        if d is None:
            continue
        if t < today:
            past_vals.append((t, d))
        elif t == today:
            today_val = (t, d)
        else:
            future_vals.append((t, d))

    # Table
    print(f"  {'Date':<12} {'Discharge (m³/s)':>18}  Note")
    print(f"  {'-'*12} {'-'*18}  {'-'*18}")
    for t, d in past_vals:
        print(f"  {t:<12} {d:>18.1f}")
    if today_val:
        t, d = today_val
        print(f"  {t:<12} {d:>18.1f}  ◀ TODAY")
    for t, d in future_vals:
        print(f"  {t:<12} {d:>18.1f}  (forecast)")

    # Risk flag
    if today_val and past_vals:
        past_avg = sum(v for _, v in past_vals) / len(past_vals)
        ratio    = today_val[1] / past_avg if past_avg > 0 else 0
        if ratio >= HIGH_MULTIPLIER:
            print(f"\n  ⚠️  ELEVATED — today is {ratio:.1f}x past-7-day avg ({past_avg:.0f} m³/s)")
        else:
            print(f"\n  ✅ Normal — {ratio:.2f}x past-7-day avg ({past_avg:.0f} m³/s)")

    if future_vals:
        peak_date, peak_val = max(future_vals, key=lambda x: x[1])
        print(f"  📈 7-day forecast peak: {peak_val:.1f} m³/s on {peak_date}")


def main():
    print("=" * 62)
    print("  GloFAS v4 River Discharge — Bangladesh Sentinel Zones")
    print(f"  API: {API_URL}")
    print("=" * 62)

    passed = 0

    for zone in ZONES:
        print(f"\n🌊 {zone['name']}  ({zone['lat']}, {zone['lon']})")
        data = fetch_zone(zone)

        if data is None:
            continue

        if data.get("error"):
            print(f"  ❌ API error: {data.get('reason', 'unknown')}")
            continue

        analyse_zone(zone, data)
        passed += 1

    print("\n" + "=" * 62)
    print(f"  {passed}/{len(ZONES)} zones succeeded.")
    if passed == len(ZONES):
        print("  ✅ GloFAS API working. Ready to integrate into agent.")
    elif passed > 0:
        print("  ⚠️  Partial success. Zones with no data may need")
        print("      coordinate adjustment (shift ±0.1°).")
    else:
        print("  ❌ All zones failed. Check internet / firewall.")
    print("=" * 62)


if __name__ == "__main__":
    main()