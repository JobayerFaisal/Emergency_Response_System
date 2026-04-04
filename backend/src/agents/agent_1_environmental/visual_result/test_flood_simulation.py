import asyncpg
import asyncio
from uuid import uuid4
from datetime import datetime, timezone

async def inject_high_risk_test():
    """Simulate a HIGH RISK flood scenario for testing"""
    
    conn = await asyncpg.connect(
        'postgresql://postgres:postgres@localhost:5432/disaster_response'
    )
    
    # Get a zone (Mirpur - known flood-prone area)
    zone = await conn.fetchrow(
        "SELECT id, name FROM sentinel_zones WHERE name = 'Mirpur' LIMIT 1"
    )
    
    if not zone:
        print("❌ Zone not found")
        return
    
    # Create HIGH RISK test prediction
    test_prediction = {
        'id': str(uuid4()),
        'zone_id': zone['id'],
        'timestamp': datetime,
        'risk_score': 0.72,  # 72% - HIGH RISK
        'severity_level': 'high',
        'confidence': 0.85,
        'time_to_impact_hours': 3.0,
        'affected_area_km2': 8.5,
        'risk_factors': '''{
            "rainfall_intensity": 0.85,
            "accumulated_rainfall": 0.90,
            "weather_severity": 0.75,
            "social_reports_density": 0.65,
            "historical_risk": 0.70,
            "drainage_factor": 0.80,
            "elevation_factor": 0.90
        }''',
        'recommended_actions': '["Evacuate low-lying areas", "Close schools and offices", "Deploy emergency teams", "Open shelters", "Monitor water levels"]'
    }
    
    # Insert test prediction
    await conn.execute("""
        INSERT INTO flood_predictions (
            id, zone_id, timestamp, risk_score, severity_level,
            confidence, time_to_impact_hours, affected_area_km2,
            risk_factors, recommended_actions
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    """,
        test_prediction['id'],
        test_prediction['zone_id'],
        test_prediction['timestamp'],
        test_prediction['risk_score'],
        test_prediction['severity_level'],
        test_prediction['confidence'],
        test_prediction['time_to_impact_hours'],
        test_prediction['affected_area_km2'],
        test_prediction['risk_factors'],
        test_prediction['recommended_actions']
    )
    
    await conn.close()
    
    print("\n" + "="*80)
    print("✅ HIGH RISK TEST PREDICTION INJECTED!")
    print("="*80)
    print(f"   Zone: {zone['name']}")
    print(f"   Risk: 72% (HIGH)")
    print(f"   Severity: high")
    print(f"   Time to Impact: 3 hours")
    print(f"   Affected Area: 8.5 km²")
    print(f"   Confidence: 85%")
    print("="*80)
    print("\nNow run: python view_all_predictions_fixed.py")

# Run
asyncio.run(inject_high_risk_test())