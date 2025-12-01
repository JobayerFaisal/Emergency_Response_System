import pandas as pd
import asyncpg
import asyncio
import json

async def get_predictions():
    # Connect to database
    conn = await asyncpg.connect(
        'postgresql://postgres:postgres@localhost:5432/disaster_db'
    )
    
    # Query predictions
    query = """
    SELECT 
        fp.timestamp,
        sz.name as zone,
        sz.latitude,
        sz.longitude,
        fp.risk_score,
        fp.severity_level,
        fp.confidence,
        fp.time_to_impact_hours,
        fp.affected_area_km2,
        fp.estimated_affected_population,
        fp.risk_factors,
        fp.recommended_actions
    FROM flood_predictions fp
    JOIN sentinel_zones sz ON fp.zone_id = sz.id
    ORDER BY fp.timestamp DESC
    LIMIT 20;
    """
    
    rows = await conn.fetch(query)
    await conn.close()
    
    # Convert to DataFrame
    df = pd.DataFrame([dict(row) for row in rows])
    
    # Format columns
    if not df.empty:
        df['risk_score'] = df['risk_score'].apply(lambda x: f"{x:.2%}")
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
        df['affected_area_km2'] = df['affected_area_km2'].apply(lambda x: f"{x:.2f}")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Parse JSON columns
        df['risk_factors'] = df['risk_factors'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        df['recommended_actions'] = df['recommended_actions'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    return df

# Run
df = asyncio.run(get_predictions())

print("\n" + "="*120)
print("FLOOD PREDICTIONS FROM DATABASE (Last 20)")
print("="*120)
print(df.to_string(index=False))
print("="*120)
print(f"\nTotal Records: {len(df)}")