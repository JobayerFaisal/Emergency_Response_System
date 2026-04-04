import pandas as pd
import asyncpg
import asyncio

async def get_all_predictions():
    conn = await asyncpg.connect(
        'postgresql://postgres:postgres@localhost:5432/disaster_response'
    )
    
    query = """
    SELECT 
        fp.timestamp,
        sz.name as zone,
        CAST(fp.risk_score * 100 AS NUMERIC(5,1)) as risk_pct,
        fp.severity_level,
        CAST(fp.confidence * 100 AS NUMERIC(5,1)) as confidence_pct,
        fp.time_to_impact_hours,
        CAST(fp.affected_area_km2 AS NUMERIC(10,2)) as area_km2
    FROM flood_predictions fp
    JOIN sentinel_zones sz ON fp.zone_id = sz.id
    ORDER BY fp.timestamp DESC
    LIMIT 50;
    """
    
    rows = await conn.fetch(query)
    await conn.close()
    
    df = pd.DataFrame([dict(row) for row in rows])
    return df

# Run
df = asyncio.run(get_all_predictions())

print("\n" + "="*100)
print("ALL FLOOD PREDICTIONS (Last 50 Records)")
print("="*100)
print(df.to_string(index=False))
print("="*100)
print(f"\nTotal Records: {len(df)}")

if not df.empty:
    print(f"\n📊 Risk Distribution:")
    severity_counts = df.groupby('severity_level').size()
    for severity, count in severity_counts.items():
        print(f"   {str(severity).upper():10} {count:3} zones")
    
    print(f"\n📈 Risk Statistics:")
    print(f"   Highest Risk: {df['risk_pct'].max():.1f}%")
    print(f"   Average Risk: {df['risk_pct'].mean():.1f}%")
    print(f"   Lowest Risk: {df['risk_pct'].min():.1f}%")
    
    # Show HIGH risk entries
    high_risk = df[df['severity_level'].isin(['high', 'critical'])]
    if not high_risk.empty:
        print(f"\n🚨 HIGH/CRITICAL RISK ZONES:")
        print(high_risk[['zone', 'risk_pct', 'severity_level', 'time_to_impact_hours']].to_string(index=False))