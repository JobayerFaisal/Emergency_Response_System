import asyncpg
import asyncio

async def check_schema():
    conn = await asyncpg.connect(
        'postgresql://postgres:postgres@localhost:5432/disaster_response'
    )
    
    # Get table structure
    query = """
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'flood_predictions'
    ORDER BY ordinal_position;
    """
    
    columns = await conn.fetch(query)
    await conn.close()
    
    print("\n" + "="*80)
    print("FLOOD_PREDICTIONS TABLE STRUCTURE")
    print("="*80)
    for col in columns:
        print(f"  {col['column_name']:30} {col['data_type']}")
    print("="*80)

asyncio.run(check_schema())