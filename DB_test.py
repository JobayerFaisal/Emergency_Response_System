import asyncio
import asyncpg
import sys
import os

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/disaster_response"

SQL_FILES = [
    "backend/database/003_resource_schema.sql",
    "backend/database/004_dispatch_schema.sql",
    "backend/database/005_agent_messages.sql",
]

async def run():
    print("Connecting to database...")
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected\n")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    print("Enabling PostGIS...")
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology;")
        print("✅ PostGIS enabled\n")
    except Exception as e:
        print(f"⚠️  {e}\n")

    try:
        await conn.fetchval("SELECT ST_AsText(ST_GeomFromText('POINT(0 0)', 4326))")
        print("✅ PostGIS geography type working\n")
    except Exception as e:
        print(f"❌ PostGIS not working: {e}")
        print("Run this in psql as postgres user: CREATE EXTENSION postgis;")
        await conn.close()
        sys.exit(1)

    for filepath in SQL_FILES:
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath} — run from D:\\Emergency_Response_System")
            continue
        try:
            with open(filepath) as f:
                sql = f.read()
            await conn.execute(sql)
            print(f"✅ Applied: {filepath}")
        except Exception as e:
            print(f"❌ Error in {filepath}: {e}")

    await conn.close()
    print("\nAll done! You can now start Agent 3.")

asyncio.run(run())