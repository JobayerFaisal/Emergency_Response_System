import asyncpg
import asyncio

DATABASE_URL = "postgresql://postgres:postgres@db:5432/disaster_db"  # asyncpg format

# 👉 Change table name here if needed
TABLE_NAME = "chat_history"

# TABLE_NAME = "emergency_reports"


async def test_connection():
    try:
        print(f"Connecting to database: {DATABASE_URL}")
        conn = await asyncpg.connect(DATABASE_URL)
        print("✅ Connected successfully!")

        # Fetch rows
        rows = await conn.fetch(f"SELECT * FROM {TABLE_NAME} LIMIT 50;")
        print(f"\nFetched {len(rows)} rows from '{TABLE_NAME}' table:\n")

        for row in rows:
            print(dict(row))

        await conn.close()
        print("\n🔌 Connection closed.")

    except Exception as e:
        print("❌ Error:", e)


if __name__ == "__main__":
    asyncio.run(test_connection())


# docker exec -it disaster_backend bash
# python app/test3.py

