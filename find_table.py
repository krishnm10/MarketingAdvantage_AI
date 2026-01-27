# find_table.py
import asyncio
from app.db.session_v2 import get_async_session
from sqlalchemy import text

async def find_table():
    async with get_async_session() as db:
        # List all tables
        result = await db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        
        tables = [row[0] for row in result]
        
        print("="*80)
        print("ALL TABLES IN DATABASE:")
        print("="*80)
        for table in tables:
            print(f"  • {table}")
        
        print("\n" + "="*80)
        print("LOOKING FOR INGESTION TABLE:")
        print("="*80)
        
        # Find tables with 'ingest' in name
        ingestion_tables = [t for t in tables if 'ingest' in t.lower()]
        
        if ingestion_tables:
            print("Found ingestion tables:")
            for table in ingestion_tables:
                print(f"  ✅ {table}")
                
                # Check first row
                try:
                    result = await db.execute(text(f"""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = '{table}'
                        ORDER BY ordinal_position
                    """))
                    
                    print(f"\n  Columns in {table}:")
                    for col in result:
                        print(f"    - {col[0]} ({col[1]})")
                except:
                    pass
        else:
            print("❌ No tables with 'ingest' in name found!")
            print("\nAll tables:")
            for table in tables:
                print(f"  • {table}")

asyncio.run(find_table())
