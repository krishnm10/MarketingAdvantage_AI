# check_validation_schema.py (FIXED)
import asyncio
import json
from app.db.session_v2 import get_async_session
from sqlalchemy import text

async def check_validation():
    async with get_async_session() as db:
        # ✅ FIXED: Use correct table name
        result = await db.execute(text("""
            SELECT 
                id,
                validation_layer,
                reasoning_ingestion
            FROM ingested_content
            WHERE validation_layer IS NOT NULL
            LIMIT 1
        """))
        
        row = result.fetchone()
        
        if row:
            print("="*80)
            print(f"CONTENT ID: {row[0]}")
            print("="*80)
            
            print("\nVALIDATION LAYER (validation_layer column):")
            print("="*80)
            if row[1]:
                print(json.dumps(row[1], indent=2))
            else:
                print("NULL")
            
            print("\n" + "="*80)
            print("REASONING INGESTION (reasoning_ingestion column):")
            print("="*80)
            if row[2]:
                print(json.dumps(row[2], indent=2))
            else:
                print("NULL")
        else:
            print("❌ No records with validation_layer found!")
            
            # Check if ANY records exist
            count_result = await db.execute(text(
                "SELECT COUNT(*) FROM ingested_content"
            ))
            total = count_result.scalar()
            print(f"Total records in database: {total}")
            
            if total > 0:
                # Show what's in random records
                result = await db.execute(text("""
                    SELECT 
                        id, 
                        validation_layer IS NULL as val_null,
                        reasoning_ingestion IS NULL as reason_null
                    FROM ingested_content 
                    LIMIT 5
                """))
                print("\nFirst 5 records:")
                for r in result:
                    print(f"  ID: {r[0]}")
                    print(f"    validation_layer is NULL: {r[1]}")
                    print(f"    reasoning_ingestion is NULL: {r[2]}")

asyncio.run(check_validation())
