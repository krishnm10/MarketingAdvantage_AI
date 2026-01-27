# check_specific_ids.py
import asyncio
import json
from app.db.session_v2 import get_async_session
from sqlalchemy import text

async def check_ids():
    async with get_async_session() as db:
        # Check the IDs that are showing tap_trust=0.0
        ids_to_check = [
            'ae2a9f0e-ae46-4a0d-9a92-34eaeb9c1ef9',
            'e09c5b69-fc11-4a2f-9eaf-261433ed421b',
            'c3a7e3c7-d0a4-4141-9b1e-2274eeb1c946',
        ]
        
        for content_id in ids_to_check:
            result = await db.execute(text("""
                SELECT 
                    id,
                    validation_layer
                FROM ingested_content
                WHERE id = :id
            """), {"id": content_id})
            
            row = result.fetchone()
            
            if row:
                print("="*80)
                print(f"ID: {row[0]}")
                print("="*80)
                
                if row[1]:
                    # Extract tap_trust_score from latest snapshot
                    snapshots = row[1] if isinstance(row[1], list) else [row[1]]
                    
                    print(f"Total snapshots: {len(snapshots)}")
                    
                    for idx, snapshot in enumerate(snapshots):
                        tap_trust = snapshot.get('tap_trust_score', 'NOT FOUND')
                        method = snapshot.get('method', 'N/A')
                        print(f"  Snapshot {idx + 1}: method='{method}', tap_trust_score={tap_trust}")
                    
                    # What does _normalize_signal return?
                    latest = snapshots[-1] if snapshots else {}
                    print(f"\nLatest snapshot (what repository gets):")
                    print(f"  Method: {latest.get('method', 'N/A')}")
                    print(f"  tap_trust_score: {latest.get('tap_trust_score', 'NOT FOUND')}")
                else:
                    print("NULL validation_layer")
            else:
                print(f"ID {content_id} not found!")
            
            print()

asyncio.run(check_ids())
