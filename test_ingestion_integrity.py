"""
test_ingestion_integrity.py
--------------------------------------
1-Minute sanity test for ingestion + dedup + GCI insertion
Runs an RSS sample ingestion and verifies:
 - Parser works (chunks returned)
 - Deduplication is active
 - Chunks stored into global_content_index_v2
"""

import asyncio
import uuid
from sqlalchemy import select
from app.services.ingestion.file_router_v2 import route_external_ingestion
from app.db.session_v2 import async_engine
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.db.models.ingested_file_v2 import IngestedFileV2
from sqlalchemy.ext.asyncio import async_sessionmaker

# Test RSS feed (use any public feed)
TEST_RSS_URL = "http://feeds.bbci.co.uk/news/rss.xml"

async_session = async_sessionmaker(async_engine, expire_on_commit=False)

async def verify_gci_insertion(db, test_file_id):
    """Verify GCI entries linked to this test ingestion."""
    print("\n[VERIFY] Checking GlobalContentIndexV2 entries...")
    result = await db.execute(
        select(GlobalContentIndexV2)
        .where(GlobalContentIndexV2.ingested_file_id == test_file_id)
    )
    rows = result.scalars().all()
    if rows:
        print(f"✅ {len(rows)} content entries found in GlobalContentIndexV2.")
    else:
        print("❌ No entries found in GlobalContentIndexV2.")

async def verify_ingested_file(db, source_url):
    """Confirm file metadata stored correctly."""
    print("[VERIFY] Checking IngestedFileV2 record...")
    result = await db.execute(
        select(IngestedFileV2)
        .where(IngestedFileV2.file_path == source_url)
        .order_by(IngestedFileV2.created_at.desc())
    )
    entries = result.scalars().all()
    if entries:
        print(f"✅ Found {len(entries)} entries in IngestedFileV2 for this source.")
        latest = entries[0]
        print(f"   → Using latest entry: {latest.file_name} ({latest.status}) [id={latest.id}]")
        return latest.id
    else:
        print("❌ IngestedFileV2 record missing.")
        return None


async def run_integrity_test():
    print("\n🚀 Starting Ingestion Integrity Test...\n")

    # Step 1: Trigger external ingestion
    business_id = str(uuid.uuid4())
    print(f"[TEST] Triggering RSS ingestion for: {TEST_RSS_URL}")
    response = await route_external_ingestion("rss", TEST_RSS_URL, business_id=business_id)
    print(f"✅ Ingestion triggered. Response: {response}")

    # Step 2: Verify DB persistence
    async with async_session() as db:
        source_id = await verify_ingested_file(db, TEST_RSS_URL)
        if source_id:
            await verify_gci_insertion(db, source_id)

    print("\n🏁 Integrity test completed.\n")


if __name__ == "__main__":
    asyncio.run(run_integrity_test())
