import asyncio
import os
from sqlalchemy import select, delete
from app.db.session_v2 import async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2

async def cleanup_and_reingest(filename: str):
    async_session = async_sessionmaker(async_engine, expire_on_commit=False)
    async with async_session() as db:
        print(f"🔍 Looking for file: {filename}")
        
        # Find the file
        result = await db.execute(
            select(IngestedFileV2).where(IngestedFileV2.file_name.like(f"%{filename}%"))
        )
        files = result.scalars().all()
        
        if not files:
            print(f"❌ No files found matching: {filename}")
            print("✅ File is clean, ready for fresh ingestion")
            return
        
        print(f"📋 Found {len(files)} file(s) to clean up:")
        
        for file_record in files:
            file_id = file_record.id
            print(f"\n  🗑️  Deleting: {file_record.file_name}")
            print(f"      ID: {file_id}")
            print(f"      Status: {file_record.status}")
            print(f"      Chunks: {file_record.total_chunks}")
            
            # Delete content chunks
            content_result = await db.execute(
                select(IngestedContentV2).where(IngestedContentV2.file_id == file_id)
            )
            content_chunks = content_result.scalars().all()
            
            # Collect semantic hashes for GCI cleanup
            semantic_hashes = [c.semantic_hash for c in content_chunks if c.semantic_hash]
            
            # Delete chunks
            await db.execute(
                delete(IngestedContentV2).where(IngestedContentV2.file_id == file_id)
            )
            print(f"      ✅ Deleted {len(content_chunks)} content chunks")
            
            # Delete file record
            await db.execute(
                delete(IngestedFileV2).where(IngestedFileV2.id == file_id)
            )
            print(f"      ✅ Deleted file record")
            
            # Optional: Clean up orphaned GCI entries
            if semantic_hashes:
                # Check which hashes are still used by other files
                still_used = await db.execute(
                    select(IngestedContentV2.semantic_hash)
                    .where(IngestedContentV2.semantic_hash.in_(semantic_hashes))
                    .distinct()
                )
                still_used_set = set(still_used.scalars().all())
                orphaned = [h for h in semantic_hashes if h not in still_used_set]
                
                if orphaned:
                    await db.execute(
                        delete(GlobalContentIndexV2)
                        .where(GlobalContentIndexV2.semantic_hash.in_(orphaned))
                    )
                    print(f"      🧹 Cleaned {len(orphaned)} orphaned GCI entries")
        
        await db.commit()
        print(f"\n✅ Cleanup complete! Ready for re-ingestion.")
        print(f"💡 Now copy {filename} to ./static/uploads/manual/")

if __name__ == "__main__":
    asyncio.run(cleanup_and_reingest("Crisil.pdf"))
