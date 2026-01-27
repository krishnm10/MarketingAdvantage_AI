"""
verify_ingestion_integrity_v2.py — Cross-checks PostgreSQL vs ChromaDB after ingestion
Supports all file types: DOCX, PDF, JSON, XML, CSV, EXCEL, TEXT, RSS, API, WEB.
"""

import asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import select
import chromadb

from app.db.session_v2 import async_engine
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.utils.logger import log_info, log_warning

# ✅ Ensure relationship import ordering doesn't break ORM
# If IngestedContentV2 refers to GlobalContentIndexV2, that class must be imported AFTER model definitions
try:
    from app.db.models.global_content_index_v2 import GlobalContentIndexV2
except ImportError:
    GlobalContentIndexV2 = None
    log_warning("[IntegrityCheck] Warning: GlobalContentIndexV2 not imported, continuing without direct join checks.")

# ---------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------
CHROMA_PATH = "./chroma_db"
CHROMA = chromadb.PersistentClient(path=CHROMA_PATH)
COLLECTION = CHROMA.get_or_create_collection(name="ingested_content")
async_session = async_sessionmaker(async_engine, expire_on_commit=False)

# ---------------------------------------------------------
# MAIN VERIFICATION LOGIC
# ---------------------------------------------------------
async def verify_ingestion_integrity():
    async with async_session() as db:
        try:
            log_info("[IntegrityCheck] Starting verification...")

            # Step 1: Retrieve all ingested files
            files = (await db.execute(select(IngestedFileV2))).scalars().all()
            total_files = len(files)
            log_info(f"[IntegrityCheck] Found {total_files} files in PostgreSQL")

            # Step 2: Count vectors in ChromaDB
            chroma_data = COLLECTION.get(include=["metadatas"])
            chroma_vectors = chroma_data.get("metadatas", [])
            chroma_count = len(chroma_vectors)
            log_info(f"[IntegrityCheck] Found {chroma_count} vectors in ChromaDB")

            # Step 3: Verify each file’s content chunks
            for file in files:
                file_id = str(file.id)
                file_type = file.file_type or "unknown"

                chunks = (
                    await db.execute(
                        select(IngestedContentV2).where(IngestedContentV2.file_id == file.id)
                    )
                ).scalars().all()
                pg_count = len(chunks)

                matching_chroma = [
                    meta for meta in chroma_vectors if meta and meta.get("file_id") == file_id
                ]
                chroma_match_count = len(matching_chroma)

                # Consistency check
                if pg_count == chroma_match_count and pg_count > 0:
                    log_info(f"[OK] {file_type.upper()} | File {file_id} | PG={pg_count}, Chroma={chroma_match_count}")
                elif pg_count == 0:
                    log_warning(f"[WARN] {file_type.upper()} | File {file_id} has no content in PG.")
                elif chroma_match_count == 0:
                    log_warning(f"[WARN] {file_type.upper()} | File {file_id} missing vectors in Chroma.")
                else:
                    log_warning(f"[Mismatch] {file_type.upper()} | File {file_id}: PG={pg_count}, Chroma={chroma_match_count}")

            log_info("[IntegrityCheck] ✅ Verification complete — all file types checked.")

        except Exception as e:
            log_warning(f"[IntegrityCheck] ❌ Failed: {e}")

# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(verify_ingestion_integrity())
