# app/api/v2/ingestion_sync_api.py

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.auth.guards import require_role

from app.db.session_v2 import get_db
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.services.ingestion.ingestion_service_v2 import (
    get_chroma_collection,
    get_embedder,
)

router = APIRouter(
    prefix="/api/v2/ingestion-admin/sync",
    tags=["Ingestion Sync"],
)

# -----------------------------------------------------------
# DETECT ORPHANS (READ-ONLY)
# -----------------------------------------------------------
@router.get("/orphans")
async def detect_orphans(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin", "viewer")),
):
    """
    Detects Postgres ↔ Chroma drift.
    Read-only. No mutations.
    """

    # 1️⃣ Get all semantic_hashes from DB
    result = await db.execute(
        select(GlobalContentIndexV2.semantic_hash)
    )
    db_hashes = {row[0] for row in result.all()}

    # 2️⃣ Get all vector IDs from Chroma
    _, collection = get_chroma_collection()

    try:
        chroma_data = collection.get(include=[])
        chroma_ids = set(chroma_data.get("ids", []))
    except Exception:
        chroma_ids = set()

    # 3️⃣ Diff
    db_without_chroma = sorted(db_hashes - chroma_ids)
    chroma_without_db = sorted(chroma_ids - db_hashes)

    return {
        "db_without_chroma": db_without_chroma,
        "chroma_without_db": chroma_without_db,
        "counts": {
            "db_total": len(db_hashes),
            "chroma_total": len(chroma_ids),
            "db_orphans": len(db_without_chroma),
            "chroma_orphans": len(chroma_without_db),
        },
    }


# -----------------------------------------------------------
# FIX: CHROMA → DB (DELETE ORPHAN VECTORS)
# -----------------------------------------------------------
@router.post("/fix/chroma-to-db")
async def fix_chroma_to_db(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin")),
):
    result = await db.execute(
        select(GlobalContentIndexV2.semantic_hash)
    )
    db_hashes = {row[0] for row in result.all()}

    _, collection = get_chroma_collection()
    chroma_data = collection.get(include=[])
    chroma_ids = set(chroma_data.get("ids", []))

    orphans = chroma_ids - db_hashes

    if orphans:
        collection.delete(ids=list(orphans))

    return {
        "status": "ok",
        "deleted_vectors": len(orphans),
    }


# -----------------------------------------------------------
# FIX: DB → CHROMA (RE-EMBED MISSING VECTORS)
# -----------------------------------------------------------
@router.post("/fix/db-to-chroma")
async def fix_db_to_chroma(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin")),
):
    """
    Re-embeds semantic hashes that exist in DB
    but are missing from Chroma.
    SAFE & IDEMPOTENT.
    """

    result = await db.execute(
        select(
            GlobalContentIndexV2.semantic_hash,
            GlobalContentIndexV2.cleaned_text,
        )
    )
    rows = result.all()

    _, collection = get_chroma_collection()

    try:
        chroma_data = collection.get(include=[])
        chroma_ids = set(chroma_data.get("ids", []))
    except Exception:
        chroma_ids = set()

    embedder = get_embedder()
    reembedded = 0

    for semantic_hash, cleaned_text in rows:
        if not semantic_hash or not cleaned_text:
            continue

        if semantic_hash in chroma_ids:
            continue

        vector = embedder.encode(
            cleaned_text,
            normalize_embeddings=True,
        ).tolist()

        collection.upsert(
            ids=[semantic_hash],
            embeddings=[vector],
            documents=[cleaned_text],
            metadatas=[{"repair_source": "db_to_chroma"}],
        )

        reembedded += 1

    return {
        "status": "ok",
        "reembedded": reembedded,
    }

# -----------------------------------------------------------
# FIX: ORPHANS (DB ↔ CHROMA FULL CLEANUP)
# -----------------------------------------------------------
@router.post("/orphans")
async def cleanup_orphans(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_role("admin")),
):
    """
    Performs cleanup of both DB and Chroma orphans.
    Combines detection and cleanup logic.
    """

    try:
        # STEP 1 — Detect
        result = await db.execute(select(GlobalContentIndexV2.semantic_hash))
        db_hashes = {row[0] for row in result.all()}

        _, collection = get_chroma_collection()
        chroma_data = collection.get(include=[])
        chroma_ids = set(chroma_data.get("ids", []))

        db_orphans = db_hashes - chroma_ids
        chroma_orphans = chroma_ids - db_hashes

        # STEP 2 — Delete from Chroma
        if chroma_orphans:
            collection.delete(ids=list(chroma_orphans))

        # STEP 3 — Optionally delete DB orphans (if needed)
        # Currently we skip DB deletions for safety.
        # Uncomment below if you want DB cleanups:
        # await db.execute(delete(GlobalContentIndexV2).where(GlobalContentIndexV2.semantic_hash.in_(db_orphans)))

        await db.commit()

        return {
            "status": "ok",
            "db_orphans_found": len(db_orphans),
            "chroma_orphans_deleted": len(chroma_orphans),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))