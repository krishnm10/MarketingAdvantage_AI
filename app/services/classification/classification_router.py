# services/classification/classification_router.py

import asyncio
from typing import Dict, Any, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models.ingested_content import IngestedContent
from app.services.classification.classification_service import ClassificationService
from app.utils.logger import log_info, log_warning


class ClassificationRouter:

    @staticmethod
    async def classify_file(
        db: AsyncSession,
        file_id: str,
        parallel: bool = True,
        max_concurrency: int = 4
    ) -> Dict[str, Any]:
        """
        Classifies ALL chunks belonging to a given ingested_file.

        Inputs:
            file_id: UUID of ingested_file
            parallel: run concurrent classification
            max_concurrency: concurrency cap

        Output:
            {
                "file_id": ...,
                "total_chunks": ...,
                "classified": [...],
                "pending_taxonomies": [...]
            }
        """

        log_info(f"[classification_router] Starting classification for file {file_id}")

        # -------------------------------------------------
        # 1. Load chunks
        # -------------------------------------------------
        result = await db.execute(
            select(IngestedContent).where(IngestedContent.file_id == file_id)
        )
        chunks = result.scalars().all()

        if not chunks:
            raise ValueError(f"[classification_router] No chunks found for file {file_id}")

        log_info(f"[classification_router] Found {len(chunks)} chunks")

        # -------------------------------------------------
        # 2. Prepare tasks
        # -------------------------------------------------
        async def classify_one(chunk):
            try:
                return await ClassificationService.classify_chunk(
                    db=db,
                    chunk={
                        "id": str(chunk.id),
                        "cleaned_text": chunk.cleaned_text,
                        "text": chunk.text,
                        "source_type": chunk.source_type
                    }
                )
            except Exception as e:
                log_warning(f"[classification_router] Chunk {chunk.id} failed: {e}")
                return {
                    "error": str(e),
                    "chunk_id": str(chunk.id)
                }

        results: List[Dict[str, Any]] = []

        # -------------------------------------------------
        # 3. Run classification in parallel
        # -------------------------------------------------
        if parallel:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def sem_task(chunk):
                async with semaphore:
                    return await classify_one(chunk)

            tasks = [sem_task(c) for c in chunks]
            results = await asyncio.gather(*tasks)

        else:
            # Sequential mode (slower but safer)
            for c in chunks:
                results.append(await classify_one(c))

        # -------------------------------------------------
        # 4. Extract pending taxonomy IDs
        # -------------------------------------------------
        pending_list = [
            r["pending_taxonomy_id"]
            for r in results
            if isinstance(r, dict) and r.get("pending_taxonomy_id") is not None
        ]

        return {
            "file_id": file_id,
            "total_chunks": len(chunks),
            "classified": results,
            "pending_taxonomies": pending_list
        }
