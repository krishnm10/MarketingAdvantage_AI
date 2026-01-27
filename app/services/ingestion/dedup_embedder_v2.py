# =============================================
# Deduplication-aware Embedding and Chroma Storage
# =============================================

import uuid
import re
from sqlalchemy import select
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.utils.logger import log_info


class IngestionDedupEmbedder:

    @staticmethod
    async def embed_and_store_dedup(file_id, business_id, file_type, chunks, db, EMBEDDER, COLLECTION, BATCH_SIZE=256):
        try:
            # Normalize text and filter duplicates globally
            clean_chunks = []
            for c in chunks:
                text = re.sub(r'\s*---(BLOCK|ENTRY) BREAK---\s*', '\n\n', c.get("cleaned_text", c.get("cleaned", "")))
                semantic_hash = c.get("semantic_hash")

                # Check in Postgres global_content_index
                result = await db.execute(select(GlobalContentIndexV2.id).where(GlobalContentIndexV2.semantic_hash == semantic_hash))
                existing_pg = result.scalar_one_or_none()

                # Check in ChromaDB (metadata search by semantic_hash)
                existing_chroma = COLLECTION.get(where={"semantic_hash": semantic_hash})

                if existing_pg or (existing_chroma and len(existing_chroma.get("ids", [])) > 0):
                    log_info(f"[EmbedDedup] Skipping existing semantic_hash: {semantic_hash}")
                    continue

                clean_chunks.append({
                    "text": text,
                    "semantic_hash": semantic_hash,
                    "source_type": c.get("source_type"),
                    "meta_data": c.get("metadata", {}),
                })

            if not clean_chunks:
                log_info(f"[EmbedDedup] No new chunks to embed for file {file_id}")
                return

            # Prepare embeddings
            texts = [c["text"] for c in clean_chunks]
            ids = [str(uuid.uuid4()) for _ in clean_chunks]

            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                batch_ids = ids[i:i + BATCH_SIZE]
                embeddings = EMBEDDER.encode(batch, normalize_embeddings=True)

                COLLECTION.upsert(
                    ids=batch_ids,
                    embeddings=embeddings.tolist(),
                    metadatas=[
                        {
                            "file_id": str(file_id),
                            "business_id": str(business_id) if business_id else None,
                            "source_type": file_type,
                            "semantic_hash": clean_chunks[j]["semantic_hash"]
                        }
                        for j in range(i, i + len(batch))
                    ],
                    documents=batch,
                )

            log_info(f"[EmbedDedup] Stored {len(texts)} new unique vectors in ChromaDB")

        except Exception as e:
            log_info(f"[ERROR] Embedding or Chroma dedup storage failed: {e}")
