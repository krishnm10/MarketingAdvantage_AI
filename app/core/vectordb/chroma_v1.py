# ============================================================
# app/core/vectordb/chroma_v1.py
#
# WHAT THIS FILE IS:
#   ChromaDB implementation of BaseVectorDB.
#   This is the ONLY file that imports chromadb.
#   All other files use BaseVectorDB only.
#
# HOW IT'S REGISTERED:
#   app/core/vectordb/register.py calls:
#   vectordb_registry.register("chroma", ChromaVectorDB)
# ============================================================

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from app.core.vectordb.base import BaseVectorDB, BatchUpsertResult, VectorHit

logger = logging.getLogger(__name__)


class ChromaVectorDB(BaseVectorDB):
    """
    ChromaDB connector implementing BaseVectorDB.

    Supports both:
    - PersistentClient: data saved to disk (production)
    - EphemeralClient:  in-memory only (testing)

    Constructor args:
        persist_directory:     Path to save ChromaDB data (e.g. "./chroma_db")
                               If None, uses in-memory EphemeralClient.
        anonymized_telemetry:  Set False to disable ChromaDB telemetry (default: False)
    """

    def __init__(
        self,
        persist_directory: Optional[str] = "./chroma_db",
        anonymized_telemetry: bool = False,
    ):
        self._path = persist_directory
        self._client = self._make_client(persist_directory, anonymized_telemetry)
        # Cache open collections: {collection_name: Collection}
        self._collections: Dict[str, chromadb.Collection] = {}
        logger.info(
            "[ChromaVectorDB] Initialized | path=%s | telemetry=%s",
            persist_directory or "in-memory",
            anonymized_telemetry,
        )

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    def kind(self) -> str:
        return "chroma"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception as exc:
            logger.warning("[ChromaVectorDB] health_check failed: %s", exc)
            return False

    def ensure_collection(
        self,
        collection: str,
        *,
        embedding_dim: int,
        distance_metric: str = "cosine",
    ) -> None:
        """
        Create the ChromaDB collection if it doesn't exist.
        Chroma uses HNSW internally — we pass the distance metric via metadata.

        Note: embedding_dim is stored in metadata for introspection.
        Chroma does NOT enforce dimension at collection creation time.
        """
        _metric_map = {
            "cosine": "cosine",
            "dotproduct": "ip",       # inner product
            "euclidean": "l2",
        }
        hnsw_space = _metric_map.get(distance_metric, "cosine")

        col = self._client.get_or_create_collection(
            name=collection,
            metadata={
                "hnsw:space": hnsw_space,
                "embedding_dim": embedding_dim,
            },
        )
        self._collections[collection] = col
        logger.info(
            "[ChromaVectorDB] Collection '%s' ensured | dim=%d | metric=%s",
            collection, embedding_dim, hnsw_space,
        )

    def delete_collection(self, collection: str) -> None:
        try:
            self._client.delete_collection(name=collection)
            self._collections.pop(collection, None)
            logger.info("[ChromaVectorDB] Collection '%s' deleted.", collection)
        except Exception as exc:
            logger.warning(
                "[ChromaVectorDB] delete_collection '%s' failed: %s", collection, exc
            )

    # ── Write ─────────────────────────────────────────────────────────────

    def upsert(
        self,
        *,
        collection: str,
        doc_id: str,
        embedding: List[float],
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        col = self._get_collection(collection)
        # Chroma metadata values must be str | int | float | bool
        safe_meta = _sanitize_metadata(metadata)
        col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[safe_meta],
        )

    def batch_upsert(
        self,
        *,
        collection: str,
        doc_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> BatchUpsertResult:
        """
        Single Chroma API call for the entire batch.
        Chroma handles batching internally via HNSW.
        """
        if not doc_ids:
            return BatchUpsertResult()

        col = self._get_collection(collection)
        safe_metas = [_sanitize_metadata(m) for m in metadatas]

        # Check which IDs already exist so we can report inserted vs updated
        existing = set(self.exists(collection=collection, ids=doc_ids))
        inserted = sum(1 for i in doc_ids if i not in existing)
        updated = len(doc_ids) - inserted

        try:
            col.upsert(
                ids=doc_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=safe_metas,
            )
            logger.info(
                "[ChromaVectorDB] batch_upsert '%s': +%d new, ~%d updated",
                collection, inserted, updated,
            )
            return BatchUpsertResult(inserted=inserted, updated=updated)
        except Exception as exc:
            logger.error("[ChromaVectorDB] batch_upsert failed: %s", exc)
            return BatchUpsertResult(failed=len(doc_ids))

    # ── Read ──────────────────────────────────────────────────────────────

    def search(
        self,
        *,
        collection: str,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        col = self._get_collection(collection)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, col.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        try:
            res = col.query(**kwargs)
        except Exception as exc:
            logger.error("[ChromaVectorDB] search failed: %s", exc)
            return []

        hits: List[VectorHit] = []
        ids       = res.get("ids", [[]])[0]
        docs      = res.get("documents", [[]])[0]
        metas     = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            # Chroma cosine distance is 0=identical, 2=opposite
            # Convert to similarity score: 1 - (distance / 2)
            distance = distances[i] if i < len(distances) else 1.0
            score = max(0.0, 1.0 - (distance / 2.0))
            hits.append(VectorHit(
                id=doc_id,
                text=docs[i] if i < len(docs) else "",
                score=round(score, 6),
                metadata=metas[i] if i < len(metas) else {},
            ))

        return hits

    def exists(
        self,
        *,
        collection: str,
        ids: List[str],
    ) -> List[str]:
        """
        Returns the subset of IDs that already exist in ChromaDB.

        CRITICAL: Chroma's get() returns only the IDs that exist.
        If you pass ["a", "b", "c"] and only "a" and "c" exist,
        you get back ["a", "c"].
        """
        if not ids:
            return []
        col = self._get_collection(collection)
        try:
            # include=[] means: return only IDs, skip documents/embeddings/metadata
            # This is the fastest possible existence check.
            result = col.get(ids=ids, include=[])
            return result.get("ids", [])
        except Exception as exc:
            logger.warning("[ChromaVectorDB] exists() check failed: %s", exc)
            return []

    def get_by_ids(
        self,
        *,
        collection: str,
        ids: List[str],
    ) -> List[VectorHit]:
        if not ids:
            return []
        col = self._get_collection(collection)
        try:
            result = col.get(
                ids=ids,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("[ChromaVectorDB] get_by_ids() failed: %s", exc)
            return []

        hits = []
        for i, doc_id in enumerate(result.get("ids", [])):
            docs  = result.get("documents") or []
            metas = result.get("metadatas") or []
            hits.append(VectorHit(
                id=doc_id,
                text=docs[i] if i < len(docs) else "",
                score=1.0,   # No score for direct lookup
                metadata=metas[i] if i < len(metas) else {},
            ))
        return hits

    def count(self, collection: str) -> int:
        try:
            col = self._get_collection(collection)
            return col.count()
        except Exception as exc:
            logger.warning("[ChromaVectorDB] count() failed: %s", exc)
            return 0

    # ── Delete ────────────────────────────────────────────────────────────

    def delete(self, *, collection: str, doc_id: str) -> None:
        col = self._get_collection(collection)
        try:
            col.delete(ids=[doc_id])
        except Exception as exc:
            logger.warning("[ChromaVectorDB] delete(%s) failed: %s", doc_id, exc)

    def delete_many(self, *, collection: str, doc_ids: List[str]) -> int:
        if not doc_ids:
            return 0
        col = self._get_collection(collection)
        try:
            # First check which ones actually exist
            existing = self.exists(collection=collection, ids=doc_ids)
            if existing:
                col.delete(ids=existing)
            return len(existing)
        except Exception as exc:
            logger.warning("[ChromaVectorDB] delete_many failed: %s", exc)
            return 0

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_collection(self, name: str) -> chromadb.Collection:
        """
        Return cached collection object.
        If not cached (e.g. app restart), fetch from Chroma.
        If it doesn't exist at all, raise clearly so caller knows to
        call ensure_collection() first.
        """
        if name not in self._collections:
            try:
                self._collections[name] = self._client.get_collection(name=name)
            except Exception:
                raise RuntimeError(
                    f"[ChromaVectorDB] Collection '{name}' does not exist. "
                    f"Call ensure_collection('{name}', embedding_dim=N) first."
                )
        return self._collections[name]

    @staticmethod
    def _make_client(
        persist_directory: Optional[str],
        anonymized_telemetry: bool,
    ) -> chromadb.ClientAPI:
        settings = Settings(anonymized_telemetry=anonymized_telemetry)
        if persist_directory:
            return chromadb.PersistentClient(
                path=persist_directory,
                settings=settings,
            )
        return chromadb.EphemeralClient(settings=settings)


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChromaDB only accepts metadata values of type: str, int, float, bool.
    This converts anything else (None, list, dict) to string.

    WHY: Chroma silently drops or errors on complex metadata values.
    This prevents silent data loss.
    """
    safe: Dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        elif v is None:
            safe[k] = ""
        else:
            safe[k] = str(v)
    return safe
