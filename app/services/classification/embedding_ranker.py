# services/classification/embedding_ranker.py

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb

from app.utils.logger import log_info, log_warning
from app.services.classification.taxonomy_loader import TAXONOMY_COLLECTION, EMBEDDER


TOP_K = 10
MIN_SIMILARITY = 0.55      # below this → throw away
STRONG_MATCH = 0.80        # above this → LLM gets strong prior


def embed_text(text: str):
    """Generate normalized BGE-large embedding."""
    return EMBEDDER.encode(text, normalize_embeddings=True).tolist()


def rank_taxonomy_candidates(text: str) -> Dict[str, Any]:
    """
    Retrieves top-K taxonomy candidates from ChromaDB based on embedding similarity.
    Output:
        {
            "candidates": [
                {
                    "taxonomy_id": "...",
                    "name": "...",
                    "similarity": 0.87,
                    "level": "industry"
                },
                ...
            ],
            "strong_candidates": [...],
            "raw_results": {...}
        }
    """

    if not text or not text.strip():
        return {"candidates": [], "strong_candidates": [], "raw_results": {}}

    embed = embed_text(text)

    results = TAXONOMY_COLLECTION.query(
        query_embeddings=[embed],
        n_results=TOP_K,
        include=["metadatas", "documents", "distances"]
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    candidates = []
    strong = []

    for i, taxonomy_id in enumerate(ids):
        similarity = 1 - distances[i]  # chroma uses distance

        if similarity < MIN_SIMILARITY:
            continue

        item = {
            "taxonomy_id": taxonomy_id,
            "name": docs[i],
            "similarity": round(similarity, 4),
            "level": metas[i].get("level")
        }

        candidates.append(item)

        if similarity >= STRONG_MATCH:
            strong.append(item)

    return {
        "candidates": candidates,
        "strong_candidates": strong,
        "raw_results": {
            "ids": ids,
            "docs": docs,
            "metas": metas,
            "distances": distances
        }
    }
