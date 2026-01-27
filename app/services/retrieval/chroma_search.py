"""
ChromaDB Search Service
Compatible with RetrievalRepository
"""

import asyncio
import chromadb
from typing import List, Tuple
from app.utils.logger import log_debug, log_info, log_warning


# =========================================================
# CONFIGURATION
# =========================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "ingested_content"


# =========================================================
# CHROMADB SEARCH CLASS
# =========================================================

class ChromaSearch:
    """ChromaDB search service"""
    
    def __init__(self, chroma_path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            log_info(f"[ChromaSearch] Initializing ChromaDB at {self.chroma_path}")
            self._client = chromadb.PersistentClient(path=self.chroma_path)
            self._collection = self._client.get_collection(name=self.collection_name)
            log_info(f"[ChromaSearch] ✅ Connected to collection '{self.collection_name}'")
        except Exception as e:
            log_warning(f"[ChromaSearch] Failed to initialize: {e}")
            raise
    
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 200
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            limit: Max results
        
        Returns:
            List of (semantic_hash, similarity_score) tuples
        """
        
        if not query_embedding or len(query_embedding) == 0:
            log_warning("[ChromaSearch] Empty query embedding")
            return []
        
        log_debug(f"[ChromaSearch] Searching with limit={limit}, dim={len(query_embedding)}")
        
        # Run in thread pool (ChromaDB is sync)
        loop = asyncio.get_running_loop()
        
        def _query():
            return self._collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
            )
        
        try:
            result = await loop.run_in_executor(None, _query)
        except Exception as e:
            log_warning(f"[ChromaSearch] Query failed: {e}")
            return []
        
        # Extract IDs and distances
        ids = result.get("ids") or []
        distances = result.get("distances")
        
        # Handle nested lists (ChromaDB format)
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        
        if distances and isinstance(distances, list) and isinstance(distances[0], list):
            distances = distances[0]
        
        log_debug(f"[ChromaSearch] Found {len(ids)} results")
        
        # Convert to (hash, score) tuples
        hits = []
        for idx, semantic_hash in enumerate(ids):
            if distances and idx < len(distances):
                # Convert L2 distance to similarity
                score = max(0.0, min(1.0, 1.0 - distances[idx]))
            else:
                score = 1.0
            hits.append((semantic_hash, score))
        
        return hits


# =========================================================
# SINGLETON
# =========================================================

_chroma_search_instance = None


def get_chroma_search() -> ChromaSearch:
    """Get global ChromaSearch instance"""
    global _chroma_search_instance
    
    if _chroma_search_instance is None:
        _chroma_search_instance = ChromaSearch()
    
    return _chroma_search_instance
