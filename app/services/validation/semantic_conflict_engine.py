"""
Semantic Conflict Detection Engine - Production Grade
Version: 2.0 (Async-Safe, Retrieval-Optimized)

Responsibilities:
- Detect contradictory information across sources
- Compute conflict risk scores
- Provide retrieval-safe conflict modifiers
- Never block async retrieval pipeline
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
import math

from app.db.session_v2 import AsyncSessionLocal
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.utils.logger import log_info, log_warning, log_debug

# ChromaDB imports
import chromadb

# ============================================================
# VERSIONING
# ============================================================

CONFLICT_VERSION = "conflict_analysis_v2"
CONFLICT_SCHEMA_CONTRACT = "retrieval_v2_compatible"

# ============================================================
# CONFLICT DETECTION CONFIG
# ============================================================

class ConflictConfig:
    """Configurable conflict detection parameters"""
    
    # Similarity threshold for conflict detection
    SIMILARITY_THRESHOLD = 0.80  # 80% similar = potential conflict
    
    # Number of similar documents to check
    TOP_K_SIMILAR = 10
    
    # Minimum collection size to perform conflict detection
    MIN_COLLECTION_SIZE = 2
    
    # Conflict risk weights
    SIMILARITY_WEIGHT = 0.9  # How similar the texts are
    POLARITY_WEIGHT = 1.0    # Opposite polarities = stronger conflict
    
    # Cache TTL (seconds)
    CACHE_TTL = 300  # 5 minutes

# ============================================================
# POLARITY DETECTION (Domain-Specific)
# ============================================================

# Positive sentiment keywords
POSITIVE_TERMS = {
    "increase", "improve", "boost", "grow", "enhance", "strengthen",
    "optimize", "accelerate", "expand", "elevate", "maximize",
    "reduce risk", "reduce cost", "reduce time", "increase revenue",
    "increase profit", "increase efficiency", "better", "higher",
    "faster", "stronger", "successful", "effective"
}

# Negative sentiment keywords
NEGATIVE_TERMS = {
    "decrease", "reduce", "decline", "worsen", "diminish", "weaken",
    "hurt", "damage", "impair", "limit", "restrict", "constrain",
    "raise cost", "raise risk", "increase risk", "increase cost",
    "lower", "slower", "worse", "failed", "ineffective", "problematic"
}

# Metric categories (what the polarity applies to)
METRIC_KEYWORDS = {
    "productivity": {
        "productivity", "efficiency", "output", "throughput",
        "utilization", "performance", "capacity"
    },
    "cost": {
        "cost", "expense", "spend", "budget", "price",
        "overhead", "opex", "capex"
    },
    "revenue": {
        "revenue", "sales", "income", "profit", "margin",
        "earnings", "return", "roi"
    },
    "risk": {
        "risk", "exposure", "liability", "threat", "vulnerability",
        "compliance", "security", "safety"
    },
    "quality": {
        "quality", "accuracy", "reliability", "consistency",
        "defect", "error", "issue", "problem"
    },
    "time": {
        "time", "duration", "speed", "latency", "cycle time",
        "lead time", "response time", "turnaround"
    },
    "customer": {
        "customer satisfaction", "nps", "churn", "retention",
        "engagement", "loyalty", "experience"
    },
}

# ============================================================
# PUBLIC ENTRY POINT (Worker)
# ============================================================

async def run_semantic_conflict_detection(batch_size: int = 50) -> Dict[str, Any]:
    """
    Main conflict detection worker.
    
    Returns:
        Processing stats
    """
    start_time = datetime.now(timezone.utc)
    
    async with AsyncSessionLocal() as session:
        # Fetch candidates (already validated in Step-2.1)
        rows = await _fetch_candidates(session, batch_size)
        
        if not rows:
            log_info("[ConflictEngine] No rows pending conflict analysis.")
            return {
                "processed": 0,
                "conflicts_detected": 0,
                "duration_ms": 0
            }
        
        log_info(f"[ConflictEngine] Processing {len(rows)} chunks...")
        
        # Get ChromaDB client
        client = _get_chroma_client()
        collection = client.get_collection(name="ingested_content")
        
        # Check collection size
        if collection.count() < ConflictConfig.MIN_COLLECTION_SIZE:
            log_info(
                f"[ConflictEngine] Collection too small ({collection.count()} docs), "
                f"skipping conflict detection"
            )
            return {
                "processed": 0,
                "conflicts_detected": 0,
                "duration_ms": 0
            }
        
        processed = 0
        conflicts_found = 0
        
        for row in rows:
            try:
                analysis = await analyze_conflicts_for_chunk(
                    session, collection, row
                )
                
                if analysis:
                    await append_conflict_snapshot(session, row.id, analysis)
                    
                    # Count conflicts
                    if analysis.get("conflicts_detected"):
                        conflicts_found += len(analysis["conflicts_detected"])
                
                processed += 1
                
            except Exception as e:
                log_warning(f"[ConflictEngine] Failed for {row.id}: {e}")
        
        await session.commit()
        
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        log_info(
            f"[ConflictEngine] ✅ Processed {processed} chunks, "
            f"found {conflicts_found} conflicts in {duration_ms:.2f}ms"
        )
        
        return {
            "processed": processed,
            "conflicts_detected": conflicts_found,
            "duration_ms": round(duration_ms, 2)
        }

# ============================================================
# CANDIDATE SELECTION
# ============================================================

async def _fetch_candidates(
    session: AsyncSession,
    batch_size: int
) -> List[IngestedContentV2]:
    """
    Fetch chunks that have Step-2.1 validation but no conflict analysis.
    """
    stmt = (
        select(IngestedContentV2)
        .where(
            IngestedContentV2.validation_layer.is_not(None),
            ~IngestedContentV2.validation_layer.op("@>")(
                [{"method": CONFLICT_VERSION}]
            ),
        )
        .order_by(IngestedContentV2.created_at.asc())
        .limit(batch_size)
    )
    
    result = await session.execute(stmt)
    return list(result.scalars().all())

# ============================================================
# CORE CONFLICT ANALYSIS
# ============================================================

async def analyze_conflicts_for_chunk(
    session: AsyncSession,
    collection: chromadb.Collection,
    chunk: IngestedContentV2
) -> Optional[Dict[str, Any]]:
    """
    Analyze semantic conflicts for a single chunk.
    
    Process:
    1. Get embedding from ChromaDB (single source of truth)
    2. Find semantically similar chunks (vector search)
    3. Check for polarity conflicts (opposite claims)
    4. Compute conflict risk score
    5. Return analysis snapshot
    
    Returns:
        Conflict analysis dict or None if no analysis possible
    """
    
    # -------------------------------------------------
    # 1. Validate Input Data
    # -------------------------------------------------
    if not chunk.global_content_id:
        log_debug(f"[ConflictAnalysis] Chunk {chunk.id} missing global_content_id")
        return None
    
    gci = await session.get(GlobalContentIndexV2, chunk.global_content_id)
    if not gci:
        log_debug(f"[ConflictAnalysis] GCI not found for chunk {chunk.id}")
        return None
    
    if not gci.cleaned_text:
        log_debug(f"[ConflictAnalysis] Chunk {chunk.id} has no cleaned text")
        return None
    
    canonical_text = gci.cleaned_text
    semantic_hash = gci.semantic_hash
    
    # -------------------------------------------------
    # 2. Get Embedding from ChromaDB (Single Source of Truth)
    # -------------------------------------------------
        # -------------------------------------------------
    # 2. Get Embedding from ChromaDB (Single Source of Truth)
    # -------------------------------------------------
    query_embedding = None
    
    try:
        import numpy as np
        
        # Fetch the stored document from ChromaDB by semantic_hash
        chroma_result = collection.get(
            where={"semantic_hash": semantic_hash},
            include=["embeddings"]
        )
        
        # Validate result structure
        if chroma_result is None:
            log_debug(f"[ConflictAnalysis] ChromaDB returned None for {chunk.id}")
            return None
        
        # Get embeddings list
        embeddings = chroma_result.get("embeddings")
        
        # Check embeddings exist
        if embeddings is None:
            log_debug(f"[ConflictAnalysis] No embeddings key for {chunk.id}")
            return None
        
        # ✅ Accept list, tuple, or numpy array (ChromaDB can return any of these)
        if not isinstance(embeddings, (list, tuple, np.ndarray)):
            log_warning(
                f"[ConflictAnalysis] Unexpected embeddings type {type(embeddings)} "
                f"for {chunk.id}"
            )
            return None
        
        # Check has content
        if len(embeddings) == 0:
            log_debug(
                f"[ConflictAnalysis] Empty embeddings list for {chunk.id} "
                f"(semantic_hash: {semantic_hash[:16]}...)"
            )
            return None
        
        # Extract first embedding
        query_embedding = embeddings[0]
        
        # Validate embedding content
        if query_embedding is None:
            log_debug(f"[ConflictAnalysis] Null embedding for {chunk.id}")
            return None
        
        # ✅ Convert to list (handle numpy arrays, lists, tuples)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        elif hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        elif not isinstance(query_embedding, list):
            query_embedding = list(query_embedding)
        
        # Final validation
        if len(query_embedding) == 0:
            log_warning(f"[ConflictAnalysis] Zero-length embedding for {chunk.id}")
            return None
        
        log_debug(
            f"[ConflictAnalysis] ✅ Retrieved {len(query_embedding)}-dim embedding "
            f"from ChromaDB for {chunk.id}"
        )
        
    except Exception as e:
        log_warning(
            f"[ConflictAnalysis] Failed to fetch embedding from ChromaDB "
            f"for {chunk.id}: {e}"
        )
        import traceback
        log_debug(f"[ConflictAnalysis] Traceback: {traceback.format_exc()}")
        return None

    
    # -------------------------------------------------
    # 3. Find Similar Chunks (Vector Search)
    # -------------------------------------------------
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=ConflictConfig.TOP_K_SIMILAR,
            where={"semantic_hash": {"$ne": semantic_hash}},  # Exclude self
            include=["metadatas", "distances"]
        )
    except Exception as e:
        log_warning(f"[ConflictAnalysis] ChromaDB query failed for {chunk.id}: {e}")
        return None
    
    # Validate query results
    if results is None:
        log_debug(f"[ConflictAnalysis] ChromaDB query returned None for {chunk.id}")
        return None
    
    metadatas = results.get("metadatas")
    distances = results.get("distances")
    
    if metadatas is None or distances is None:
        log_debug(f"[ConflictAnalysis] Missing metadatas or distances for {chunk.id}")
        return None
    
    if len(metadatas) == 0 or len(metadatas[0]) == 0:
        log_debug(f"[ConflictAnalysis] No similar chunks found for {chunk.id}")
        return None
    
    # Get first result set
    metadatas = metadatas[0]
    distances = distances[0]
    
    # -------------------------------------------------
    # 4. Detect Conflicts
    # -------------------------------------------------
    conflicts = []
    
    for metadata, distance in zip(metadatas, distances):
        # Convert distance to similarity
        similarity = 1.0 - (distance / 2.0)
        
        # Filter by similarity threshold
        if similarity < ConflictConfig.SIMILARITY_THRESHOLD:
            continue
        
        other_hash = metadata.get("semantic_hash")
        if not other_hash:
            continue
        
        # Fetch the other chunk's text
        other_gci = await _get_gci_by_hash(session, other_hash)
        if not other_gci or not other_gci.cleaned_text:
            continue
        
        # Check for polarity conflict
        metric, polarity_self, polarity_other = _detect_polarity_conflict(
            canonical_text,
            other_gci.cleaned_text
        )
        
        if not metric:
            # No polarity conflict detected
            continue
        
        # Compute pair conflict score
        pair_score = _compute_pair_conflict_score(
            similarity,
            polarity_self,
            polarity_other
        )
        
        conflicts.append({
            "semantic_hash": other_hash,
            "similarity": round(similarity, 4),
            "metric": metric,
            "polarity_self": polarity_self,
            "polarity_other": polarity_other,
            "pair_conflict_score": round(pair_score, 4),
        })
    
    # -------------------------------------------------
    # 5. Aggregate Conflict Risk
    # -------------------------------------------------
    if len(conflicts) == 0:
        # No conflicts = clean snapshot
        return {
            "method": CONFLICT_VERSION,
            "schema_contract": CONFLICT_SCHEMA_CONTRACT,
            "evaluated_at": utc_now_iso(),
            "chunk_id": str(chunk.id),
            "similarity_threshold": ConflictConfig.SIMILARITY_THRESHOLD,
            "conflicts_detected": [],
            "conflict_risk": 0.0,
            "confidence": 1.0,
        }
    
    # Aggregate multiple conflict scores
    conflict_risk = _aggregate_conflict_scores(
        [c["pair_conflict_score"] for c in conflicts]
    )
    
    # Confidence based on number of comparisons
    confidence = min(1.0, len(conflicts) / ConflictConfig.TOP_K_SIMILAR + 0.5)
    
    return {
        "method": CONFLICT_VERSION,
        "schema_contract": CONFLICT_SCHEMA_CONTRACT,
        "evaluated_at": utc_now_iso(),
        "chunk_id": str(chunk.id),
        "similarity_threshold": ConflictConfig.SIMILARITY_THRESHOLD,
        "conflicts_detected": conflicts,
        "conflict_risk": round(conflict_risk, 4),
        "confidence": round(confidence, 2),
        "metadata": {
            "top_k_checked": ConflictConfig.TOP_K_SIMILAR,
            "conflicts_found": len(conflicts),
            "embedding_dimension": len(query_embedding),
            "embedding_source": "chromadb",
        },
    }


# ============================================================
# POLARITY CONFLICT DETECTION
# ============================================================

def _detect_polarity_conflict(
    text_a: str,
    text_b: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Detect if two texts make opposing claims about the same metric.
    
    Returns:
        (metric, polarity_a, polarity_b) or (None, None, None)
    """
    text_a_lower = text_a.lower()
    text_b_lower = text_b.lower()
    
    # Check each metric category
    for metric_name, metric_keywords in METRIC_KEYWORDS.items():
        
        # Both texts must mention this metric
        a_has_metric = any(kw in text_a_lower for kw in metric_keywords)
        b_has_metric = any(kw in text_b_lower for kw in metric_keywords)
        
        if not (a_has_metric and b_has_metric):
            continue
        
        # Determine polarity for text A
        a_positive = any(term in text_a_lower for term in POSITIVE_TERMS)
        a_negative = any(term in text_a_lower for term in NEGATIVE_TERMS)
        
        if a_positive and not a_negative:
            polarity_a = "positive"
        elif a_negative and not a_positive:
            polarity_a = "negative"
        else:
            polarity_a = "neutral"
        
        # Determine polarity for text B
        b_positive = any(term in text_b_lower for term in POSITIVE_TERMS)
        b_negative = any(term in text_b_lower for term in NEGATIVE_TERMS)
        
        if b_positive and not b_negative:
            polarity_b = "positive"
        elif b_negative and not b_positive:
            polarity_b = "negative"
        else:
            polarity_b = "neutral"
        
        # Conflict exists if polarities are opposite (not neutral)
        if (polarity_a != polarity_b and 
            "neutral" not in (polarity_a, polarity_b)):
            return metric_name, polarity_a, polarity_b
    
    # No conflict detected
    return None, None, None

# ============================================================
# CONFLICT SCORING
# ============================================================

def _compute_pair_conflict_score(
    similarity: float,
    polarity_a: str,
    polarity_b: str
) -> float:
    """
    Compute conflict score for a single pair.
    
    Higher similarity + opposite polarities = higher conflict
    """
    # Base score from similarity
    base_score = similarity * ConflictConfig.SIMILARITY_WEIGHT
    
    # Polarity factor (opposite = full weight, same = no conflict)
    if polarity_a != polarity_b and "neutral" not in (polarity_a, polarity_b):
        polarity_factor = ConflictConfig.POLARITY_WEIGHT
    else:
        polarity_factor = 0.0
    
    return base_score * polarity_factor


def _aggregate_conflict_scores(scores: List[float]) -> float:
    """
    Aggregate multiple conflict scores into single risk metric.
    
    Uses probabilistic OR logic:
    - Multiple weak conflicts compound
    - Single strong conflict dominates
    
    Formula: 1 - ∏(1 - score_i)
    """
    if not scores:
        return 0.0
    
    # Probabilistic OR aggregation
    risk = 1.0
    for score in scores:
        risk *= (1.0 - score)
    
    return max(0.0, min(1.0, 1.0 - risk))

# ============================================================
# DATABASE HELPERS
# ============================================================

async def _get_gci_by_hash(
    session: AsyncSession,
    semantic_hash: str
) -> Optional[GlobalContentIndexV2]:
    """Fetch GlobalContentIndex by semantic hash"""
    stmt = (
        select(GlobalContentIndexV2)
        .where(GlobalContentIndexV2.semantic_hash == semantic_hash)
        .limit(1)
    )
    
    result = await session.execute(stmt)
    return result.scalars().first()


async def append_conflict_snapshot(
    session: AsyncSession,
    ingested_id,
    snapshot: Dict[str, Any]
) -> None:
    """Append conflict analysis to validation_layer"""
    stmt = text("""
        UPDATE ingested_content
        SET validation_layer =
            COALESCE(validation_layer, '[]'::jsonb)
            || CAST(:payload AS jsonb)
        WHERE id = :id
    """)
    
    payload_json = json.dumps([snapshot])
    
    await session.execute(
        stmt,
        {"id": ingested_id, "payload": payload_json}
    )

# ============================================================
# CHROMA CLIENT
# ============================================================

def _get_chroma_client():
    """Get or create ChromaDB client"""
    import os
    chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
    return chromadb.PersistentClient(path=chroma_path)

# ============================================================
# UTILITIES
# ============================================================

def utc_now_iso() -> str:
    """Get current UTC timestamp as ISO string"""
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# RETRIEVAL ADAPTER (Async-Safe, Production-Grade)
# ============================================================

async def get_conflict_modifier_async(
    session: AsyncSession,
    content_id: str
) -> float:
    """
    Fetch conflict modifier for retrieval (ASYNC-SAFE).
    
    Returns:
        Float in [0.0, 1.0] where:
        - 1.0 = no conflict
        - 0.0 = maximum conflict
    
    Guarantees:
    - Never raises exceptions
    - Always returns valid float
    - Fails OPEN (1.0) on errors
    - Logs all failures
    """
    
    try:
        # Fetch content with validation_layer
        stmt = select(IngestedContentV2).where(
            IngestedContentV2.id == content_id
        )
        result = await session.execute(stmt)
        content = result.scalar_one_or_none()
        
        if not content:
            log_debug(
                f"[ConflictModifier] Content {content_id} not found, "
                f"returning 1.0 (no conflict)"
            )
            return 1.0
        
        if not content.validation_layer:
            log_debug(
                f"[ConflictModifier] Content {content_id} has no validation_layer, "
                f"returning 1.0 (no conflict analysis)"
            )
            return 1.0
        
        # Find most recent conflict analysis (reverse iteration)
        for snapshot in reversed(content.validation_layer):
            if not isinstance(snapshot, dict):
                continue
            
            if snapshot.get("method") == CONFLICT_VERSION:
                conflict_risk = snapshot.get("conflict_risk", 0.0)
                
                # Convert risk [0,1] to modifier [1,0]
                # risk=0 → modifier=1 (no conflict, full trust)
                # risk=1 → modifier=0 (max conflict, zero trust)
                modifier = 1.0 - float(conflict_risk)
                
                # Clamp to valid range
                modifier = max(0.0, min(1.0, modifier))
                
                log_debug(
                    f"[ConflictModifier] Content {content_id}: "
                    f"risk={conflict_risk:.3f}, modifier={modifier:.3f}"
                )
                
                return modifier
        
        # No conflict analysis found
        log_debug(
            f"[ConflictModifier] Content {content_id} has no conflict analysis, "
            f"returning 1.0 (default)"
        )
        return 1.0
        
    except Exception as e:
        # CRITICAL: Never crash retrieval due to conflict detection failure
        log_warning(
            f"[ConflictModifier] EXCEPTION for content {content_id}: {e} "
            f"(returning 1.0 - fail open)"
        )
        return 1.0


def get_conflict_modifier_sync(content_id: str) -> float:
    """
    DEPRECATED: Synchronous wrapper (for backward compatibility only).
    
    Use get_conflict_modifier_async() in async contexts.
    
    WARNING: This creates a new event loop - use sparingly.
    """
    import asyncio
    
    async def _fetch():
        async with AsyncSessionLocal() as session:
            return await get_conflict_modifier_async(session, content_id)
    
    try:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            log_warning(
                "[ConflictModifier] Called sync wrapper in async context! "
                "Use get_conflict_modifier_async() instead."
            )
            # Fall back to default
            return 1.0
        except RuntimeError:
            # No running loop, safe to create new one
            return asyncio.run(_fetch())
    except Exception as e:
        log_warning(
            f"[ConflictModifier] Sync wrapper failed: {e} (returning 1.0)"
        )
        return 1.0


# Backward compatibility alias
get_conflict_modifier = get_conflict_modifier_async
