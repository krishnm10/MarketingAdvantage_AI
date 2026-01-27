"""
Retrieval Scorer - Uses Unified Trust Calculator
"""

from typing import Optional

from app.retrieval.types_retrieve import RetrievalCandidate
from app.retrieval.policy import RetrievalPolicy
from app.retrieval.trust_calculator import compute_ranking_score


def compute_final_score(
    candidate: RetrievalCandidate,
    policy: RetrievalPolicy,
) -> Optional[float]:
    """
    Compute final ranking score using unified trust calculator.
    
    This is a thin wrapper that applies policy-specific gates
    before calling the canonical ranking formula.
    """
    
    trust = candidate.trust
    
    # -------------------------------------------------
    # Hard Trust Gate (Policy-Specific)
    # -------------------------------------------------
    min_tap = getattr(policy, "min_tap_trust_score", None)
    if min_tap is not None and trust.tap_trust_score < min_tap:
        return None  # Below minimum threshold, reject
    
    # -------------------------------------------------
    # Compute Final Score (UNIFIED CALCULATOR)
    # -------------------------------------------------
    final_score = compute_ranking_score(
        semantic_score=candidate.semantic.score,
        trust_signals=trust
    )
    
    return final_score
