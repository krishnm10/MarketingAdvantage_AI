# app/retrieval/policy.py

from enum import Enum
from app.retrieval.types_retrieve import RetrievalCandidate


# =========================================================
# Trust Decision States
# =========================================================

class TrustDecision(Enum):
    TRUSTED = "trusted"
    PROVISIONAL = "provisional"
    REJECTED = "rejected"


# =========================================================
# Helper Functions
# =========================================================

def compute_policy_trust_score(trust_obj) -> float:
    """
    Unified calculator for governance trust.
    Logic should match your enterprise requirements.
    """
    # Assuming trust_obj has a score attribute; adjust as per your types_retrieve.py
    return getattr(trust_obj, 'score', 0.0)


# =========================================================
# Core Retrieval Policy
# =========================================================

class RetrievalPolicy:
    """
    Enterprise-grade retrieval policy.

    Distinguishes between:
    - TRUSTED: Explicit governance validation exists
    - PROVISIONAL: Semantic relevance strong, governance pending
    - REJECTED: Weak relevance or negative governance
    """

    MIN_SEMANTIC_SCORE = 0.30
    PROVISIONAL_SEMANTIC_SCORE = 0.35
    MIN_TRUSTED_SCORE = 0.60

    # Runtime operational limits
    max_results = 5            # final answers returned to user
    min_results = 1            # minimum acceptable answers

    def decide(self, candidate: RetrievalCandidate) -> TrustDecision:
        """
        Make trust decision using unified trust calculator.
        """
        semantic_score = candidate.semantic.score
        trust = candidate.trust

        # 1. Hard semantic rejection
        if semantic_score < self.MIN_SEMANTIC_SCORE:
            return TrustDecision.REJECTED

        # 2. Compute governance trust score (UNIFIED CALCULATOR)
        policy_trust_score = compute_policy_trust_score(trust)

        # 3. Trusted
        if policy_trust_score >= self.MIN_TRUSTED_SCORE:
            return TrustDecision.TRUSTED

        # 4. Provisional
        if semantic_score >= self.PROVISIONAL_SEMANTIC_SCORE:
            return TrustDecision.PROVISIONAL

        # 5. Reject
        return TrustDecision.REJECTED


# =========================================================
# Policy Registry (what runtime expects)
# =========================================================

class RetrievalPolicyRegistry:
    """
    Runtime-facing policy registry.
    """

    def __init__(self):
        self._default_policy = RetrievalPolicy()

    def resolve(self, intent=None) -> "RetrievalPolicy":
        # Intent routing comes later; for now always default
        return self._default_policy


# =========================================================
# Singleton (used by runtime & CLI)
# =========================================================

DEFAULT_POLICY_REGISTRY = RetrievalPolicyRegistry()
