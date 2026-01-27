# services/classification/canonicalizer.py

import uuid
from typing import Optional, Dict, Any
from rapidfuzz import fuzz

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert, select

from app.db.models.pending_taxonomy import PendingTaxonomy
from app.db.models.taxonomy import Taxonomy
from app.services.classification.taxonomy_loader import (
    find_canonical,
    get_taxonomy_tree,
    get_taxonomy_node,
)
from app.utils.logger import log_info, log_warning


# ----------------------------------
# CONFIG VALUES
# ----------------------------------
FUZZY_MATCH_THRESHOLD = 85  # 0–100
FUZZY_SECONDARY_THRESHOLD = 75


# ----------------------------------
# HELPER FUNCTIONS
# ----------------------------------

def normalize_name(name: Optional[str]) -> Optional[str]:
    """Clean and standardize for fuzzy matching."""
    if not name:
        return None
    return name.strip().lower()


def fuzzy_match_taxonomy(name: str) -> Optional[str]:
    """
    Attempts a fuzzy match across all taxonomy names.
    Returns taxonomy_id on success.
    """

    tree = get_taxonomy_tree()
    best_id = None
    best_score = 0

    for t_id, node in tree.items():
        score = fuzz.ratio(name, node.name.lower())
        if score > best_score:
            best_score = score
            best_id = t_id

    if best_score >= FUZZY_MATCH_THRESHOLD:
        return best_id

    return None


async def create_pending_taxonomy(
    db: AsyncSession,
    name: str,
    parent_id: Optional[str],
    level: str,
    detected_from_chunk: Optional[str],
    reason: str,
    confidence: float = 0.65,
    similar_ids: list = None
) -> str:

    new_id = uuid.uuid4()

    row = {
        "id": new_id,
        "raw_name": name,
        "canonical_name": name,   # LLM's best canonical guess
        "suggested_parent_id": parent_id,
        "suggested_level": level,
        "similar_existing_ids": similar_ids or [],
        "confidence": confidence,
        "status": "pending",
        "level": level,
        "detected_from_chunk": detected_from_chunk,
        "llm_reason": reason,
    }

    await db.execute(insert(PendingTaxonomy).values(**row))
    await db.commit()

    log_info(f"[canonicalizer] Created pending taxonomy {new_id}")
    return str(new_id)



# ----------------------------------
# MAIN CANONICALIZATION PIPELINE
# ----------------------------------

async def canonicalize_llm_output(db, chunk_id, llm_output):

    industry = normalize_name(llm_output.get("industry"))
    sub = normalize_name(llm_output.get("sub_industry"))
    sub_sub = normalize_name(llm_output.get("sub_sub_industry"))

    proposed_new = normalize_name(llm_output.get("proposed_new_taxonomy"))
    requires_admin = llm_output.get("requires_admin_approval", False)
    reason = llm_output.get("reason") or ""
    confidence = float(llm_output.get("confidence") or 0.6)

    out = {
        "industry_id": None,
        "sub_industry_id": None,
        "sub_sub_industry_id": None,
        "pending_taxonomy_id": None
    }

    def resolve(name):
        if not name:
            return None
        cid = find_canonical(name)
        if cid: return cid
        fid = fuzzy_match_taxonomy(name)
        return fid

    out["industry_id"] = resolve(industry)
    out["sub_industry_id"] = resolve(sub)
    out["sub_sub_industry_id"] = resolve(sub_sub)

    # handle new taxonomy suggestion
    if requires_admin or proposed_new:
        name = proposed_new or sub_sub or sub or industry

        pending_id = await create_pending_taxonomy(
            db=db,
            name=name,
            parent_id=out["industry_id"] or out["sub_industry_id"],
            level=llm_output.get("suggested_level") or "industry",
            detected_from_chunk=chunk_id,
            reason=reason,
            confidence=confidence,
            similar_ids=[]
        )

        out["pending_taxonomy_id"] = pending_id

    return out
