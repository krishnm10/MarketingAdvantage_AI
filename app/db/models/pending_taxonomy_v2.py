import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Float, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship

from app.db.base import Base


def generate_uuid():
    # <<< PATCH: return uuid.UUID object instead of str to match UUID(as_uuid=True) columns >>>
    return uuid.uuid4()


class PendingTaxonomyV2(Base):
    """
    PendingTaxonomyV2
    -----------------
    Temporary queue for unverified or AI-suggested taxonomy entries.

    Purpose:
      • Stores new taxonomy candidates discovered from content ingestion.
      • Tracks similarity to existing TaxonomyV2 nodes.
      • Holds LLM reasoning & confidence scores for human review.

    Extended Features:
      • Confidence & similarity tracking
      • Optional linkage to detected content chunk
      • AI explanation field for reasoning traceability
    """

    __tablename__ = "pending_taxonomy"

    # -----------------------------------------------------------
    # PRIMARY FIELDS
    # -----------------------------------------------------------
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)

    raw_name = Column(String(255), nullable=False)
    canonical_name = Column(String(255), nullable=False)

    suggested_parent_id = Column(UUID(as_uuid=True), ForeignKey("taxonomy.id"), nullable=True)
    suggested_level = Column(Integer, nullable=True)

    # -----------------------------------------------------------
    # AI + SIMILARITY METADATA
    # -----------------------------------------------------------
    similar_existing_ids = Column(ARRAY(String), default=list, doc="Array of related taxonomy UUIDs")
    confidence = Column(Float, default=0.0, doc="AI classification confidence score")

    status = Column(
        String(20),
        default="pending",
        doc="pending | approved | rejected | merged",
    )

    level = Column(String(50), nullable=True, doc="Taxonomy depth level proposed by AI")
    detected_from_chunk = Column(UUID(as_uuid=True), nullable=True, doc="Chunk ID from which this taxonomy was derived")

    llm_reason = Column(
        String(2000),
        nullable=True,
        doc="Explanation or justification provided by the AI classifier",
    )

    review_notes = Column(String(1000), nullable=True, doc="Manual reviewer comments or override notes")

    # -----------------------------------------------------------
    # TIMESTAMP
    # -----------------------------------------------------------
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # -----------------------------------------------------------
    # RELATIONSHIPS
    # -----------------------------------------------------------
    parent = relationship("TaxonomyV2", backref="pending_children", lazy="joined")

    # -----------------------------------------------------------
    # REPRESENTATION
    # -----------------------------------------------------------
    def __repr__(self):
        return (
            f"<PendingTaxonomyV2(id={self.id}, raw_name='{self.raw_name}', "
            f"canonical_name='{self.canonical_name}', confidence={self.confidence}, status='{self.status}')>"
        )

    # -----------------------------------------------------------
    # SERIALIZATION
    # -----------------------------------------------------------
    def to_dict(self):
        return {
            "id": str(self.id),
            "raw_name": self.raw_name,
            "canonical_name": self.canonical_name,
            "suggested_parent_id": str(self.suggested_parent_id) if self.suggested_parent_id else None,
            "suggested_level": self.suggested_level,
            "similar_existing_ids": self.similar_existing_ids,
            "confidence": self.confidence,
            "status": self.status,
            "level": self.level,
            "detected_from_chunk": str(self.detected_from_chunk) if self.detected_from_chunk else None,
            "llm_reason": self.llm_reason,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
