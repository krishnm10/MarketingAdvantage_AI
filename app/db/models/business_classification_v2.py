import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, JSON, ForeignKey, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


class BusinessClassificationV2(Base):
    """
    BusinessClassificationV2
    -------------------------
    Stores AI-assigned or human-reviewed industry classifications
    derived from ingested content and taxonomy hierarchy.

    Improvements over v1:
      • Integrates with TaxonomyV2 + PendingTaxonomyV2
      • Captures multi-level classification hierarchy (industry, sub, sub-sub)
      • Links classification reasoning and model version
      • Enables confidence-based revalidation and business-level analytics
    """

    __tablename__ = "business_classification"

    # -----------------------------------------------------------
    # PRIMARY & RELATIONSHIPS
    # -----------------------------------------------------------
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    content_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ingested_content.id", ondelete="CASCADE"),
        nullable=False,
    )

    business_id = Column(UUID(as_uuid=True), nullable=True)

    industry_id = Column(UUID(as_uuid=True), ForeignKey("taxonomy.id"), nullable=True)
    sub_industry_id = Column(UUID(as_uuid=True), ForeignKey("taxonomy.id"), nullable=True)
    sub_sub_industry_id = Column(UUID(as_uuid=True), ForeignKey("taxonomy.id"), nullable=True)

    pending_taxonomy_id = Column(
        UUID(as_uuid=True),
        ForeignKey("pending_taxonomy.id", ondelete="SET NULL"),
        nullable=True,
    )

    # -----------------------------------------------------------
    # CLASSIFICATION DETAILS
    # -----------------------------------------------------------
    confidence = Column(Float, nullable=False, default=0.0)
    llm_model = Column(String(128), default="llama-3.1-8b-instruct", doc="Model name/version used for classification")
    model_version = Column(String(64), nullable=True)

    # <<< PATCH: replace mutable default {} with callable dict() to avoid shared mutable default >>>
    raw_output = Column(JSON, default=dict, doc="Full AI reasoning or structured classification JSON output")

    review_status = Column(
        String(32),
        default="auto",
        doc="Classification source or review status: auto | human | verified | flagged",
    )

    # -----------------------------------------------------------
    # RELATIONSHIPS
    # -----------------------------------------------------------
    content = relationship("IngestedContentV2", backref="business_classifications", lazy="joined")
    industry = relationship("TaxonomyV2", foreign_keys=[industry_id], lazy="joined")
    sub_industry = relationship("TaxonomyV2", foreign_keys=[sub_industry_id], lazy="joined")
    sub_sub_industry = relationship("TaxonomyV2", foreign_keys=[sub_sub_industry_id], lazy="joined")
    pending_taxonomy = relationship("PendingTaxonomyV2", foreign_keys=[pending_taxonomy_id], lazy="joined")

    # -----------------------------------------------------------
    # TIMESTAMPS
    # -----------------------------------------------------------
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    # -----------------------------------------------------------
    # REPRESENTATION
    # -----------------------------------------------------------
    def __repr__(self):
        return (
            f"<BusinessClassificationV2(id={self.id}, content_id={self.content_id}, "
            f"confidence={self.confidence}, review_status={self.review_status})>"
        )

    # -----------------------------------------------------------
    # SERIALIZATION
    # -----------------------------------------------------------
    def to_dict(self):
        return {
            "id": str(self.id),
            "content_id": str(self.content_id),
            "business_id": str(self.business_id) if self.business_id else None,
            "industry_id": str(self.industry_id) if self.industry_id else None,
            "sub_industry_id": str(self.sub_industry_id) if self.sub_industry_id else None,
            "sub_sub_industry_id": str(self.sub_sub_industry_id) if self.sub_sub_industry_id else None,
            "pending_taxonomy_id": str(self.pending_taxonomy_id) if self.pending_taxonomy_id else None,
            "confidence": self.confidence,
            "llm_model": self.llm_model,
            "model_version": self.model_version,
            "review_status": self.review_status,
            "raw_output": self.raw_output,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
