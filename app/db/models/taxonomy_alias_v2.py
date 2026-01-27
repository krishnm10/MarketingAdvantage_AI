# =============================================
# taxonomy_alias_v2.py — ORM Model (Production-Ready)
# Fully compatible with TaxonomyV2 and ingestion_v2 architecture
# =============================================

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


def generate_uuid():
    # <<< PATCH: return UUID object instead of string >>>
    return uuid.uuid4()


class TaxonomyAliasV2(Base):
    """
    TaxonomyAliasV2
    -----------------
    Maps user-entered or alternate taxonomy names (aliases)
    to their canonical TaxonomyV2 entities.
    """

    __tablename__ = "taxonomy_alias"

    # -----------------------------------------------------------
    # PRIMARY + FOREIGN KEYS
    # -----------------------------------------------------------
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    alias_name = Column(String(255), nullable=False, index=True)

    canonical_taxonomy_id = Column(
        UUID(as_uuid=True),
        ForeignKey("taxonomy.id", ondelete="CASCADE"),
        nullable=False,
    )

    # -----------------------------------------------------------
    # RELATIONSHIPS
    # -----------------------------------------------------------
    taxonomy = relationship("TaxonomyV2", backref="aliases", lazy="joined")

    # -----------------------------------------------------------
    # METADATA
    # -----------------------------------------------------------
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "alias_name",
            "canonical_taxonomy_id",
            name="uq_taxonomy_alias_v2_alias_mapping",
        ),
    )

    # -----------------------------------------------------------
    # UTILITY + REPRESENTATION
    # -----------------------------------------------------------
    def __repr__(self):
        return f"<TaxonomyAliasV2(id={self.id}, alias='{self.alias_name}', canonical_id={self.canonical_taxonomy_id})>"

    def to_dict(self):
        return {
            "id": str(self.id),
            "alias_name": self.alias_name,
            "canonical_taxonomy_id": str(self.canonical_taxonomy_id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
