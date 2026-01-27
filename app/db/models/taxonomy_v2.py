# taxonomy_v2.py — Enhanced Taxonomy ORM Model (Production-Ready)
# Fully aligned with ingestion_v2 ecosystem and classification hierarchy

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base


def generate_uuid():
    # <<< PATCH: return a uuid.UUID object (not a string) to match UUID(as_uuid=True) columns >>>
    return uuid.uuid4()


class TaxonomyV2(Base):
    """
    TaxonomyV2
    -----------
    Hierarchical business taxonomy structure for content classification and strategy mapping.
    """

    __tablename__ = "taxonomy"

    # -----------------------------------------------------------
    # PRIMARY + STRUCTURE FIELDS
    # -----------------------------------------------------------
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    canonical_name = Column(String(255), nullable=False, unique=True)

    parent_id = Column(UUID(as_uuid=True), ForeignKey("taxonomy.id", ondelete="CASCADE"), nullable=True)
    level = Column(Integer, nullable=False, default=0, doc="Hierarchy level: 0=root, 1=sub, 2=sub-sub, etc.")

    slug = Column(String(255), nullable=True, index=True)

    # -----------------------------------------------------------
    # METADATA + NLP SUPPORT
    # -----------------------------------------------------------
    meta_data = Column(JSONB, default=dict, doc="Dynamic JSONB for attributes like embeddings, descriptions, etc.")
    embedding_id = Column(String(255), nullable=True)
    vector_enabled = Column(Boolean, default=False, doc="Flag for semantic search readiness.")

    # -----------------------------------------------------------
    # RELATIONSHIPS
    # -----------------------------------------------------------
    parent = relationship("TaxonomyV2", remote_side=[id], backref="children", lazy="joined")

    # -----------------------------------------------------------
    # TIMESTAMPS
    # -----------------------------------------------------------
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # -----------------------------------------------------------
    # REPRESENTATION
    # -----------------------------------------------------------
    def __repr__(self):
        return f"<TaxonomyV2(id={self.id}, name={self.name}, level={self.level}, slug={self.slug})>"

    # -----------------------------------------------------------
    # UTILITY METHODS
    # -----------------------------------------------------------
    def to_dict(self, include_children: bool = False):
        data = {
            "id": str(self.id),
            "name": self.name,
            "canonical_name": self.canonical_name,
            "slug": self.slug,
            "level": self.level,
            "meta_data": self.meta_data,
            "embedding_id": self.embedding_id,
            "vector_enabled": self.vector_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

        if include_children:
            data["children"] = [child.to_dict(include_children=False) for child in getattr(self, "children", [])]

        return data
