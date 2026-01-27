-- ================================================
-- PHASE 1: TAXONOMY SYSTEM (FINAL)
-- Fully compatible with Python models
-- ================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- TAXONOMY
CREATE TABLE taxonomy (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    parent_id UUID NULL REFERENCES taxonomy(id) ON DELETE CASCADE,
    level INT NOT NULL CHECK (level IN (0, 1, 2)),
    slug TEXT,
    meta_data JSONB DEFAULT '{}'::jsonb,
    embedding_id TEXT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT uq_taxonomy_unique UNIQUE (canonical_name, level, parent_id)
);
CREATE INDEX idx_taxonomy_parent_id ON taxonomy(parent_id);
CREATE INDEX idx_taxonomy_level ON taxonomy(level);
CREATE INDEX idx_taxonomy_canonical_name_trgm ON taxonomy USING gin (canonical_name gin_trgm_ops);

-- TAXONOMY ALIAS
CREATE TABLE taxonomy_alias (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alias_name TEXT NOT NULL,
    canonical_taxonomy_id UUID NOT NULL REFERENCES taxonomy(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE UNIQUE INDEX idx_taxonomy_alias_unique ON taxonomy_alias(alias_name, canonical_taxonomy_id);

-- PENDING TAXONOMY
CREATE TABLE pending_taxonomy (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    raw_name TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    suggested_parent_id UUID NULL REFERENCES taxonomy(id),
    suggested_level INT CHECK (suggested_level IN (0, 1, 2)),
    similar_existing_ids TEXT[],
    confidence FLOAT DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_pending_taxonomy_status ON pending_taxonomy(status);
CREATE INDEX idx_pending_taxonomy_canonical_trgm ON pending_taxonomy USING gin (canonical_name gin_trgm_ops);

-- BUSINESS TAXONOMY
CREATE TABLE business_taxonomy (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    business_id UUID NOT NULL,
    taxonomy_id UUID NOT NULL REFERENCES taxonomy(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL DEFAULT 'primary',
    confidence FLOAT DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(business_id, taxonomy_id)
);
CREATE INDEX idx_business_taxonomy_relationship_type ON business_taxonomy(relationship_type);
CREATE INDEX idx_business_taxonomy_source ON business_taxonomy(source);
