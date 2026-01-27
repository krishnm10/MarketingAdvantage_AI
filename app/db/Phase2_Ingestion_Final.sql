-- ================================================
-- PHASE 2: INGESTION SYSTEM (FINAL)
-- Fully compatible with Python models
-- ================================================

CREATE TABLE ingested_file (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    business_id UUID NULL,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_path TEXT NULL,
    source_url TEXT NULL,
    meta_data JSONB DEFAULT '{}'::jsonb,
    total_chunks INT DEFAULT 0,
    unique_chunks INT DEFAULT 0,
    duplicate_chunks INT DEFAULT 0,
    dedup_ratio FLOAT DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'processed', 'error')),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_ingested_file_business_id ON ingested_file(business_id);
CREATE INDEX idx_ingested_file_file_type ON ingested_file(file_type);

CREATE TABLE global_content_index (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    semantic_hash VARCHAR(256) UNIQUE NOT NULL,
    cleaned_text TEXT NOT NULL,
    raw_text TEXT NULL,
    tokens INT DEFAULT 0,
    embedding_model VARCHAR(128) DEFAULT 'BAAI/bge-large-en',
    confidence_avg FLOAT DEFAULT 0 CHECK (confidence_avg >= 0 AND confidence_avg <= 1),
    occurrence_count INT DEFAULT 1,
    business_id UUID NULL,
    first_seen_file_id UUID NULL REFERENCES ingested_file(id),
    meta_data JSONB DEFAULT '{}'::jsonb,
    source_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_gci_semhash ON global_content_index(semantic_hash);
CREATE INDEX idx_gci_business_id ON global_content_index(business_id);
CREATE INDEX idx_gci_source_type ON global_content_index(source_type);

CREATE TABLE ingested_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_id UUID NOT NULL REFERENCES ingested_file(id) ON DELETE CASCADE,
    business_id UUID NULL,
    global_content_id UUID NULL REFERENCES global_content_index(id),
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    cleaned_text TEXT NOT NULL,
    tokens INT NOT NULL,
    semantic_hash TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    source_type TEXT NOT NULL 
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_of UUID NULL,
    similarity_score FLOAT NULL,
    duplicate_percentage FLOAT NULL,
    meta_data JSONB DEFAULT '{}'::jsonb,
	reasoning_ingestion JSONB DEFAULT '{}'::jsonb,
	validation_layer JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(file_id, chunk_index)
);
CREATE INDEX idx_ingested_content_file_id ON ingested_content(file_id);
CREATE INDEX idx_ingested_content_business_id ON ingested_content(business_id);
CREATE INDEX idx_ingested_content_source_type ON ingested_content(source_type);
CREATE INDEX idx_ingested_content_confidence ON ingested_content(confidence);
CREATE INDEX idx_ingested_content_semantic_hash ON ingested_content(semantic_hash);

CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_ingested_file BEFORE UPDATE ON ingested_file
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trg_update_ingested_content BEFORE UPDATE ON ingested_content
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER trg_update_global_content BEFORE UPDATE ON global_content_index
FOR EACH ROW EXECUTE FUNCTION update_timestamp();
