ALTER TABLE document_chunk
    ADD COLUMN IF NOT EXISTS chunk_type TEXT,
    ADD COLUMN IF NOT EXISTS section_label TEXT,
    ADD COLUMN IF NOT EXISTS char_start INTEGER,
    ADD COLUMN IF NOT EXISTS char_end INTEGER,
    ADD COLUMN IF NOT EXISTS token_count INTEGER,
    ADD COLUMN IF NOT EXISTS chunk_weight DOUBLE PRECISION DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS metadata_json JSONB DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_document_chunk_chunk_type ON document_chunk (chunk_type);
CREATE INDEX IF NOT EXISTS idx_document_chunk_chunk_weight ON document_chunk (chunk_weight);
CREATE INDEX IF NOT EXISTS idx_document_chunk_metadata_json ON document_chunk USING GIN (metadata_json);
