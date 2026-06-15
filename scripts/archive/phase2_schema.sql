ALTER TABLE literature_document
  ADD COLUMN IF NOT EXISTS project_context varchar(32) NOT NULL DEFAULT 'eva',
  ADD COLUMN IF NOT EXISTS source_type varchar(64),
  ADD COLUMN IF NOT EXISTS disease_or_condition varchar(128),
  ADD COLUMN IF NOT EXISTS scenario_type varchar(128),
  ADD COLUMN IF NOT EXISTS geographic_scope varchar(128),
  ADD COLUMN IF NOT EXISTS evidence_category varchar(64);

ALTER TABLE document_chunk
  ADD COLUMN IF NOT EXISTS search_vector tsvector;

UPDATE document_chunk
SET search_vector = to_tsvector('simple', coalesce(content, ''))
WHERE search_vector IS NULL;

CREATE INDEX IF NOT EXISTS ix_document_chunk_search_vector
  ON document_chunk
  USING gin (search_vector);

CREATE OR REPLACE FUNCTION document_chunk_search_vector_update() RETURNS trigger AS $$
BEGIN
  NEW.search_vector := to_tsvector('simple', coalesce(NEW.content, ''));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_document_chunk_search_vector ON document_chunk;

CREATE TRIGGER trg_document_chunk_search_vector
BEFORE INSERT OR UPDATE OF content ON document_chunk
FOR EACH ROW
EXECUTE FUNCTION document_chunk_search_vector_update();
