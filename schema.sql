-- ============================================================
-- LiteRev-Evidence — Schéma PostgreSQL
-- Généré automatiquement par generate_schema.py
-- Enrichi manuellement : type vector, séquences, trigger function, index GIN
-- ============================================================

-- Extensions requises
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ──────────────────────────────────────────────────────────
-- Séquences
-- ──────────────────────────────────────────────────────────
CREATE SEQUENCE IF NOT EXISTS literature_document_id_seq
    START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

CREATE SEQUENCE IF NOT EXISTS document_chunk_id_seq
    START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

-- ──────────────────────────────────────────────────────────
-- Table: alembic_version
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- ──────────────────────────────────────────────────────────
-- Table: literature_document
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS literature_document (
    id              BIGINT      NOT NULL DEFAULT nextval('literature_document_id_seq'::regclass),
    external_id     TEXT,
    source          TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    abstract        TEXT,
    year            INTEGER,
    url             TEXT,
    created_at      TIMESTAMP   DEFAULT now(),
    updated_at      TIMESTAMP   DEFAULT now(),
    project_context VARCHAR(32) DEFAULT 'eva'::character varying,
    source_type     VARCHAR(64),
    disease_or_condition  VARCHAR(128),
    scenario_type         VARCHAR(128),
    geographic_scope      VARCHAR(128),
    evidence_category     VARCHAR(64),
    CONSTRAINT literature_document_pkey PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS ix_literature_document_external_id
    ON literature_document (external_id);
CREATE INDEX IF NOT EXISTS ix_literature_document_project_context
    ON literature_document (project_context);
CREATE INDEX IF NOT EXISTS ix_literature_document_scenario_type
    ON literature_document (scenario_type);
CREATE INDEX IF NOT EXISTS ix_literature_document_year
    ON literature_document (year);

-- ──────────────────────────────────────────────────────────
-- Table: document_chunk
-- ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_chunk (
    id            BIGINT             NOT NULL DEFAULT nextval('document_chunk_id_seq'::regclass),
    document_id   BIGINT             NOT NULL,
    chunk_index   INTEGER            NOT NULL,
    content       TEXT               NOT NULL,
    embedding     vector(1536),                          -- pgvector : dimension OpenAI text-embedding-3-small
    created_at    TIMESTAMP          DEFAULT now(),
    search_vector TSVECTOR,
    chunk_type    TEXT,
    section_label TEXT,
    char_start    INTEGER,
    char_end      INTEGER,
    token_count   INTEGER,
    chunk_weight  DOUBLE PRECISION   DEFAULT 1.0,
    metadata_json JSONB              DEFAULT '{}'::jsonb,
    CONSTRAINT document_chunk_pkey PRIMARY KEY (id),
    CONSTRAINT document_chunk_document_id_fkey
        FOREIGN KEY (document_id) REFERENCES literature_document (id) ON DELETE CASCADE
);

-- Index B-tree standard
CREATE INDEX IF NOT EXISTS document_chunk_doc_idx
    ON document_chunk (document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_document_chunk_chunk_type
    ON document_chunk (chunk_type);
CREATE INDEX IF NOT EXISTS idx_document_chunk_chunk_weight
    ON document_chunk (chunk_weight);

-- Index GIN pour recherche full-text et JSONB
CREATE INDEX IF NOT EXISTS ix_document_chunk_search_vector
    ON document_chunk USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS idx_document_chunk_metadata_json
    ON document_chunk USING GIN (metadata_json);

-- Index IVFFlat pour recherche vectorielle (pgvector)
-- Note : à créer APRÈS l'ingestion initiale du corpus (>= 1000 lignes)
-- CREATE INDEX IF NOT EXISTS document_chunk_embedding_ivfflat
--     ON document_chunk USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ──────────────────────────────────────────────────────────
-- Fonction trigger : mise à jour automatique du search_vector
-- ──────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION document_chunk_search_vector_update()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', coalesce(NEW.section_label, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(NEW.content, '')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ──────────────────────────────────────────────────────────
-- Triggers
-- ──────────────────────────────────────────────────────────
DROP TRIGGER IF EXISTS trg_document_chunk_search_vector ON document_chunk;
CREATE TRIGGER trg_document_chunk_search_vector
    BEFORE INSERT OR UPDATE ON document_chunk
    FOR EACH ROW EXECUTE FUNCTION document_chunk_search_vector_update();
