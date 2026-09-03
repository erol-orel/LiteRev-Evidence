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

-- ──────────────────────────────────────────────────────────
-- Table: article_scenarios  (lien N-N document ↔ scénario)
-- ──────────────────────────────────────────────────────────
-- Cette table était ABSENTE de ce fichier alors que main.py l'interroge partout
-- (≈100 références). Elle n'existait qu'en production, posée historiquement par un
-- script ad hoc : les migrations Alembic se contentent de lui AJOUTER des colonnes et
-- se sautent elles-mêmes quand elle manque, et le DDL de démarrage l'ALTER directement.
-- Sur une base neuve, cet ALTER échouait et faisait ANNULER (rollback transactionnel)
-- la création de user_scenarios faite juste avant dans le même bloc — d'où une base
-- inutilisable où /health répondait pourtant 200.
-- Garde-fou : tests/test_fresh_db_bootstrap.py.
CREATE TABLE IF NOT EXISTS article_scenarios (
    scenario_id        TEXT   NOT NULL,
    document_id        BIGINT NOT NULL,
    similarity_score   DOUBLE PRECISION,
    rerank_score       FLOAT,                  -- rerank cross-encoder (ordre d'affichage)
    cluster_id         INTEGER,
    cluster_label      TEXT,
    -- Screening par scénario (migration c8d4e2f1a9b3)
    screening_status   TEXT,
    screening_reason   TEXT,
    screening_notes    TEXT,
    screened_at        TIMESTAMP,
    -- Double lecture aveugle par scénario (migration e2f6a8b3c5d7)
    reviewer_1_status  VARCHAR(20),
    reviewer_1_reason  TEXT,
    reviewer_2_status  VARCHAR(20),
    reviewer_2_reason  TEXT,
    kappa_resolved     BOOLEAN DEFAULT FALSE,
    kappa_final_status VARCHAR(20),
    assigned_at        TIMESTAMP,
    CONSTRAINT article_scenarios_pkey PRIMARY KEY (scenario_id, document_id)
);

-- Pas de clé étrangère vers literature_document : la production n'en a pas (les
-- fixtures d'intégration le notent explicitement), et en ajouter une ici ferait
-- diverger une base neuve de la base existante.
CREATE INDEX IF NOT EXISTS ix_article_scenarios_document
    ON article_scenarios (document_id);
CREATE INDEX IF NOT EXISTS ix_article_scenarios_scen_screen
    ON article_scenarios (scenario_id, screening_status);
CREATE INDEX IF NOT EXISTS ix_article_scenarios_scen_kappa
    ON article_scenarios (scenario_id, kappa_final_status);

-- ─────────────────────────────────────────────────────────────────────────────
-- Comptabilité des appels OpenAI (migration a7c2e9b5d413 ; cf. llm_usage.py)
-- L'application appelait l'API depuis une trentaine d'endroits sans jamais lire
-- `response.usage` : la seule trace d'une dépense était la facture. Une ligne par
-- appel, étiquetée de la fonction appelante, rend la question « qui dépense »
-- répondable en une requête (/llm-usage).
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS llm_usage (
    id                BIGSERIAL PRIMARY KEY,
    ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
    purpose           TEXT        NOT NULL,
    model             TEXT        NOT NULL,
    prompt_tokens     INTEGER     NOT NULL DEFAULT 0,
    completion_tokens INTEGER     NOT NULL DEFAULT 0,
    total_tokens      INTEGER     NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_llm_usage_ts ON llm_usage (ts DESC);
CREATE INDEX IF NOT EXISTS idx_llm_usage_purpose_ts ON llm_usage (purpose, ts DESC);
