-- Migration : Ajout des colonnes de déduplication à literature_document
-- ─────────────────────────────────────────────────────────────────────────────
-- Commande d'exécution (sans prompt de mot de passe) :
--   PGPASSWORD="MyNewStrongPassword!" psql -U literev -d literev -h 10.10.1.10 -f /opt/literev-api/migrate_add_dedup_columns.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- Idempotent : utilise ADD COLUMN IF NOT EXISTS

BEGIN;

-- Indique si ce document est un doublon (marqué lors de la déduplication)
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS is_duplicate BOOLEAN DEFAULT FALSE;

-- Pointe vers le document maître (canonical) si ce document est un doublon
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS canonical_id BIGINT REFERENCES literature_document(id) ON DELETE SET NULL;

-- Score de qualité du document (0-100) : full-text > abstract, plus de citations = meilleur score
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS quality_score FLOAT DEFAULT 0.0;

-- Hash du titre normalisé pour la déduplication rapide
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS title_hash TEXT;

-- Index pour la déduplication
CREATE INDEX IF NOT EXISTS idx_literature_document_is_duplicate ON literature_document(is_duplicate) WHERE is_duplicate = TRUE;
CREATE INDEX IF NOT EXISTS idx_literature_document_canonical_id ON literature_document(canonical_id) WHERE canonical_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_literature_document_title_hash ON literature_document(title_hash) WHERE title_hash IS NOT NULL;

COMMIT;

-- Vérification
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'literature_document'
  AND column_name IN ('is_duplicate', 'canonical_id', 'quality_score', 'title_hash')
ORDER BY column_name;
