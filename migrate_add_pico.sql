-- Migration : ajout de la colonne pico_json pour stocker l'extraction PICO par article
-- Exécuter avec :
--   PGPASSWORD="$DB_PASSWORD" psql -U literev -d literev -h 10.10.1.10 -f migrate_add_pico.sql

BEGIN;

-- Colonne PICO JSON (stocke {P, I, C, O, study_design, pico_confidence})
ALTER TABLE literature_document
    ADD COLUMN IF NOT EXISTS pico_json    JSONB    DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS pico_extracted_at TIMESTAMPTZ DEFAULT NULL;

-- Index GIN pour requêtes sur le contenu PICO
CREATE INDEX IF NOT EXISTS idx_literature_document_pico
    ON literature_document USING GIN (pico_json)
    WHERE pico_json IS NOT NULL;

COMMIT;

-- Vérification
SELECT
    COUNT(*)                                            AS total_docs,
    COUNT(*) FILTER (WHERE pico_json IS NOT NULL)       AS with_pico,
    COUNT(*) FILTER (WHERE pico_json IS NULL)           AS without_pico
FROM literature_document
WHERE project_context = 'gesica';
