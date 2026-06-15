-- Migration : Ajout des colonnes bibliographiques complètes à literature_document
-- À exécuter sur db-01 : psql -U literev -d literev -f migrate_add_bibliographic_columns.sql
-- Idempotent : utilise ADD COLUMN IF NOT EXISTS

BEGIN;

-- Auteurs (liste séparée par des virgules ou JSON array)
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS authors TEXT;

-- DOI (Digital Object Identifier)
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS doi TEXT;

-- Revue / Journal de publication
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS journal TEXT;

-- Mots-clés (séparés par des virgules)
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS keywords TEXT;

-- Langue de publication (ISO 639-1 : 'en', 'fr', 'de', etc.)
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS language VARCHAR(10);

-- Type d'étude (RCT, observational, systematic review, meta-analysis, etc.)
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS study_design TEXT;

-- Taille de l'échantillon (nombre de patients/sujets)
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS sample_size INTEGER;

-- Pays de l'étude (ISO 3166-1 alpha-2 ou nom complet)
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS country TEXT;

-- Nombre de citations (depuis OpenAlex ou CrossRef)
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS citation_count INTEGER;

-- Accès ouvert (true = open access, false = payant, null = inconnu)
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS open_access BOOLEAN;

-- PMID (PubMed ID) pour les articles issus de PubMed
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS pmid TEXT;

-- OpenAlex ID pour les articles issus d'OpenAlex
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS openalex_id TEXT;

-- Volume et numéro de la revue
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS volume TEXT;
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS issue TEXT;

-- Pages
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS pages TEXT;

-- ISSN de la revue
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS issn TEXT;

-- Type de publication (journal-article, conference-paper, preprint, book-chapter, etc.)
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS publication_type TEXT;

-- Résumé structuré (BACKGROUND, METHODS, RESULTS, CONCLUSIONS) en JSON
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS structured_abstract JSONB;

-- MeSH terms (PubMed Medical Subject Headings) en JSON array
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS mesh_terms JSONB;

-- Affiliations des auteurs
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS affiliations TEXT;

-- Financement / Funding
ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS funding TEXT;

-- Index pour les nouvelles colonnes les plus utilisées
CREATE INDEX IF NOT EXISTS idx_literature_document_doi ON literature_document(doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_literature_document_pmid ON literature_document(pmid) WHERE pmid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_literature_document_journal ON literature_document(journal) WHERE journal IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_literature_document_language ON literature_document(language) WHERE language IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_literature_document_open_access ON literature_document(open_access) WHERE open_access IS NOT NULL;

COMMIT;

-- Vérification
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'literature_document' 
ORDER BY ordinal_position;
