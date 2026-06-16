# One-off production maintenance operations

These scripts were run **once** against the production database (read DB_URL from
the environment) via the audited `server-command` workflow. They are kept as an
audit trail. They are not part of the app runtime.

All are idempotent or guarded; the destructive one (`_phase2_execute.py`) creates
in-DB backup tables before deleting.

## 2026-06-16 — corpus de-duplication + vector index (audit C2/C3)

Context: audit found 8,957 duplicate-DOI groups in `literature_document` (no unique
constraint) and no ANN index on `document_chunk.embedding` (323k×1536, 5 GB).

| Script | Purpose | Result |
|---|---|---|
| `_build_hnsw.py` | Build HNSW index on `document_chunk.embedding` (`vector_cosine_ops`, CONCURRENTLY) | built in 111s, valid, 2424 MB |
| `_dedup_analyze.py` | Read-only impact analysis of DOI/PMID duplicates | 10,688 DOI dups; 2,850 links; 299 unique links to repoint |
| `_softdedup.py` | Phase 1 (reversible): mark dup rows `is_duplicate=TRUE`, set `canonical_id` | 10,688 DOI rows flagged; 0 remaining non-dup DOI groups |
| `_phase2_preflight.py` | Read-only: FK delete behavior + post-delete uniqueness feasibility | FKs CASCADE/SET NULL; UNIQUE(doi) feasible; UNIQUE(pmid) not (21 residual) |
| `_phase2_execute.py` | Phase 2 (destructive): backup → repoint links → delete dups → add `UNIQUE(doi)` | deleted 10,758 docs (cascade ~31.7k chunks); `uq_literature_document_doi` created; backups in `_dedup_bak_*` |

### Follow-ups owed
- Ingest paths (`POST /documents`, bulk scripts) need `ON CONFLICT (doi) ... DO NOTHING`
  / IntegrityError handling now that `UNIQUE(doi)` is enforced.
- `UNIQUE(pmid)` not added — 21 residual PMID groups (differing/null DOI).
- Drop `_dedup_bak_documents` / `_dedup_bak_chunks` / `_dedup_bak_links` once verified.
- `VACUUM ANALYZE literature_document, document_chunk` to reclaim space + refresh stats.
