# Pipeline Audit — Article → Evidence → Model (2026-06-17)

Scope: the end-to-end path of an article through the LiteRev-Evidence / GESICA
system — **recherche (search/retrieval) → ingestion → chunking/embedding →
screening → PICO/evidence extraction → predictive-model creation**. Focus is the
data pipeline that is supposed to feed model creation.

Method: static read of `main.py` (~11k lines), `frontend/src`, `schema.sql`,
`embed_corpus.py`, `ingest_pipeline.py`, the `*_model.py` modules and
`gesica_scenario_enriched_metadata.py`; **plus a read-only production query**
(server-command run #178) to ground the highest-severity claims in live data.

---

## 0. Headline

**The pipeline's stages run, but they are not wired to one another.** Each stage
works in isolation; the *gates and provenance between stages are missing*:

- Screening does not gate anything downstream — and in production it is barely
  used (4 included / 0 excluded / 73,023 pending).
- `quality_score` is 0 for 99.8 % of the corpus, so the entire "niveau de
  preuve / GRADE" reporting collapses to a single bucket.
- The "predictive model" is **not fit to the corpus** at all: GESICA models are
  hand-coded formulas run on hardcoded demo patients; user-scenario "models" are
  LLM-proposed variables from a 25-article snippet, cached and never recomputed,
  with no provenance back to studies.

Two reassuring live facts: embedding coverage is ~99.99 % and the `UNIQUE(doi)`
+ HNSW indexes exist in prod — so several "silent failure" risks are currently
latent rather than active. They remain real for fresh deploys and active
ingestion.

---

## 1. Live production state (run #178, non-duplicate `literev` corpus)

| Metric | Value | Reading |
|---|---|---|
| Documents (non-dup) | 73,027 | corpus size |
| `quality_score` non-null / > 0 | 73,027 / **157** | populated with 0; real scores for 0.2 % |
| Chunks total / embedded / NULL | 295,021 / 294,978 / **43** | 99.99 % embedded |
| Docs with ≥1 embedded chunk / total-with-chunk | 72,794 / 72,794 | 100 % of chunked docs |
| Docs with **no chunk at all** | **233** (73,027 − 72,794) | invisible to search/RAG |
| Docs with **duplicate `title_abstract` chunks** | **116** | dup-chunk bug, confirmed |
| Screening: pending / included / excluded | **73,023 / 4 / 0** | screening unused |
| PICO eligible (abstract > 50) / with `pico_json` | 49,307 / **49,307** | 100 % of eligible |
| Docs with no usable abstract | ~23,720 (32 %) | never get PICO, still counted |
| `study_design` column non-null | **0** | all design data only in `pico_json` |
| `uq_literature_document_doi` unique index | **present** | ON CONFLICT works in prod |
| `document_chunk_embedding_hnsw` | **present** | vector search is indexed |

---

## 2. Findings by stage (severity-ranked)

Severity = (likelihood × impact on the *evidence/model* output). "Live" = what
the production query shows today.

### Stage A — Recherche & Ingestion

**A1 [High] `ON CONFLICT (doi)` depends on a UNIQUE index that version-controlled
schema/migrations never create.** `_ingest_doc_direct` (`main.py:7456`) and
`create_document` (`main.py:645`) use `ON CONFLICT (doi) WHERE doi IS NOT NULL`.
The biblio migration creates only a *non-unique* index; the unique one
(`uq_literature_document_doi`) exists solely in the ad-hoc `scripts/_phase2_execute.py:52`.
*Live:* the index **exists** in prod, so ingestion works today. *Risk:* any fresh
DB (new env, rebuild) raises `no unique or exclusion constraint matching the ON
CONFLICT` on every insert, which is then swallowed per-article → **zero rows
ingested, silently**. **Fix:** add the unique index to `schema.sql` + a real
migration; the two pipeline-critical tables (`article_scenarios`,
`user_scenarios`) are also absent from `schema.sql` — add them.

**A2 [High] `search/live` background ingest bypasses the populate concurrency
lock.** `populate_user_scenario` guards via `_populate_jobs_lock`
(`main.py:9225-9233`); `search_live` spawns `threading.Thread(target=
_run_user_scenario_populate, …)` directly (`main.py:2016-2022`) with no lock and
no "already running" check, and passes `filters={}` (ignoring stored filters).
*Impact:* concurrent populate jobs on the same scenario run overlapping
DELETE/UPDATE post-processing (`main.py:8139, 8180`) → racing writes, corrupted
counters, links deleted out from under the other job. **Fix:** route both paths
through the same locked job-launcher.

**A3 [High] No transaction spans document + chunk + scenario-link → orphan rows.**
`_ingest_doc_direct` uses three separate `engine.begin()` blocks (doc
`:7447`, chunk `:7472`, link `:7528`). A crash between them leaves a document
with no chunk (never searchable) or unlinked. *Live:* **233 docs have no chunk**
— exactly this partial state. **Fix:** wrap doc+chunk in one transaction; make
the chunk insert part of the document insert path.

**A4 [Med] Dedup hole for DOI-less records → cross-source duplicates.** The only
guards are existing `external_id` (per-source, differs across sources) and
`ON CONFLICT (doi)`. A DOI-less paper from two sources → two rows; `title_hash`/
`canonical_id` are only set by the offline `deduplicate_corpus.py`. **Fix:**
compute `title_hash` at insert and dedup on it when DOI is NULL.

**A5 [Med] Pervasive silent per-source failure.** Every per-article insert
swallows errors to a counter, often with no log (OpenAlex `:7711`, Crossref
`:7768`, EuropePMC `:7830`, preprints `:7889`, prospero `:7978`); abstract-parse
failures are bare `except: pass` (`:7695, 7751, 7811`). A systematically failing
source yields a "done" job with silent data loss. **Fix:** log at warning with
source+error; surface per-source error counts in job status.

**A6 [Med] Inconsistent caps / rate-limit handling across sources.**
`max_results` is `500` by default but `populate_user_scenario` passes `100000`
(`:9214`) while the docstring says "1000/source" (`:7499`); the *live* search
path has NCBI 429 backoff (`_ncbi_get :1604`) but the *populate* path does not
(`_fetch_pubmed :7581`); preprints silently cap at a 90-day window (`:7847`).
*Impact:* uneven coverage and silent partial corpora under rate limiting.

**A7 [Low] `in_local_db` dedup compares DOI against `external_id`,** which is the
DOI only for Crossref (`:1910-1913, 1463-1478`). Live results already in the
corpus are reported as "new" → redundant re-ingestion. **Fix:** compare against
the real `doi` column.

**A8 [Low] Ingest never sets `pmid`, `study_design`, `keywords`, `open_access`,
`language`.** `_ingest_doc_direct` sets only 11 fields (`:7449-7462`). *Live:*
`study_design` column is **100 % NULL**; PubMed `pmid` is stored only in
`external_id`/`url`. Downstream code papers over this by reading
`pico_json->>'study_design'` first — but anything reading the column gets NULL.

### Stage B — Chunking & Embedding

**B1 [Med, latent] Chunks are inserted without embeddings; embedding is a
deferred best-effort UPDATE.** `embedding vector(1536)` is nullable
(`schema.sql:67`); the title_abstract chunk is inserted with no embedding
(`:7471-7485`) and filled later by a separate step that is **skipped entirely if
`OPENAI_API_KEY` is absent** (`:8806-8807`). Failed embed batches are swallowed
to `warning` with no retry/backoff (`:8795-8797, 497-498`). *Live:* coverage is
99.99 % (worker has caught up), so impact is currently small — but during active
ingestion the corpus the user sees in stats ≠ the corpus the LLM reasons over.
**Fix:** add backoff+retry; expose an embed-coverage health metric.

**B2 [Med] "Ingested vs embedded vs shown" use three different denominators that
never reconcile.** `/search` count filters `embedding IS NOT NULL`
(`:1320-1321`) while the display query UNIONs unembedded docs back in
(`:1231-1233`); KG filters embedded-only for both nodes and `n_total`
(`:6189, 6218-6219`); clustering includes all-with-abstract and silently
degrades unembedded docs to TF-IDF (`:4395-4405, 4272-4303`). *Impact:* totals
and rows disagree whenever coverage < 100 %. **Fix:** one canonical "eligible
doc" definition shared across surfaces.

**B3 [Med] Duplicate `title_abstract` chunks.** The chunk insert has no
`ON CONFLICT` / uniqueness on `(document_id, chunk_type)` (`:7472-7485`,
`schema.sql:62-80`). *Live:* **116 docs** have >1 title_abstract chunk → those
docs are double-counted in similarity search and chunk stats. **Fix:** unique
index on `(document_id, chunk_index, chunk_type)`; de-dupe existing 116.

**B4 [Med] `embed_corpus.py` writes a 1536-d zero vector for empty content**
(`embed_corpus.py:72-73`) — a non-NULL "valid" vector that passes every
`embedding IS NOT NULL` filter and produces degenerate cosine distances.
*Unverified live* (the `l2_norm` probe hit an ambiguous-function error). **Fix:**
skip empty content instead of storing a zero vector; audit for existing ones.

**B5 [Low/Med, fresh-deploy] HNSW index + dim/model are not in `schema.sql`.**
The ANN index is commented out in `schema.sql:96-99`; it lives only in
`scripts/_build_hnsw.py`. `vector(1536)` and `text-embedding-3-small` are
hardcoded in ~10 places with no central constant. *Live:* the HNSW index
**exists** in prod. *Risk:* a fresh env has no ANN index → sequential scans; a
model swap silently breaks all writes at the CAST. **Fix:** version the index;
centralize the model+dim constant.

### Stage C — Screening & Evidence/PICO Extraction

**C1 [High] Screening status gates nothing downstream — and is effectively
unused.** RAG retrieval filters only duplicates, not `screening_status`
(`:4616-4618`); generic RAG even *widens* to include `similarity IS NULL OR
screening='included'` and merely sorts included-first (`:11149-11167`). PICO
extraction has no screening filter (`:504-513, 4964-4970, 5065-5083`). *Live:*
**73,023 pending / 4 included / 0 excluded** — so the entire evidence base is an
*unscreened* search dump; the PRISMA stage is cosmetic. **Fix:** decide the
intended contract (does the model use *included* only?) and enforce it in every
downstream query; surface screening progress as a gate.

**C2 [High] `quality_score` is effectively unpopulated → evidence grading is
meaningless.** No `SET quality_score` exists anywhere; the column carries its
default. *Live:* non-null for all 73,027 but **> 0 for only 157**. The GRADE-like
bucket logic (`:5647-5651, 9755-9757`) maps `quality_score IS NOT NULL` (i.e. 0)
to **"Faible"**, so ~99.8 % of the corpus shows as weak evidence and 0 % as
"Non évaluée" — the distribution is uninformative, not just empty. **Fix:**
compute a real quality score (study-design + sample-size + journal signals, or a
risk-of-bias proxy) and write it; until then, stop presenting the GRADE chart as
meaningful.

**C3 [Med] PICO extraction is not idempotent; concurrent writers can clobber.**
On-demand "extrait (ou re-extrait)" unconditionally overwrites
(`:5019-5028`); batch re-processes anything with `pico_confidence < 0.5`
(`:5070`); background worker, manual, and batch all target the same `pico_json`
with no coordination. **Fix:** skip-if-present guard + a single writer or a
version/lock.

**C4 [Med] LLM/JSON-parse failures are silently swallowed.** Background per-doc
drops to `logger.debug` (`:442-444`); missing required keys return `None` with
no log (`:426-427`); `max_tokens=400` can truncate JSON. *Live:* PICO coverage is
100 % of eligible, so current loss is ~0 — but failures are invisible by design.
**Fix:** log at warning; add one repair/retry.

**C5 [Med] `study_design` normalization is display-only and the column is empty.**
The LLM emits free text; `_STUDY_DESIGN_CASE` (`main.py:182`) normalizes only at
aggregation (`:5604, 9731`), order-dependent with an `'Autre'` catch-all. *Live:*
the `study_design` column is **100 % NULL**, so all design info comes from
`pico_json->>'study_design'`; any reader of the raw column gets nothing, and
readers of `pico_json` get unnormalized free text. **Fix:** normalize at write
time into the column; make it the single source.

**C6 [Med] Evidence/stats derive from incompletely-processed docs.** Abstract-less
docs (~32 %) never get PICO yet still count in study-design/evidence
distributions (`:5594, 5645`); RAG requires embeddings. *Impact:* the corpus in
the stats ≠ the corpus in the answer. **Fix:** show denominators ("PICO on N of M")
explicitly.

**C7 [Med] Double-blind resolution overwrites manual screening and mixes
`pending` into kappa.** On both-reviewers-ruled it forces `screening_status`
(`:5861-5872`), clobbering a prior single-reviewer decision; kappa treats
`pending` as a real category and coerces unknowns to it (`:5912-5933`). **Fix:**
exclude `pending` from kappa; reconcile the two screening workflows.

**C8 [Low] Dead `scenario_kappa_cache` read (`if False`, `:2766-2769`)** — the
cross-scenario dashboard kappa is always null; nothing ever writes that table.

### Stage D — Model creation

**D1 [High] GESICA "models" are hand-coded formulas on hardcoded demo patients,
with static variable metadata.** `get_scenario_model_status` imports a
`*_model.py` and calls `predict_demo()` (`:4161-4182`), which runs fixed
fictional cases (e.g. `clinical_deterioration_model.py:309-373`). The displayed
variables/algorithm/`plugged` flags come from a static 1892-line dict
(`gesica_scenario_enriched_metadata.py:40-93`), served verbatim
(`main.py:3979, 3983`). *Nothing* derives from the corpus; the UI labels it
"Statut Live du Modèle / Dernière valeur live calculée." **Fix:** relabel as a
reference/demo, or wire real inputs; stop claiming "live."

**D2 [High] User-scenario "model" = LLM-invented variables with no provenance,
cached forever.** `_generate_variables_from_pico` feeds the **first 25** PICO
articles (`:10811`) to gpt-4.1 and asks it to *name* predictors, algorithm,
thresholds, and an `evidence_level` it *estimates* (`:10847-10861`). No article
IDs/effect sizes attach to any variable. `get_scenario_variables` returns the
cached JSON forever with **no invalidation when the corpus changes**
(`:10948-10964`). "Validation" is a boolean flip accepting arbitrary client JSON
(`:10977-11001`). **Fix:** attach study provenance per variable; recompute on
corpus change; real validation or honest labeling.

**D3 [Med] Model evidence selection includes screening-excluded and unscored
articles.** `_get_above_threshold_articles` keeps a doc if `screening='included'
OR similarity >= threshold OR similarity IS NULL` (`:10389-10393`) — so excluded
and never-reranked docs feed the model. Combined with C1 (nothing screened),
the "evidence base" is the raw corpus. **Fix:** included-only once screening is
enforced.

**D4 [Low] Silent green fallback.** If model import/`predict_demo` throws,
`model_result` stays None and status defaults to **green/"Normal"**
(`:4159-4197`); scenarios absent from `MODEL_ENDPOINT_MAP` always show green —
a broken model is indistinguishable from a healthy one.

---

## 3. Prioritized remediation

**Tier 1 — correctness of the evidence the model rests on**
1. C1 — enforce a screening gate (or explicitly document that the corpus is
   unscreened) in every downstream query; make screening usable.
2. C2 — compute and write a real `quality_score`; until then stop presenting the
   GRADE distribution as meaningful.
3. D1/D2 — stop labeling hardcoded/LLM artifacts as "live model"; add evidence
   provenance and corpus-change invalidation.

**Tier 2 — pipeline integrity**
4. A3 + B3 — single transaction for doc+chunk; unique index on chunks; clean the
   233 chunkless docs and 116 duplicate chunks.
5. A1 — version `UNIQUE(doi)`, the HNSW index, and the `article_scenarios` /
   `user_scenarios` tables into `schema.sql` + migrations.
6. A2 — one locked job-launcher for both populate paths.

**Tier 3 — observability / hygiene**
7. A5/C4 — replace silent counters with logged, surfaced per-source/per-doc
   errors.
8. B2/C6 — one canonical "eligible doc" definition; show denominators.
9. C5 — normalize `study_design` at write time; C7 reconcile kappa/screening;
   C8 remove dead kappa-cache path; D4 fail loud instead of green.

---

*No application code was changed by this audit. One read-only production query
(server-command run #178) was executed to ground the numbers in §1.*
