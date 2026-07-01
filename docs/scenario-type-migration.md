# Migration 1 — `scenario_type` → `article_scenarios` (Way B)

**Decision:** the ~29 endpoints that scope by `d.scenario_type = :sid` ("ingestion
membership") will switch to **scored membership via `article_scenarios`** ("Way B"),
**gated on a before/after review against a real data copy.**

## Why (plain terms)

A document can belong to several scenarios. Two membership notions exist:
- **Way A — ingestion:** `literature_document.scenario_type`, stamped once when the
  paper was fetched. Single-valued, never updated.
- **Way B — scored:** `article_scenarios (scenario_id, document_id, similarity_score)`,
  the relevance-scoring result. Many-to-many; this is what the dashboard already uses.

`d.scenario_type = :sid` and an `article_scenarios` join return different document
sets, so switching the 29 sites changes what each scenario shows. We want Way B
("every relevant paper, even if found via another scenario"), but we verify the
impact on real data before flipping anything.

## Step 1 — Before/after preview (do this first, zero risk)

`scripts/migration1_scenario_type_diff.py` is **read-only** (SELECT only). It prints,
per scenario:

```
scénario        A (actuel)  B (cible)  communs  perdus(A)  gagnés(B)
sc-a                     3          2        2          1          0
sc-b                     2          0        0          2          0  ⚠ VIDÉ
sc-c                     0          1        0          0          1
```

- **A** = docs under today's definition (`scenario_type`).
- **B** = docs under Way B (`article_scenarios`).
- **perdus(A)** = docs that would leave the scenario; **gagnés(B)** = docs that would join.
- **⚠ VIDÉ** = scenario had ingestion docs but **no** scored membership → Way B would
  empty it. These need a backfill before the switch (see Step 2).

Run it against a **read-only copy/snapshot** of production (a read-only role is enough):
```bash
DATABASE_URL='postgresql+psycopg://USER:PWD@HOST:5432/literev' \
    python3 scripts/migration1_scenario_type_diff.py
```
(Validated locally on Postgres 16; correct counts + ⚠ detection confirmed on a
fixture with a deliberately divergent scenario.)

**Review the output together** before any code lands. If the deltas look right and no
scenario is unexpectedly emptied, proceed.

## Step 2 — Backfill (only if the preview flags ⚠ VIDÉ scenarios)

For any scenario that exists via `scenario_type` but has no `article_scenarios` rows,
create the missing links so Way B doesn't lose those documents:
```sql
INSERT INTO article_scenarios (scenario_id, document_id, similarity_score)
SELECT d.scenario_type, d.id, NULL
FROM literature_document d
WHERE d.scenario_type IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM article_scenarios ars
      WHERE ars.scenario_id = d.scenario_type AND ars.document_id = d.id
  );
```
`similarity_score = NULL` means "member but unscored" — already treated as *not above
threshold* by the canonical predicate (see #115), so this changes membership without
inflating "relevant" counts. Run on staging, re-run the preview, confirm no ⚠ remain.

## Step 3 — Rewrite the 29 sites

Replace the scoping predicate everywhere it appears:
```sql
-- before
WHERE d.scenario_type = :sid
-- after
WHERE EXISTS (SELECT 1 FROM article_scenarios ars
              WHERE ars.document_id = d.id AND ars.scenario_id = :sid)
```
Notes:
- The `ix_article_scenarios_document` / `ix_article_scenarios_scenario` indexes
  (shipped in #122) make the EXISTS cheap.
- Some sites also write `scenario_type` (document insert) — keep writing it for now
  (harmless, and useful as a provenance field) but stop *reading* it for scoping.
- The screening endpoints' `AND scenario_type = :sid` gate becomes the membership
  EXISTS too (this is also what Migration 2 needs).

## Step 4 — Validate & ship

- On staging: re-run the preview (expect A==B per scenario after backfill) and a
  corpus-count parity check; confirm dashboards/search/PRISMA are unchanged except
  for the intended additions.
- Ship behind the same CI as everything else.

## Preview result (run on production, 40 scenarios)

Way A → Way B totals: A=8,656 → B=12,875 (common 4,652; lost 4,004; gained 8,223).
Patterns: most GESICA scenarios are clean supersets; some user scenarios balloon
(e.g. 13→3013); and **15 scenarios are ⚠ VIDÉ** (zero `article_scenarios` rows):
`stroke-detection`, `triage-support`, and 13 `usr-…` scenarios.

**Decision (original): pause and RE-SCORE the 15 first** (so pure Way B isn't just
emptying scenarios that were never scored).

**Decision (updated 2026-06-30): ship the Step 2 backfill instead of gating on a
manual re-score.** Re-scoring the 15 needs prod/server access + OpenAI cost and left
the migration stuck. The backfill (alembic `a7c3e1b9d2f4`) creates the missing
`article_scenarios` rows from `scenario_type` with `similarity_score = NULL`
("member but unscored"), making Way B membership a superset of Way A so the read
switch can no longer empty any scenario. NULL-score rows are never counted as
*relevant* (canonical predicate), so relevance/PRISMA counts don't move; only
`article_scenarios`-based membership converges (the 15 VIDÉ scenarios stop showing
empty). Validated end-to-end on local Postgres 16 (upgrade → downgrade → idempotent
re-upgrade). The `rebuild-corpus` re-score below remains available as an optional
quality pass to turn NULL-score members into scored ones.

### Why the 15 have no scored membership
`article_scenarios` membership is built only by the boolean-query corpus assignment
(`_boolean_corpus_ids` → `_set_scenario_corpus`), which runs inside user `/populate`
(and, historically, a one-off GESICA backfill). `/scenarios/{id}/rerank` only SCORES
rows that already exist — it can't repopulate an empty corpus. The 13 user scenarios
were never populated (or got reset); the 2 GESICA scenarios were missed by the
historical backfill, and there was **no live endpoint to rebuild a GESICA corpus**.

### Re-score mechanism (added)
New endpoint **`POST /scenarios/{id}/rebuild-corpus`** (api-key gated): rebuilds
membership from the scenario's boolean query against the **local DB only (no live
re-ingestion)**, then runs cosine + cross-encoder scoring. Cost ≈ 1 query embedding
per scenario. Works for both GESICA and user scenarios (same `user_scenarios` table).
Progress via `GET /scenarios/{id}/rerank/status`.

Re-score the 15 (run on the server, with the write key). NOTE: hitting the backend
directly on `localhost:8000` uses **no `/api` prefix** (that prefix is added by nginx
only for the public URL).
```bash
# Clean key extraction (strips surrounding quotes if any):
KEY=$(grep -E '^WRITE_API_KEY=' /etc/literev-api.env | cut -d= -f2- | tr -d "\"'")

# Test ONE first, then check it reached "done":
curl -fsS -X POST "http://localhost:8000/scenarios/stroke-detection/rebuild-corpus" -H "X-API-Key: $KEY"; echo
sleep 60; curl -fsS "http://localhost:8000/scenarios/stroke-detection/rerank/status"; echo

# Then the rest:
for sid in triage-support \
  usr-3a5a01dc0c44 usr-438ee5ca28fb usr-5600e99627d2 usr-6a64a31b0fbb \
  usr-7cadc989ea46 usr-868dc56be997 usr-a9498067e6f9 usr-ca0030a574e0 \
  usr-cb537e36f0be usr-e2bb05832b44 usr-ecf0ee84ef7d usr-ed591d707a04 usr-ee71de7f1523; do
    echo "== $sid =="
    curl -fsS -X POST "http://localhost:8000/scenarios/$sid/rebuild-corpus" -H "X-API-Key: $KEY"
    echo; sleep 2
done
# wait a few minutes, then re-run the diff tool — the ⚠ VIDÉ list should shrink/clear.
```

## Status

- [x] Decision: Way B, gated on before/after.
- [x] Read-only preview tool (`scripts/migration1_scenario_type_diff.py`).
- [x] Run preview against production → 15 ⚠ VIDÉ scenarios found.
- [x] Decision: re-score the 15 first; add `rebuild-corpus` endpoint.
- [x] **Backfill shipped** as alembic `a7c3e1b9d2f4` (supersedes the manual
      re-score gate); validated on local PG; auto-applies on deploy.
- [x] Post-deploy: re-ran the diff on production → **0 ⚠ VIDÉ**, `perdus(A)=0`
      everywhere (Way B is a strict superset; nothing is lost). Deltas reviewed
      with the user (the "ballooning" is the intended cross-scored membership;
      a few user scenarios grow, e.g. `usr-d523cedda9aa` 13→3020).
- [x] **Rewrote the 11 scoping reads** (`scenario_type = :param` →
      `article_scenarios` EXISTS): PRISMA stats, screening-progress, pico-bulk,
      kappa, double-blind conflicts, knowledge-graph, RAG `/ask/stream`, plus the
      two screening **write** gates (screen + double-blind resolve — also what
      Migration 2 needs). Validated on local PG (aliased/unaliased SELECT +
      correlated EXISTS in UPDATE; a `scenario_type`-only doc is correctly NOT
      scoped in). Since `perdus(A)=0`, the flip is purely additive in prod.
- [x] **`/search` filter mapping flipped to Way B** (`_build_where`): a
      `scenario_type` filter now emits `EXISTS (article_scenarios ars WHERE
      ars.document_id = d.id AND ars.scenario_id = :scenario_type)` instead of
      `d.scenario_type = :scenario_type`. This was the **last read-path predicate**
      still on Way A; `/search` (and `/ask`) now match corpus/stats/PRISMA/RAG.
      Validated on local PG: identical to Way A where `article_scenarios` mirrors
      `scenario_type` (no cross-scoring), and the correct superset once a document
      is scored into a scenario it wasn't ingested under. Facet values in
      `/filters-options` still come from the `scenario_type` column (which equals
      `article_scenarios.scenario_id` post-backfill), so the dropdown is unchanged.
      (User-scenario `usr-*` ids remain excluded from the facet — a separate product
      decision, untouched.)
- [ ] Document INSERT still **writes** `scenario_type` (provenance, intended) —
      the only remaining use of the column. No reader depends on it now.
- [ ] Later: `DROP COLUMN scenario_type` once the provenance write is dropped and a
      soak period confirms nothing reads it.
- [ ] Then unblock Migration 2 (`screening-status-per-scenario-migration.md`).
