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

## Status

- [x] Decision: Way B, gated on before/after.
- [x] Read-only preview tool (`scripts/migration1_scenario_type_diff.py`).
- [ ] Run preview against a production copy (needs a snapshot / read-only DSN).
- [ ] Backfill if ⚠ scenarios appear.
- [ ] Rewrite the 29 sites + validate on staging.
- [ ] Then unblock Migration 2 (`screening-status-per-scenario-migration.md`).
