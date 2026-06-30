# Migration plan — per-scenario `screening_status`

**Status:** proposal (no code yet — review before implementation)
**Author:** generated for review
**Scope:** move PRISMA screening decisions from *global per-document* to *per (scenario, document)*.

---

## 1. Why

Today `literature_document.screening_status` (`included` / `excluded` / `pending` / `NULL`)
is a **single value per document**. But a document can belong to several scenarios
through the many-to-many `article_scenarios` table. So a screening decision made in
one scenario applies to the document **everywhere**.

This is visible in the screening endpoints — they accept a `scenario_id` but write a
global value:

```python
# main.py — POST /gesica/scenarios/{scenario_id}/articles/{article_id}/screen
UPDATE literature_document
SET screening_status = :status, screening_reason = :reason, screening_notes = :notes
WHERE id = :article_id
  AND project_context = 'literev'
  AND scenario_type = :scenario_id      -- gated by ingestion-scenario…
                                        -- …but the column it sets is global
```

Consequences:
- Excluding a paper in scenario A also excludes it in scenario B.
- The write is gated by `scenario_type` (ingestion scenario), which is itself not the
  same as `article_scenarios` membership (see the companion note on `scenario_type`),
  so a paper scored into a scenario it was *not* ingested for can't be screened there
  at all.

**Goal:** a screening decision is scoped to the (scenario, document) pair.

## 2. Current footprint (what the migration touches)

| Area | Where | Notes |
|------|-------|-------|
| Column | `literature_document.screening_status` (+ `screening_reason`, `screening_notes`) | global today |
| Write endpoints | `POST /gesica/scenarios/{id}/articles/{aid}/screen` (main.py ~6284), `POST /user-scenarios/{id}/articles/{aid}/screen` (~11522) | both `UPDATE literature_document … WHERE id=…` |
| Other writers | living-review / bulk paths (main.py ~6543, ~6683) | set status during ingestion/auto-screening |
| Readers | **~118 references** to `screening_status` across main.py | the canonical relevant-subset predicate, corpus stats, PRISMA funnel, briefs, both PDFs, search counts, RAG filters |
| Join table | `article_scenarios (scenario_id, document_id, similarity_score, assigned_at)` | target home for the per-scenario status |
| Frontend | `ScenarioDetailPage.tsx`, `lib/api.ts` | screening UI + the `screen()` API call |

The **canonical predicate** (repeated across the readers) is:
```sql
d.is_duplicate IS NOT TRUE
AND d.screening_status IS DISTINCT FROM 'excluded'
AND (d.screening_status = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
```
After the migration this must read the **per-scenario** status. Because it already
joins `article_scenarios ars`, the rewrite is mechanical once the column moves.

## 3. Target state

- New columns on `article_scenarios`: `screening_status`, `screening_reason`,
  `screening_notes`, `screened_at`.
- Screening writes target the `(scenario_id, document_id)` row.
- Readers use `ars.screening_status` (per scenario), via a shared SQL helper so the
  canonical predicate is defined **once**.
- `literature_document.screening_status` is retained (read-only) until cutover is
  proven, then dropped in a later release.

## 4. Phased rollout (each phase independently shippable & reversible)

### Phase 0 — Prerequisites
- A **staging database** loaded from a production snapshot. None of this should be
  validated only on synthetic data — the backfill correctness depends on real
  multi-scenario membership.
- A verified backup / PITR window before any DDL on production.
- A reusable count-parity script (see §6) run on staging before/after each phase.

### Phase 1 — Additive schema + backfill (non-destructive)
```sql
ALTER TABLE article_scenarios
  ADD COLUMN IF NOT EXISTS screening_status TEXT,
  ADD COLUMN IF NOT EXISTS screening_reason TEXT,
  ADD COLUMN IF NOT EXISTS screening_notes  TEXT,
  ADD COLUMN IF NOT EXISTS screened_at      TIMESTAMP;

-- Backfill: copy each document's current global decision onto its scenario links.
UPDATE article_scenarios ars
SET screening_status = d.screening_status,
    screening_reason = d.screening_reason,
    screening_notes  = d.screening_notes
FROM literature_document d
WHERE d.id = ars.document_id
  AND ars.screening_status IS DISTINCT FROM d.screening_status;

CREATE INDEX IF NOT EXISTS ix_article_scenarios_scen_screen
  ON article_scenarios (scenario_id, screening_status);
```
- Global column unchanged → zero behaviour change. Fully reversible (`DROP COLUMN`).
- **Decision point:** for documents in *N* scenarios, the global status is copied to
  all *N* links. That's the only faithful interpretation of existing data, but it
  means an "excluded" doc becomes excluded in every scenario it's scored into —
  same as today, just now editable per scenario going forward.

### Phase 2 — Dual-write
- Screening endpoints write the `(scenario_id, document_id)` row in `article_scenarios`
  **and** keep `literature_document.screening_status` updated (so un-migrated readers
  stay correct).
- Drop the `scenario_type` gating in the WHERE — scope by `article_scenarios`
  membership instead (the row being screened).
- Reversible: revert the endpoints; the global column is still authoritative.

### Phase 3 — Dual-read, migrated incrementally
- Introduce one helper that emits the relevant-subset predicate against
  `COALESCE(ars.screening_status, d.screening_status)` (per-scenario, falling back to
  global for any not-yet-backfilled row).
- Migrate the ~118 read sites in grouped batches (canonical predicate → corpus stats →
  PRISMA funnel → briefs/PDFs → search counts → RAG). After each batch, run the
  count-parity script on staging: per-scenario counts must equal today's for scenarios
  where no per-scenario divergence has been introduced yet.

### Phase 4 — Frontend
- Screening UI reads/writes per-scenario status; the PRISMA funnel and corpus badges
  reflect the active scenario. `screen()` in `api.ts` already passes `scenario_id`, so
  the change is mostly display + cache-keying by scenario.

### Phase 5 — Cutover & cleanup
- Flip readers to per-scenario only (drop the `COALESCE` fallback).
- Stop writing the global column; mark it deprecated.
- After a soak period with backups, `DROP COLUMN literature_document.screening_status`
  (+ reason/notes) in a separate release.

## 5. Risks

| Risk | Mitigation |
|------|------------|
| Backfill mis-maps a doc's status | Run on staging first; parity script; global column retained for rollback |
| A reader missed during Phase 3 still reads the global column | `COALESCE(per-scenario, global)` keeps it correct until cutover |
| Count drift between surfaces (the class of bug fixed in #115–#117) | Single shared predicate helper; parity script after every batch |
| Frontend caches stale per-doc status across scenarios | Key screening cache by `(scenarioId, documentId)` |
| Irreversible data loss | No `DROP COLUMN` until Phase 5, after soak + backup |

## 6. Validation harness

A local **Postgres 16 + pgvector** instance (used to validate the #123 ANN rewrite)
can host the staging snapshot. Parity script per scenario:
```sql
-- BEFORE vs AFTER each phase, expect identical rows where no divergence was introduced
SELECT ars.scenario_id,
       COUNT(*) FILTER (WHERE <relevant predicate>) AS relevant,
       COUNT(*) FILTER (WHERE <status='included'>)  AS included,
       COUNT(*) FILTER (WHERE <status='excluded'>)  AS excluded
FROM article_scenarios ars JOIN literature_document d ON d.id = ars.document_id
GROUP BY ars.scenario_id ORDER BY ars.scenario_id;
```

## 7. Rough effort

| Phase | Effort |
|-------|--------|
| 1 — schema + backfill | ~0.5 day (incl. staging validation) |
| 2 — dual-write | ~0.5 day |
| 3 — dual-read (118 sites) | ~2–3 days (the bulk; batched + parity-checked) |
| 4 — frontend | ~1 day |
| 5 — cutover + drop | ~0.5 day + soak |

## 8. Product decisions (answered)

1. **Backfill semantics → option A.** Copy the current global status onto *all* of a
   document's scenario links. Preserves today's behaviour exactly; nothing is lost;
   decisions become editable per scenario going forward. (Phase 1 backfill UPDATE
   already reflects this.)
2. **Auto-screening (living review) → the one scenario being updated.** An automatic
   status set during ingestion applies only to the scenario currently being processed,
   not to every scenario the document belongs to.
3. **`scenario_type` (Migration 1) is resolved first.** This migration's "scope by
   membership" depends on `article_scenarios` being the source of truth, which is
   exactly what Migration 1 establishes. See `scenario-type-migration.md`.

So this migration is **blocked on Migration 1 completing**, and starts at Phase 1
once a staging snapshot is available.
