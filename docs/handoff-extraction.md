# Handoff: structured extraction (GEOAI4EI use case)

State as of 2026-10-08. Everything below is merged to `main` (PR #309, #310, #311, #313) and deployed
to https://literev-scenario.com. Nothing is pending on a branch.

## Goal

Fill the team's Excel extraction template (DATA_Extraction_Template_scenario_xxxx.xlsx) from the
literature: exposure/transmission, KAP, sex/gender (T2.4), One Health hazard/host/environment
(T4.5, T4.6), health-system capacity indicators. Test scenario: `usr-4757684f5462` (558 relevant
papers, only 5 extracted so far).

## What exists (api/ module, UI component)

| Piece | Backend | Frontend |
|---|---|---|
| Per-article extraction (map), cached in `literature_document.extraction_json` | `api/extraction.py` | `ExtractionSection.tsx` (Evidence tab, Extraction sub-tab) |
| Codebook (label normalisation at read time, never rewrites stored labels) | `api/codebook.py` | `CodebookPanel.tsx` |
| Reviewer decisions (accept/edit/reject, Cohen's kappa) in table `extraction_review` | `api/extraction_review.py` | `ObservationPanel.tsx`, `ReviewSummaryCard.tsx` |
| Pooled estimates (logit, DerSimonian-Laird, Hartung-Knapp CI, prediction interval, min 3 studies) | `api/pooling.py` | `PooledPanel.tsx` (forest plots) |
| Report as Markdown, Word, PDF | `api/extraction_report.py` | download buttons |
| Country and NUTS region of each study (`geo_nuts` table, Eurostat CSV import) | `api/geography.py` | `GeographyPanel.tsx` |
| Source text drawer ("View in the paper") | `/articles/{aid}/text` | `SourceTextDrawer.tsx` |
| Assistant gets the structured block | `api/assistant.py` | n/a |

Endpoints are under `/user-scenarios/{id}/extraction/...` (run, status, coverage, articles, dataset,
export, pooled, report, geography, codebook, review). Write endpoints need `X-API-Key`.

## Rules to keep (also in CLAUDE.md)

- No em dash (U+2014) anywhere; `tests/test_no_em_dashes.py` fails CI.
- Every extraction reads ALL relevant papers (map once, cache on the row; reduce in SQL). Caps default
  to 0. `EXTRACTION_MAX_ARTICLES == 0` is pinned by a test.
- `main.py` shares one namespace: a later module overrides same-named top-level names. Keep names
  unique. `tests/test_namespace.py` guards it (a `to_markdown` clash already broke CI once).
- CI runs Python 3.10: no backslash or same-quote reuse inside f-string fields.
- i18n: `frontend/src/i18n/locales/{fr,en}.ts` need identical keys and `{placeholders}`.
- Tailwind opacity classes like `/3` and `/8` are invalid.
- Every merge to `main` deploys and restarts the API, cutting searches, pipelines and extraction runs
  in flight. Never merge during a presentation or a run.
- Integration tests truncate the tables of `DB_URL`.

## Running it

Server (root@literev-app-01), API on localhost:8000, key in `/etc/literev-api.env` (WRITE_API_KEY):

    curl -X POST "http://localhost:8000/user-scenarios/usr-4757684f5462/extraction/run" -H "X-API-Key: <key>"
    curl http://localhost:8000/user-scenarios/usr-4757684f5462/extraction/status

Add `?max_articles=5` for a trial. A full run costs OpenAI credits. Pooled estimates stay mostly
empty until it is done. In the browser, click the header badge to store the admin key, otherwise the
app is read-only (this caused the "Acces non autorise" bug in the parameters panel, now hidden).

## Not done yet

- Full 558-paper extraction run (user decides when).
- User inputs needed: Annex 2 label hierarchy, gold-standard rows, Eurostat NUTS CSV upload
  (`POST /geo/nuts/import`).
- Rayyan import, LLM screening suggestion from eligibility criteria, typed relation graph for T4.6,
  multi-host SEIR, PubMed POST for long queries.
- Search query advice given in chat: structure `A AND (B OR C)` then limits, per-database syntax.
