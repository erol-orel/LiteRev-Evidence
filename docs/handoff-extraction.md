# Handoff: structured extraction (GEOAI4EI use case)

State of production on 2026-10-08, after the full run. Everything below is merged to `main`
(PRs #309, #310, #311, #313, #315) and deployed to https://literev-scenario.com. Nothing is pending on
a branch.

**This note replaces the first version, which said the corpus held 558 relevant papers of which 5 were
extracted and listed the full run as not done. The full run has happened.** Every figure here was read
from the live system and re-checked independently.

A partner-facing write-up of the same material, with the field-by-field comparison against a human
reading of one paper, is in `docs/hpai-report-italian-team.md`. Test steps are in
`docs/test-checklist.md`.

## Goal

Fill the partner team's Excel extraction template (`DATA_Extraction_Template_scenario_xxxx.xlsx`) from
the literature: exposure and transmission, KAP, sex and gender (T2.4), One Health hazard, host and
environment (T4.5, T4.6), health-system capacity indicators.

The template file is **not in the repository**, and there is no spreadsheet of any kind tracked here.
The column titles in `api/extraction.py` were transcribed by hand from the partner file, so the export
cannot be diffed against the real template from this checkout. Getting the file is a prerequisite for
that check.

## The run, as it stands

Scenario `usr-4757684f5462` (HPAI), threshold 0.30, extraction version 1.

| | |
|---|---|
| Corpus | 640 articles |
| Relevant | 602 |
| Extracted | 601 (573 from full text, 28 from abstract) |
| Observations | 10 893 |
| Quote located in the source | 10 291 (94.5%) |
| Given up after 3 attempts | 1 (article 163468, never retried) |
| Extracted but zero observations | 7 |
| Text cut at the 60 000 character limit | 192 (33.5% of the full-text set) |
| Reviewed by a human | 0 |
| Rows carrying a codebook label | 1 896 (17.4%) |

Ran 2026-10-08, roughly 10:49 to 12:11 UTC. Yield is 18.8 rows per full-text paper (median 18, range 0
to 68) against 4.6 per abstract-only paper (median 4.5, range 1 to 11). No abstract produced a table or
figure row.

Rows by sheet: animal 3 336 (445 papers), human_exp 3 172 (441), human_susc 2 449 (360), env 1 619
(364), vector 317 (114).

**Two facts that bound what this corpus can support.** About half the rows are not avian influenza:
COVID-19 alone is 2 091 rows (19.2%) across 136 papers, and roughly 4 300 rows (about 39%) carry an
avian or H5/H7/H9 label. The exact split depends on the label rule, since `disease` is free text with
626 distinct spellings, so quote it as a range. And nothing has been human-reviewed, so every figure is
the model's unchecked reading.

## What exists

| Piece | Backend | Frontend |
|---|---|---|
| Per-article extraction (map), cached in `literature_document.extraction_json` | `api/extraction.py` | `ExtractionSection.tsx` (Evidence tab, Extraction sub-tab) |
| Codebook (label normalisation at read time, never rewrites stored labels) | `api/codebook.py` | `CodebookPanel.tsx` |
| Reviewer decisions (accept, edit, reject, clear; Cohen's kappa) in table `extraction_review` | `api/extraction_review.py` | `ObservationPanel.tsx`, `ReviewSummaryCard.tsx` |
| Pooled estimates (logit, DerSimonian-Laird, Hartung-Knapp CI, prediction interval, min 3 studies) | `api/pooling.py` | `PooledPanel.tsx` (forest plots) |
| Report as Markdown, Word, PDF | `api/extraction_report.py` | download buttons |
| Country and NUTS region of each study (`geo_nuts` table, Eurostat CSV import) | `api/geography.py` | `GeographyPanel.tsx` |
| Source text drawer ("View in the paper") | `/user-scenarios/{sid}/articles/{aid}/text` | `SourceTextDrawer.tsx` |
| Assistant gets the structured block | `api/assistant.py` | n/a |

### What one call returns

Three blocks, and nothing else: a study-level `ref` record of 11 fields (description, article_type,
study_start, study_end, location, notes_geo, risk_pop, positive, percent_positive, math_model,
model_type), 10 strict boolean `coverage` flags (sex_gender, age, occupation, kap_risk_perception, ppe,
vaccination, human_testing, animal_host, environment, vector), and a list of observations.

A stored observation has 15 keys: sheet, transmission_mode, disease, group, covariate, value, descr,
notes, n_cases, pop_risk, original_name, page_section, source_kind, quote, quote_verified. The model
supplies 14; `quote_verified` is computed by the code.

`sheet` is enforced to one of human_susc, human_exp, env, animal, vector, and a row with any other
sheet or with an empty covariate is dropped. `source_kind` is enforced to table, figure or text.
`article_type` is constrained by the prompt but not enforced by the parser. There is no confidence
field anywhere in the stack; provenance is per observation (quote, page or section, source kind,
original label, quote-found flag) and per article (version, source, truncated, n_chars, extracted_at).

### Endpoints

Reads, no key needed:

    GET  /user-scenarios/{sid}/extraction/status
    GET  /user-scenarios/{sid}/extraction/coverage
    GET  /user-scenarios/{sid}/extraction/articles
    GET  /user-scenarios/{sid}/extraction/pooled
    GET  /user-scenarios/{sid}/extraction/geography
    GET  /user-scenarios/{sid}/extraction/report
    GET  /user-scenarios/{sid}/extraction/export
    GET  /user-scenarios/{sid}/extraction/dataset
    GET  /user-scenarios/{sid}/extraction/review/summary
    GET  /user-scenarios/{sid}/articles/{aid}/extraction
    GET  /user-scenarios/{sid}/articles/{aid}/text
    GET  /user-scenarios/{sid}/codebook
    GET  /user-scenarios/{sid}/codebook/export
    GET  /user-scenarios/{sid}/codebook/unmapped
    GET  /geo/nuts/status
    GET  /geo/resolve

Writes, `X-API-Key` required:

    POST   /user-scenarios/{sid}/extraction/run
    POST   /user-scenarios/{sid}/articles/{aid}/extraction/review
    POST   /user-scenarios/{sid}/articles/{aid}/extraction/review/bulk
    PUT    /user-scenarios/{sid}/codebook
    POST   /user-scenarios/{sid}/codebook/import
    POST   /user-scenarios/{sid}/codebook/synonym
    DELETE /user-scenarios/{sid}/codebook
    POST   /geo/nuts/import

## Rules to keep (also in CLAUDE.md)

- No em dash (U+2014) anywhere; `tests/test_no_em_dashes.py` fails CI on the first one.
- Every extraction reads ALL relevant papers (map once, cache on the row; reduce in SQL). Caps default
  to 0. `EXTRACTION_MAX_ARTICLES == 0` is pinned at `tests/test_full_corpus_digest.py:150`.
- `main.py` shares one namespace: a later module overrides a same-named top-level name. Keep names
  unique. `tests/test_namespace.py` guards it (a `to_markdown` clash already broke CI once).
- CI runs Python 3.10: no backslash and no same-quote reuse inside an f-string field.
- i18n: `frontend/src/i18n/locales/{fr,en}.ts` need identical keys and identical `{placeholders}`.
- Tailwind opacity classes like `/3` and `/8` are invalid.
- Every merge to `main` deploys and restarts the API, cutting searches, pipelines and extraction runs
  in flight. Never merge during a presentation or a run.
- Integration tests truncate the tables of `DB_URL`.

## Running it

Server (root@literev-app-01), API on localhost:8000, key in `/etc/literev-api.env` (`WRITE_API_KEY`):

    curl -X POST "http://localhost:8000/user-scenarios/usr-4757684f5462/extraction/run" -H "X-API-Key: <key>"
    curl http://localhost:8000/user-scenarios/usr-4757684f5462/extraction/status

Add `?max_articles=5` for a trial. A full run costs model credits. In the browser, click the header
badge to store the write key; without it the app is read-only, and the five write actions each show an
explanatory line rather than a dead button.

Note: the run endpoint has no `force` parameter and skips any row already at the current extraction
version, so a re-run will not revisit the 601 papers unless the version changes.

## Not done yet

**Blocked on a file from the partner team.** Each unblocks code that is already shipped.

- **The Annex 2 label hierarchy.** The highest-value item. 8 997 of 10 893 rows carry a label the
  default codebook cannot map, which is why only 17.4% are normalised and why most rows pool only with
  an identical spelling. Loading it relabels every stored row retroactively with **no new model call**,
  because normalisation happens at read time. Destination: `POST /user-scenarios/{sid}/codebook/import`.
  Seven of the 19 default groups (pregnancy, ethnicity, socioeconomic, climate, persistence, vector
  species, vector density) have no level-2 node at all, and a match requires one, so rows in those
  groups can never map until the hierarchy fills them.
- **The Eurostat GISCO NUTS file.** `POST /geo/nuts/import`. Never called in production:
  `/geo/nuts/status` reports source `builtin`, 42 regions. Those 42 are Germany's sixteen Länder at
  NUTS 1 and Italy's five macro-areas at NUTS 1 plus twenty-one regions at NUTS 2. No NUTS 3, no city,
  no other country. Of 601 extracted papers, 270 resolve to a country across 36 countries and exactly
  **2 carry any NUTS region**.
- **The extraction template itself**, as above.

**Needs development.**

- **Typed transmission relations for T4.6.** Transmission is an unnormalised free-text column, and the
  concept graph holds untyped co-occurrence triples with weights, not asserted predicates. On the
  comparison paper this is the one slot where LiteRev holds every fact but cannot state the chain.
- **Health system capacity is not covered at all.** No sheet, no codebook node, no `ref` column. The
  vaccination and human_testing flags mark that a paper touches the subject without carrying a value.
  Of the six template areas this is the only one genuinely absent rather than partial.
- **Accuracy has never been measured for this extraction.** `scripts/gold_standard.py` is a real
  harness (stratified sample including screen-rejected articles, blind annotation workbooks,
  inter-annotator kappa, per-field precision, recall and F1 at a 10% value tolerance, and it refuses to
  score if the screen regex changed since sampling), but it predates this work and targets the
  epidemiological parameter extraction in `api/variables.py`. Nothing in `api/` computes precision or
  recall for the structured extraction.
- **Rayyan import, LLM screening suggestion from eligibility criteria, multi-host SEIR:** genuinely
  absent in code.
- **A PubMed POST path for long queries:** absent in a narrow sense, every eutils call is a GET.

**Defects.**

- **No model identifier and no prompt fingerprint on any of the 601 extractions.** The code stamps
  both, but the run finished about twenty-five minutes before the commit that writes them, so
  `coverage.by_model` reads `[{model: unknown, prompt_sha: unknown, n: 601}]` and the report's method
  line reads "Made with: unknown, prompt unknown (601)". Because the extraction version did not change
  and the run endpoint has no force flag, this cannot be backfilled without a code change and a paid
  re-run.
- **192 papers had their text cut** at 60 000 characters. The fact is in the interface and in
  `/extraction/articles`, and in none of the deliverables: not the report, not the workbook, not the
  CSV, not the JSONL.
- **The labelled dataset download returns 0 bytes** because nothing is reviewed and the interface does
  not pass the flag for unreviewed rows.
- **`ref.first_author` is filled on 115 of 601 rows (19.1%)**, so some report citations read "? 2013".
- **The pooled panel lists at most 120 groups** (`api/pooling.py:308`) with no truncation flag in the
  payload. Of those 120 today, 101 hold one study and 15 hold two, so only 4 reach the minimum of three.
- **The report's country table stops at 25** of the 36 countries.
- **The Excel export is not byte-reproducible.** Successive downloads differ by a byte or two and have
  different checksums, since openpyxl writes a zip with varying metadata. Never quote an exact size;
  say about 1.6 MB. The CSV is byte-stable.
- **Two resolver behaviours in geography.** The longest region *name* wins rather than the deepest
  level, so "Umbria, Central Italy" resolves to NUTS 1 while "Umbria" alone resolves to NUTS 2. And a
  paper naming several regions silently keeps one.

## The second scenario

`usr-54fc5e52fea5` (9 389 articles, 6 523 relevant) is the HPAI pathogen clause on its own, without
the exposure, transmission and KAP facet: its query appears verbatim inside the HPAI query. Because
`extraction_json` is cached on the shared `literature_document` row, it **already inherits 312
extracted papers and 5 900 observations at no extra cost**, and its extraction, coverage, pooled and
geography panels are already populated. Its weakness is full text: 16.8% against 95.3% for HPAI, so a
full run there would read four papers in five from abstracts alone.

`usr-5ee70446a248` (21 804 articles) is not demonstrable: a threshold of 0.60 keeps 32 papers, its
brief grades the evidence highly on those 32 with zero resolved references, and lowering the threshold
to get a credible corpus (0.50 keeps 1 002) would put PICO, metadata, full text and extraction back in
the queue and invalidate the brief.
