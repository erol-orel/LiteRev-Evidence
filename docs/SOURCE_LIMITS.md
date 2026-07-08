# Live Data Sources — Real Constraints & Limits

**Scope:** the sources actually wired into the running app (code-grounded, not aspirational).
For the original planning spec see `DATA_SOURCES.md` (many sources there are *not* wired).
Line references point at `main.py` / `data_connectors.py` so this stays verifiable.

---

## 0. How a corpus is built (shared mechanics)

- **12 literature sources are queried in parallel** on every search/populate
  (`ThreadPoolExecutor(max_workers=12)`, `main.py:8215`).
- **Fetch ceiling: `LIVE_MAX_PER_SOURCE = 2000` documents per source per search**
  (default; **env-configurable** — set it to e.g. `100000` to effectively remove the cap,
  after which the time budget below is the real governor). This is the *fetch* cap, before
  boolean filtering. ⚠ raising it multiplies API calls (429 risk) and OpenAI embedding cost
  on every search.
- **Wall-clock budget: `POPULATE_FEDERATION_BUDGET = 180 s`** (`main.py:242`, default;
  set the env var higher — e.g. `600` — for a more complete corpus). Populate runs in a
  **background thread**, so a larger budget doesn't block the request; it only delays the
  scenario reaching "done". Not unbounded on purpose: a truly stuck source would otherwise
  keep the job alive forever. When it expires, slow sources stop paginating mid-stream and
  the corpus is built from whatever arrived — so a slow source (arXiv, deep Crossref/
  OpenAlex) may not reach its 2000 cap.
- **Corpus membership = boolean/lexical only.** After fetching, membership is
  recomputed with the *same* boolean query on `local DB ∪ freshly-ingested`
  (`_boolean_corpus_ids` / `_multi_query_corpus_ids`, `main.py:8276`). The 2000
  cap and the API quirks below only affect *what gets discovered*, never how a
  discovered paper is judged.
- **Quality filters that drop documents:** title+abstract < 30 chars is discarded
  at ingest; **any article with no abstract is removed from the corpus** after
  populate (`main.py:8293`). Sources that return abstract-less records (Crossref,
  ClinicalTrials) therefore contribute fewer papers than they return.
- **Dedup (on ingest):** ① same `(source, external_id)` → reuse; ② `ON CONFLICT (doi)`
  → cross-source DOI merge; ③ **normalized-title** match (`title_norm` = lowercased,
  punctuation-stripped) for records **without** a DOI or with a DOI in only one source.
  Titles under 20 normalized chars are skipped to avoid merging generic "Editorial"-type
  titles. (The `is_duplicate` flag was previously never set — title dedup now prevents the
  duplicates at ingest instead.)

### Multi-query combination (union / intersection)
Each facet (main query + each sub-query) is translated to boolean if natural,
then matched lexically → a **set of document IDs**. `Union (OR)` = ∪ of the sets,
`Intersection (AND)` = ∩, then dedup (`main.py:1694`). Semantic and Cohere scores
**never** enter this — they only rank/select the relevant subset later.

### Which query each source receives
The natural query is translated once (LLM, cached) into a boolean; a plain keyword
string is derived from it. Each source gets the variant it understands:

| Query variant | Sources |
|---------------|---------|
| **PubMed boolean** (with `[MeSH]` / `[Title/Abstract]` tags) | PubMed |
| **General boolean** (`AND`/`OR`/`NOT`, quotes, groups) | Europe PMC, Preprints (EPMC `SRC:PPR`), **local corpus re-match** |
| **Plain keywords** (operators/tags stripped, ≤ 8 words) | OpenAlex, Crossref, Semantic Scholar, DOAJ, ClinicalTrials.gov, CORE, arXiv, OpenAIRE |
| **No query** (date-window scan + local term filter) | bioRxiv, medRxiv (native) |

Only PubMed + Europe PMC apply the *structured* boolean; the keyword APIs get a flattened
bag of words and rely on their own relevance ranking. That's why per-source counts differ
for the same search.

### When the scores are computed (never as a separate "search" step)
- **Corpus membership / count** = the boolean lexical match above. No score involved.
- **Semantic score** (`similarity_score`, OpenAI cosine) — at **populate = scenario
  creation** (`_run_semantic_rerank_inline`); a *soft* rank that never removes a paper.
- **Cohere rerank** (`rerank_score`) — **lazily, on the scenario page** when you open it
  (`_maybe_autorerank`), feeding the relevance **threshold** that picks the subset.

So pertinence + Cohere run at/after scenario creation and drive only the *relevant subset* —
the corpus count itself stays pure lexical.

---

## 1. Literature sources — quick table

| # | Source | Page size | Practical max / search | Date coverage | Key needed | Search type |
|---|--------|-----------|------------------------|---------------|-----------|-------------|
| 1 | **PubMed** | 200 | 2000 (newest first) | full history | optional (`NCBI_API_KEY`) | boolean |
| 2 | **OpenAlex** | 200 | 2000 (newest first) | full history | none (polite email) | keyword |
| 3 | **Crossref** | 100 | 2000 (by relevance) | full history | none (polite email) | keyword |
| 4 | **Europe PMC** | 200 | 2000 (newest first) | full history | none | boolean |
| 5 | **Preprints (Europe PMC)** | 100 | 2000 | full history | none | boolean |
| 6 | **Semantic Scholar** | 100 | **1000 (API hard cap)** | full history | optional (`SEMANTIC_SCHOLAR_API_KEY`) | keyword |
| 7 | **DOAJ** | 100 | 2000 | full history (OA only) | none | keyword |
| 8 | **ClinicalTrials.gov** | 100 | 2000 (trials, not papers) | full registry | none | keyword |
| 9 | **CORE** | 100 | 2000 | full history | **required (`CORE_API_KEY`, free)** | keyword |
| 10 | **arXiv** | 100 | **~100 (3 s/req vs 55 s budget)** | full history | none | keyword |
| 11 | **OpenAIRE** | 50 | 2000 | full history | none | keyword |
| 12 | **bioRxiv + medRxiv** | 100 | **last 45 days only** | **recent window, no keyword search** | none | date-window + local filter |

---

## 2. Per-source detail

### 1. PubMed — `eutils.ncbi.nlm.nih.gov/entrez/eutils` (`main.py:7618`)
- esearch (`usehistory=y`) → efetch in batches of **`BATCH_SIZE = 200`** (`main.py:7500`).
- `sort=pub_date` → **newest first**, so the 2000 cap keeps the most recent.
- Rate limit: **3 req/s keyless**, ~10 req/s with `NCBI_API_KEY` (optional). A
  shared lock throttles all eutils calls (`_NCBI_LOCK`) to avoid 429s.
- Uses the translated **PubMed boolean** query. Extracts authors, journal, DOI.

### 2. OpenAlex — `api.openalex.org/works` (`main.py:7716`)
- `per_page` up to **200**, `sort=publication_date:desc` (newest first), cap 2000.
- Keyless; uses the "polite pool" via `mailto`. No key = shared rate pool.
- Abstract is reconstructed from OpenAlex's *inverted index* (occasionally lossy).
- Uses the plain (non-boolean) query string.

### 3. Crossref — `api.crossref.org/works` (`main.py:7780`)
- `rows` up to **100**, offset pagination, cap 2000.
- **Sorted by relevance, NOT date** — deliberately: Crossref publication dates are
  unreliable (records with dates like "2121" would poison a date sort) (`main.py:7792`).
- **Many Crossref records have no abstract** → those are dropped by the no-abstract
  rule, so Crossref's *net* contribution is smaller than its raw count.

### 4. Europe PMC — `ebi.ac.uk/europepmc/webservices/rest/search` (`main.py:7844`)
- `pageSize = 200`, cursorMark pagination, `resultType=core`, `sort=P_PDATE_D desc`.
- Keyless, cap 2000. Uses the **boolean** query. Covers PubMed + PMC + Agricola + preprints.

### 5. Preprints (Europe PMC, `SRC:PPR`) (`main.py:7911`)
- Same EBI endpoint, query = `(<boolean>) AND (SRC:PPR)`, `pageSize = 100`, cap 2000.
- Real keyword search over preprints **as indexed by Europe PMC** (Research Square,
  bioRxiv, medRxiv, Preprints.org…). Labelled `source="preprint"`, `source_type="preprint"`.
- This is distinct from source #12 (the native bioRxiv/medRxiv scan).

### 6. Semantic Scholar — `api.semanticscholar.org/graph/v1/paper/search` (`main.py:7999`)
- `limit = 100`/page, offset pagination.
- **Hard API limit: `offset + limit ≤ 1000`** → **max 1000 papers per query**,
  regardless of the global 2000 cap (`main.py:8004`).
- Keyless works but is aggressively rate-limited (HTTP 429 → 2 s backoff, retry);
  `SEMANTIC_SCHOLAR_API_KEY` optional for higher throughput.

### 7. DOAJ — `doaj.org/api/search/articles` (`main.py:8030`)
- `pageSize = 100`, page pagination, cap 2000, keyless.
- **Open-access journals only** by definition — narrower but high-quality/full-text-able.

### 8. ClinicalTrials.gov — `clinicaltrials.gov/api/v2/studies` (`main.py:8057`)
- `pageSize = 100`, `pageToken` pagination, cap 2000, keyless.
- Returns **trials, not articles** (title + brief summary as the "abstract"). Useful
  for evidence but different unit of analysis; no journal/DOI.

### 9. CORE — `api.core.ac.uk/v3/search/works` (`main.py:8083`)
- `limit = 100`, offset pagination, cap 2000.
- **Requires `CORE_API_KEY` (free).** Without it the source is **silently skipped**
  (deployment stays green) — so CORE contributes 0 unless the key is set.
- 429 → 3 s backoff.

### 10. arXiv — `export.arxiv.org/api/query` (`main.py:8121`)
- `max_results = 100`/page, `start` pagination, Atom XML.
- **arXiv requires ≥ 3 s between requests** (`main.py:8142`). Against the 55 s
  federation budget, that means **usually only the first ~100 results** land before
  the budget cuts pagination. Keyless.
- Coverage skew: physics / CS / math / **quantitative biology** preprints.

### 11. OpenAIRE — `api.openaire.eu/search/publications` (`main.py:8147`)
- `size = 50`/page (smallest page of the set), page pagination, cap 2000, keyless,
  0.5 s between requests.

### 12. bioRxiv + medRxiv (native) — `api.biorxiv.org/details/...` (`main.py:8173`)
- **The bioRxiv API has NO keyword search** — it only serves date windows.
- The fetcher scans the **last 45 days** (from cursor 0 forward), ≤ 30 pages ×100/page
  **per server**, then filters client-side: a preprint is kept only if **≥ one-third of
  the query's ≥4-char terms** appear in title+abstract (`_parse_biorxiv`, `main.py:7288`).
- **Fundamental limitation:** this only surfaces *recent* preprints matching your terms.
  For older preprints, rely on source #5 (Europe PMC `SRC:PPR`), which *does* index them.
- Ingested under distinct `source="biorxiv"` / `source="medrxiv"` labels.

---

## 3. Environmental / variable connectors (`data_connectors.py`)

These feed the **auto-fetch** panel (predictors for a scenario's model), not the corpus.

| Connector | Endpoint | Granularity | Coverage | Key / license | Notes |
|-----------|----------|-------------|----------|---------------|-------|
| **Open-Meteo — Weather** | `api.open-meteo.com` | point (lat/lon) | ERA5, ~1940→present | keyless; **free tier non-commercial** | temp/humidity/precip/wind |
| **Open-Meteo — Air Quality** | `air-quality-api.open-meteo.com` | point (~11 km CAMS) | ~2013→present | keyless; **free tier non-commercial** | PM2.5/PM10/NO₂/O₃ |
| **EAWAG — wastewater** | GitHub CSV | catchment (Geneva, Lausanne, Neuchâtel, Porrentruy) | **frozen archive since 2024-03** | CC BY 4.0 | flu-A/flu-B/RSV/SARS loads; historical backfill only |
| **FOPH — influenza wastewater** | opendata.swiss `influenza1` | national (plants aggregated) | **live, weekly since 2022** (70k+ rows) | opendata.swiss ToU | value/valuemean7d/conc/flow/pop; pass `geoRegion=GE` to narrow |

**Caveats:**
- **Open-Meteo free tier is non-commercial** — a production deployment needs a paid
  or self-hosted Open-Meteo plan.
- **EAWAG is a frozen 2024-03 archive** → great for training backfill, not for live
  monitoring. Use the FOPH connector for current wastewater signal.
- **FOPH is influenza wastewater, not clinical Sentinella ILI/ARI.** The raw feed
  multiplexes ~100 treatment plants; rows are aggregated per date into a national
  series (mean load/conc, summed pop/flow). A per-canton filter needs the region
  column values (next refinement). Clinical ILI/ARI has **no live connector** yet →
  manual CSV upload.

---

## 4. The limitations that actually bite

1. **The 55 s budget, not the 2000 cap, is usually the binding constraint** for slow
   sources (arXiv especially, sometimes deep Crossref/OpenAlex pages).
2. **Semantic Scholar tops out at 1000/query** (hard API limit).
3. **bioRxiv/medRxiv native = recent 45-day window only** (no historical keyword search).
4. **CORE = 0 without its (free) API key.**
5. **No-abstract records are dropped** → Crossref and ClinicalTrials net lower than raw.
6. **arXiv/DOAJ/ClinicalTrials are domain-skewed** (preprints / OA journals / trials) —
   good for recall breadth, not representative on their own.
7. **All caps are per-source, pre-boolean.** Final corpus size = boolean match over the
   union of everything ingested + your local DB, so it is typically **smaller** than the
   sum of raw fetches (dedup + boolean filter + no-abstract rule).
