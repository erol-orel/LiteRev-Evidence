# Live Data Sources - Real Constraints & Limits

**Scope:** the sources actually wired into the running app (code-grounded, not aspirational).
For the original planning spec see `DATA_SOURCES.md` (many sources there are *not* wired).
Line references point at `main.py` / `data_connectors.py` so this stays verifiable.

---

## 0. How a corpus is built (shared mechanics)

- **12 literature sources are queried in parallel** on every search/populate
  (`ThreadPoolExecutor(max_workers=12)`, `main.py:8215`).
- **Every search is exhaustive** (since 13 October 2026; the "standard" search, which kept
  each source's most relevant records, was removed): every database that applies the boolean
  returns everything it matches, up to a guard rail, the larger of `EXHAUSTIVE_MAX_PER_SOURCE`
  (20 000 by default) and `LIVE_MAX_PER_SOURCE` (2000 by default; production sets 100 000).
  Crossref and OpenAlex's full-text pass, which rank by relevance and never run dry, keep
  `RANKED_PASS_MAX` (2000). This is the *fetch* cap, before boolean filtering. ⚠ raising it
  multiplies API calls (429 risk) and OpenAI embedding cost on every search. The search
  screen reads these numbers from `GET /search/limits`.
- **Wall-clock budget: the larger of `EXHAUSTIVE_FEDERATION_BUDGET` (2700 s, 45 minutes)
  and `POPULATE_FEDERATION_BUDGET` (600 s)** (`api/core.py`, defaults). Populate runs in a
  **background thread**, so a larger budget doesn't block the request; it only delays the
  scenario reaching "done". Not unbounded on purpose: a truly stuck source would otherwise
  keep the job alive forever. When it expires, slow sources stop paginating mid-stream and
  the corpus is built from whatever arrived - so a slow source (arXiv, deep Crossref/
  OpenAlex) may not reach its cap.
- **Corpus membership = boolean/lexical only.** After fetching, membership is
  recomputed with the *same* boolean query on `local DB ∪ freshly-ingested`
  (`_boolean_corpus_ids` / `_multi_query_corpus_ids`, `main.py:8276`). The 2000
  cap and the API quirks below only affect *what gets discovered*, never how a
  discovered paper is judged.
- **Quality filter that drops documents:** title+abstract < 30 chars is discarded
  at ingest. **Records without an abstract are KEPT in the corpus**, flagged
  (`no_abstract` in the counts, `has_abstract` per article, `?no_abstract=true` on
  `/corpus`): they match the boolean on their title, are never relevant on their
  similarity score (it reads only the title), and enter the relevant subset only when a
  reviewer includes them by hand. PRISMA reports them as "kept without an abstract,
  screened by title by hand". The background backfill asks Europe PMC for their abstract
  (by DOI, else by PMID); an abstract filled later makes the record eligible by itself.
- **Retries:** a page that fails on a timeout, a connection error, a 429 or a 5xx is
  retried (3 attempts, 2 s then 4 s, `Retry-After` honoured up to 30 s, never past the
  federation budget) before the source ends in `error`; a PubMed efetch batch likewise.
- **Dedup (on ingest):** ① same `(source, external_id)` → reuse; ② `ON CONFLICT (doi)`
  → cross-source DOI merge; ③ **normalized-title** match (`title_norm` = lowercased,
  punctuation-stripped) for records **without** a DOI or with a DOI in only one source.
  Titles under 20 normalized chars are skipped to avoid merging generic "Editorial"-type
  titles. (The `is_duplicate` flag was previously never set - title dedup now prevents the
  duplicates at ingest instead.)

### Multi-query combination (union / intersection)
Each facet (main query + each sub-query) is translated to boolean if natural,
then matched lexically → a **set of document IDs**. `Union (OR)` = ∪ of the sets,
`Intersection (AND)` = ∩, then dedup (`main.py:1694`). Semantic and Cohere scores
**never** enter this - they only rank/select the relevant subset later.

### Which query each source receives
The natural query is translated once (LLM, cached) into a boolean; a plain keyword
string is derived from it. Each source gets the variant it understands:

| Query variant | Sources | Source-union? |
|---------------|---------|:---:|
| **PubMed boolean** (`[MeSH]`/`[Title/Abstract]` tags) | PubMed | ✅ |
| **General boolean** (`AND`/`OR`/`NOT`, quotes, groups) | Europe PMC, Preprints (EPMC `SRC:PPR`) | ✅ |
| **Portable boolean** (field tags stripped) | OpenAlex, DOAJ, CORE, ClinicalTrials.gov, OpenAIRE (Graph v2) | ✅ |
| **Dialect boolean** | arXiv (`all:"…"`, `ANDNOT`); Semantic Scholar (`/paper/search/bulk`: space=AND, `\|`=OR) | ✅ |
| **Plain keywords / relevance** (no boolean upstream) | Crossref¹ | - (local re-filter) |

¹ Crossref is the only literature source with no boolean support (relevance-ranked free text).
Semantic Scholar now uses its **bulk** endpoint for boolean (falls back to the relevance endpoint if
the query has a `NOT`). OpenAIRE was **migrated to the Graph API v2** (`/graph/v2/researchProducts`,
cursor-paged) after its legacy `search/publications` endpoint was retired 2026-05-31.
² bioRxiv/medRxiv have no keyword API (date windows only). The native fetcher that scanned their
last 45 days and filtered on the client found 0 records on HPAI_last and was **removed from the
federation**; their preprints are boolean-searched through the Europe PMC preprint facet (#5), which
returned 457 on the same search. The bioRxiv API is still used by the full-text step, by DOI.

**Source-union** (`main.py:_link_to_scenario(boolean_native=True)`): the ✅ sources applied the *real*
boolean, so their hits enter the corpus **directly** (not re-filtered locally). The URL-limited sources
(OpenAlex, ClinicalTrials.gov, arXiv, OpenAIRE) receive the whole boolean when its URL-encoded form fits
`URL_QUERY_MAX_ENCODED` (1 800 bytes); otherwise the whole boolean split into parts whose union is
the strategy (as DOAJ), or, past `SPLIT_MAX_PARTS` parts, a reduced form re-filtered locally. CORE
receives it in a POST body and PubMed always gets the union "(MeSH) OR (portable boolean)" (POST past
the URL limit), whatever their length. That's why per-source counts differ for the same search.

### When the scores are computed (never as a separate "search" step)
- **Corpus membership / count** = the boolean lexical match above. No score involved.
- **Semantic score** (`similarity_score`, OpenAI cosine) - at **populate = scenario
  creation** (`_run_semantic_rerank_inline`); a *soft* rank that never removes a paper.
- **Cohere rerank** (`rerank_score`) - **lazily, on the scenario page** when you open it
  (`_maybe_autorerank`), feeding the relevance **threshold** that picks the subset.

So pertinence + Cohere run at/after scenario creation and drive only the *relevant subset* -
the corpus count itself stays pure lexical.

---

## 1. Literature sources - quick table

| # | Source | Page size | A search keeps | Key needed |
|---|--------|-----------|----------------|-----------|
| 1 | **PubMed** | 200 | every match (past 10 000, read in year slices) | optional (`NCBI_API_KEY`) |
| 2 | **OpenAlex** | 200 | every title/abstract match + its 2000 most relevant full-text matches | recommended (`OPENALEX_API_KEY`, free) |
| 3 | **Crossref** | 1000 | **2000, by relevance** (keywords only) | none (polite email) |
| 4 | **Europe PMC** | 1000 | every match | none |
| 5 | **Preprints (Europe PMC)** | 100 | every match | none |
| 6 | **Semantic Scholar** | bulk | every match (classic endpoint, 1000 at most, when the strategy has a NOT) | optional (`SEMANTIC_SCHOLAR_API_KEY`) |
| 7 | **DOAJ** | 100 | every match (strategy split in parts) | none |
| 8 | **ClinicalTrials.gov** | 100 | every match (trials, not papers) | none |
| 9 | **CORE** | 100 | every match | **required (`CORE_API_KEY`, free)** |
| 10 | **arXiv** | 100 | every match (3 s/request) | none |
| 11 | **OpenAIRE** | 50 | every match (strategy split in 4-operator parts when it fits) | none |

> "Every match" stops at the guard rail above (the larger of `EXHAUSTIVE_MAX_PER_SOURCE` and
> `LIVE_MAX_PER_SOURCE`). The federation budget governs how long sources may page. The PRISMA
> card names every source that kept fewer records than it announced ("capped"). Cards of
> searches run before 13 October 2026 may say "standard search": at most `LIVE_MAX_PER_SOURCE`
> records per source, in each source's relevance order.
> bioRxiv and medRxiv are no longer a source of their own: see footnote ² above.

---

## 2. Per-source detail

### 1. PubMed - `eutils.ncbi.nlm.nih.gov/entrez/eutils` (`main.py:7618`)
- esearch (`usehistory=y`) → efetch in batches of **`BATCH_SIZE = 200`** (`main.py:7500`).
- `sort=relevance` (PubMed's Best Match): at the cap, the most relevant are kept.
- eutils returns at most 10 000 identifiers per query. When a query announces more and
  the per-source cap asks for more, the same query is read in publication-year slices
  (`AND (YYYY:YYYY[dp])`, each range halved until it announces 10 000 or fewer;
  `api.sources.pubmed_esearch_by_years`, shared with the living review). Every record is
  counted once, the PRISMA total stays the whole query's count, and the card names the
  slices. A single year still over 10 000 keeps its 10 000 most relevant, and says so;
  above the per-source cap, the cap is shared between the slices in proportion. The
  scenario's years (`year_min`, `year_max`) are applied in the PubMed query itself.
- Rate limit: **3 req/s keyless**, ~10 req/s with `NCBI_API_KEY` (optional). A
  shared lock throttles all eutils calls (`_NCBI_LOCK`) to avoid 429s.
- Uses the translated **PubMed boolean** query. Extracts authors, journal, DOI.

### 2. OpenAlex - `api.openalex.org/works` (`main.py:7716`)
- `per-page` up to **200**: every title/abstract match by cursor, plus its 2000 best full-text
  matches (`sort=relevance_score:desc`).
- `mailto` on every call, and `api_key` when `OPENALEX_API_KEY` is set (`openalex_params`).
  OpenAlex meters its API: without a key a call draws on a small free daily budget shared
  by every client behind the server's IP address, and once it is spent every call answers
  429 "Insufficient budget" until midnight UTC (the exposure search of 12 October 2026
  stopped there after 1 594 records). A search page costs budget, a work read by its id
  does not. A free key gives the server a budget of its own. The key travels in the URL,
  so error texts pass through `redact_keys` before they are logged or stored.
- Abstract is reconstructed from OpenAlex's *inverted index* (occasionally lossy).
- Uses the plain (non-boolean) query string.

### 3. Crossref - `api.crossref.org/works` (`main.py:7780`)
- `rows` up to **100**, offset pagination, cap 2000.
- **Sorted by relevance, NOT date** - deliberately: Crossref publication dates are
  unreliable (records with dates like "2121" would poison a date sort) (`main.py:7792`).
- **Many Crossref records have no abstract** → they are kept, flagged, and screened by
  title by hand (see the quality filter above) until the backfill finds their abstract.
- The year is the publication date (`issued`, then `published-print`, `published-online`,
  `published`), and the DOI deposit date (`created`) only last.

### 4. Europe PMC - `ebi.ac.uk/europepmc/webservices/rest/search` (`main.py:7844`)
- `pageSize = 1000`, cursorMark pagination, `resultType=core`, default order (relevance).
- Keyless, cap 2000. Uses the **boolean** query. Covers PubMed + PMC + Agricola + preprints.

### 5. Preprints (Europe PMC, `SRC:PPR`) (`main.py:7911`)
- Same EBI endpoint, query = `(<boolean>) AND (SRC:PPR)`, `pageSize = 100`, cap 2000, default
  order (relevance; it sorted by date until the cap was found to keep the most recent).
- Real keyword search over preprints **as indexed by Europe PMC** (Research Square,
  bioRxiv, medRxiv, Preprints.org…). Labelled `source="preprint"`, `source_type="preprint"`.
- The only way bioRxiv and medRxiv are searched: their own API has no keyword search.

### 6. Semantic Scholar - `api.semanticscholar.org/graph/v1/paper/search` (`main.py:7999`)
- `limit = 100`/page, offset pagination.
- **Hard API limit: `offset + limit ≤ 1000`** → **max 1000 papers per query**,
  regardless of the global 2000 cap (`main.py:8004`).
- Keyless works but is aggressively rate-limited (HTTP 429 → 2 s backoff, retry);
  `SEMANTIC_SCHOLAR_API_KEY` optional for higher throughput.

### 7. DOAJ - `doaj.org/api/search/articles` (`main.py:8030`)
- `pageSize = 100`, page pagination, cap 2000, keyless.
- **Open-access journals only** by definition - narrower but high-quality/full-text-able.

### 8. ClinicalTrials.gov - `clinicaltrials.gov/api/v2/studies` (`main.py:8057`)
- `pageSize = 100`, `pageToken` pagination, cap 2000, keyless.
- Returns **trials, not articles** (title + brief summary as the "abstract"). Useful
  for evidence but different unit of analysis; no journal/DOI.

### 9. CORE - `api.core.ac.uk/v3/search/works` (`main.py:8083`)
- `limit = 100`, offset pagination, cap 2000.
- **Requires `CORE_API_KEY` (free).** Without it the source is **silently skipped**
  (deployment stays green) - so CORE contributes 0 unless the key is set.
- 429 → 3 s backoff.

### 10. arXiv - `export.arxiv.org/api/query` (`main.py:8121`)
- `max_results = 100`/page, `start` pagination, Atom XML.
- **arXiv requires ≥ 3 s between requests** (`main.py:8142`), retries included. The 45-minute
  federation budget leaves room for about 900 pages. Keyless.
- Coverage skew: physics / CS / math / **quantitative biology** preprints.

### 11. OpenAIRE - `api.openaire.eu/search/publications` (`main.py:8147`)
- `size = 50`/page (smallest page of the set), page pagination, cap 2000, keyless,
  0.5 s between requests.

### Retired: bioRxiv + medRxiv (native)
- The bioRxiv API has no keyword search, only date windows: the fetcher scanned the last
  45 days and filtered on the client, and returned 0 records on HPAI_last. Removed from
  the federation; source #5 searches both servers. Older library rows keep their
  `source="biorxiv"` / `source="medrxiv"` labels.

### Excluding preprints (search option)
- The search form's "Exclude preprints" box stores `exclude_preprints: true` in the
  scenario's filters (`POST /user-scenarios/{id}/populate?exclude_preprints=true` does the
  same from the API).
- The preprint servers (#5 and arXiv) are not searched; PubMed receives
  `(<query>) NOT preprint[pt]` and Europe PMC `(<query>) NOT SRC:PPR`; the local library
  matches no preprint; and the preprints the other databases return (OpenAlex, Semantic
  Scholar, Crossref, CORE…) leave the corpus before screening.
- What a preprint is lives in one SQL predicate (`api/search.py:preprint_sql`): a
  preprint-server DOI (bioRxiv/medRxiv dated DOIs, Research Square, SSRN, Preprints.org,
  arXiv, OSF…), or a preprint source, URL or journal name, unless the row carries a
  journal DOI or a journal name (a preprint merged with its published version by the
  title dedup is a published article and stays).
- The PRISMA card shows the removed preprints on their own line and names the servers
  not searched and the databases that excluded preprints in the query.

---

## 3. Environmental / variable connectors (`data_connectors.py`)

These feed the **auto-fetch** panel (predictors for a scenario's model), not the corpus.

| Connector | Endpoint | Granularity | Coverage | Key / license | Notes |
|-----------|----------|-------------|----------|---------------|-------|
| **Open-Meteo - Weather** | `api.open-meteo.com` | point (lat/lon) | ERA5, ~1940→present | keyless; **free tier non-commercial** | temp/humidity/precip/wind |
| **Open-Meteo - Air Quality** | `air-quality-api.open-meteo.com` | point (~11 km CAMS) | ~2013→present | keyless; **free tier non-commercial** | PM2.5/PM10/NO₂/O₃ |
| **EAWAG - wastewater** | GitHub CSV | catchment (Geneva, Lausanne, Neuchâtel, Porrentruy) | **frozen archive since 2024-03** | CC BY 4.0 | flu-A/flu-B/RSV/SARS loads; historical backfill only |
| **FOPH - influenza wastewater** | opendata.swiss `influenza1` | national (plants aggregated) | **live, weekly since 2022** (70k+ rows) | opendata.swiss ToU | value/valuemean7d/conc/flow/pop; pass `geoRegion=GE` to narrow |

**Caveats:**
- **Open-Meteo free tier is non-commercial** - a production deployment needs a paid
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

1. **The 600 s budget, not the 2000 cap, is usually the binding constraint** for slow
   sources (arXiv especially, sometimes deep Crossref/OpenAlex pages). Both are
   env-configurable (`POPULATE_FEDERATION_BUDGET`, `LIVE_MAX_PER_SOURCE`) - raise them
   together to fetch deeper; ⚠ multiplies API calls + OpenAI embedding cost per search.
2. **Semantic Scholar**: its relevance endpoint tops out at 1000/query (hard API limit); the bulk
   endpoint, used for a real boolean, has no such limit but returns matches unranked (ID order).
3. **bioRxiv/medRxiv are searched only through Europe PMC** (their own API has no keyword search).
4. **CORE contributes 0 until `CORE_API_KEY` (free) is set** in the environment.
5. **No-abstract records are kept but never relevant on their score** → a reviewer screens
   them by title (the corpus view lists them apart).
6. **arXiv/DOAJ/ClinicalTrials are domain-skewed** (preprints / OA journals / trials) -
   good for recall breadth, not representative on their own.
7. **All caps are per-source, pre-boolean.** Final corpus size = boolean match over the
   union of everything ingested + your local DB, so it is typically **smaller** than the
   sum of raw fetches (dedup + boolean filter).
