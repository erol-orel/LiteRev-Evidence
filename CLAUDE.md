# House rules for this repository

## No em dashes, anywhere, ever

Never write the em dash character (U+2014) in this repository or about it: not in code,
comments, interface strings, LLM prompts, documentation, scripts, workflows, commit
messages, pull request titles or descriptions, nor in chat replies about this project.
Use a comma, a colon, a plain hyphen or a middle dot instead. LLM prompts that generate
text must keep asking the model not to use it.

`tests/test_no_em_dashes.py` scans every tracked text file and fails CI on the first one.

## Every extraction reads all the relevant papers

Any information or evidence extraction (brief, recommended actions, variables and model
spec, epidemiological parameters, concepts, any future one) draws on **all** the relevant
articles of the scenario: those above the similarity threshold, plus those a reviewer
included by hand, and never the excluded. Never a sample, never a top 20, 25, 30 or 40.

A corpus of several thousand abstracts does not fit in one prompt, so the rule is kept by
map then reduce, and any new extraction must follow the same shape:

- **map**: extract the per-article facts once and cache them on the article row
  (`pico_json`, `concepts_json`), so the cost is one-time and incremental;
- **reduce**: `api/digest.py` aggregates those facts over the entire relevant subset in
  SQL, with no LLM and no sampling. The generator writes over that digest and reproduces
  only a handful of articles, for quotation, with an explicit instruction that its
  conclusions must hold for the whole corpus.

Article caps default to zero, meaning no limit. A positive `EPI_PARAM_MAX_ARTICLES`,
`CONCEPT_MAX_ARTICLES` or `CONCEPT_GRAPH_MAX_ARTICLES` is an operational fallback for a
day when the LLM budget must be held, not a normal setting. The clustering and the
similarity graph keep their caps: they are projections bounded by memory, not extractions,
and the interface says how many articles they draw.

`tests/test_full_corpus_digest.py` pins both halves.

## Working conventions

- Backend: the `api/` package, one module per domain; `main.py` is the entry point and
  re-exports every name. Tests patch names with `patch_app` (`tests/conftest.py`).
- The integration tests truncate the tables of `DB_URL`: never point them at a database
  whose data matters.
- Every merge to `main` deploys to production and restarts the API, which cuts the
  searches and pipelines in flight (they are relaunched at startup). Do not merge during
  a presentation.
- Frontend strings live in `frontend/src/i18n/locales/{fr,en}.ts`; both files must carry
  the same keys and the same `{placeholders}`.
- The interface has two themes, and **no colour is written in stone**. Tailwind's hues are
  CSS variables, so a class like `text-rose-300` flips on its own: the light theme gives
  each shade the opposite shade of the same hue. `white` means the foreground, not the
  colour white, so `text-white/70` reads dark on a light page; the white that must stay
  white, on a solid button, is `text-onsolid`. A new hue goes in `HUES`
  (`frontend/tailwind.config.js`) and in `frontend/scripts/gen-theme.mjs`, then
  `node scripts/gen-theme.mjs` regenerates `src/theme.css`. Never write `bg-[#0a1410]`:
  use a theme variable, as in `bg-[rgb(var(--surface))]`. An SVG attribute cannot read a
  CSS variable, so a chart takes its colours from `chartColors(useResolvedTheme())`.
  `tests/test_the_interface_has_two_themes.py` pins all of this.

## Where the keys live (names only, never a value)

- The API reads `/etc/literev-api.env`, the canonical file. On the server,
  `scripts/env_audit.py --live` says which file each setting comes from, by fingerprint
  only; `GET /sources/health` says whether each source key is loaded (`config`).
- `SEMANTIC_SCHOLAR_API_KEY`, `COHERE_API_KEY` and `OPENALEX_API_KEY` are kept in the
  GitHub **environment "production"**, not as repository secrets. The deploy job writes
  them into the canonical file on every deploy, reading the first of their usual names that
  is set (secret or variable: `S2_API_KEY`, `COHERE_KEY`, `OPENALEX_KEY`...), and its log
  names the one that held each key, or says none did, never a value: read it there when a
  key seems missing. The manual server command runs in "production-admin" and sees them
  empty.
- OpenAlex takes its key in the URL (`api_key=`): every OpenAlex call goes through
  `openalex_params` (`api/sources.py`), and any error text that can carry a URL goes through
  `redact_keys` (`api/core.py`) before it is logged or stored. Without the key, OpenAlex
  draws on a small daily budget shared by the server's IP address and answers 429
  "Insufficient budget" once it is spent, until midnight UTC.
- Never print, commit or paste a key: check presence, never values. More in
  `docs/ops-runbook.md`, "Keys".
