"""Lexical corpus membership through PostgreSQL full-text search.

WHY
---
Corpus membership is lexical: a document belongs to a scenario when its title, abstract
or full text matches the boolean query (main._boolean_corpus_ids). Until this module,
every term compiled to three `LIKE '%term%'` predicates — title, abstract, an EXISTS over
the document's chunks — assisted by trigram indexes. Measured on the production corpus
(346 152 documents, 1 245 182 chunks) that took 55 to 240 SECONDS per query: the
trigram index yields candidates, and verifying a LIKE means re-reading every candidate
row's text (see the note above main._boolean_ast_to_sql). Substrings matched too:
`%ai%` inside "chain", `%ml%` inside "html" — which is how one scenario's corpus
reached 238 438 documents.

WHAT
----
One tsvector per document, materialised in `document_search` (title + abstract +
full-text chunks, `english` configuration, positions kept so phrases work), indexed
with GIN. The WHOLE boolean expression compiles to ONE tsquery evaluated inside the
index — AND, OR, NOT, quoted phrases and `prefix*` alike — with PER-DOCUMENT
semantics: `A AND B` holds when A and B occur anywhere in the same document, in two
different chunks included, exactly what the LIKE path defined. On the same corpus the
saved queries answer in milliseconds instead of minutes (scripts/compare_fts.py).

What changes for the person searching: stemming (forecast = forecasting = forecasts),
no more substrings (`ai` no longer matches "chain"), stop words ignored ("the", "it",
"of" — PostgreSQL drops them from the query, and `stopword_terms` lets the caller say
so), `word*` is a real prefix, accents are kept.

HOW IT STAYS CURRENT
--------------------
Triggers, so that no write path has to remember:
  literature_document INSERT / UPDATE OF title, abstract  → the row is recomputed at once
  document_chunk INSERT / DELETE / UPDATE OF content       → the row is marked STALE
The chunk side only marks, because chunk writers insert one row per statement and
recomputing a 30-chunk document 30 times would be quadratic. A background worker
(started from main.startup_event) backfills documents that have no row yet and
recomputes stale rows. Until the backfill is complete the search falls back to the
LIKE path — correct, merely slow — so a deploy never answers with a partial corpus.

Limits, accepted on purpose: a tsvector holds at most 1 MB and positions up to
16 383; a full text beyond that keeps its title+abstract vector (phrase queries stop
matching in the tail of very long documents, single words still match).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger("literev-api")

#: Same configuration in the stored vectors and in every query. If these diverge the
#: match silently degrades (a stem on one side only), so it lives in exactly one place.
TS_CONFIG = "english"

#: Chunks whose text is NOT part of the document vector: a `title_abstract` chunk is a
#: copy of two columns the vector already covers, and counting it twice would only
#: double positions. Every other chunk type (fulltext_section, legacy full_text, NULL)
#: is body text.
_HEAD_ONLY_CHUNK_TYPE = "title_abstract"

# ── DDL: table, function, triggers ───────────────────────────────────────────
# Each statement runs in its own transaction (main._exec_ddl_isolated) and is idempotent:
# IF NOT EXISTS / OR REPLACE / "create the trigger only if absent". Triggers are created
# through DO blocks because `CREATE OR REPLACE TRIGGER` needs PostgreSQL 14 and the
# logic lives in the functions anyway — replacing a function updates every trigger.
DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS document_search (
        document_id BIGINT PRIMARY KEY
                    REFERENCES literature_document (id) ON DELETE CASCADE,
        tsv         TSVECTOR,
        -- TRUE = a chunk changed since `tsv` was computed; the worker recomputes it.
        stale       BOOLEAN NOT NULL DEFAULT TRUE,
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_document_search_tsv ON document_search USING gin (tsv)",
    "CREATE INDEX IF NOT EXISTS ix_document_search_stale ON document_search (document_id) WHERE stale",
    # The one definition of "the document's text", used by the trigger and by the worker.
    # VOLATILE on purpose: the worker's UPDATE may wait on a row lock held by a chunk
    # writer, and a volatile function takes a fresh snapshot after that writer commits,
    # so the recomputed vector includes the chunks that were being written.
    f"""
    CREATE OR REPLACE FUNCTION literev_document_tsv(p_id BIGINT) RETURNS TSVECTOR
    LANGUAGE plpgsql VOLATILE AS $$
    DECLARE
        v_head TSVECTOR;
        v_body TSVECTOR;
    BEGIN
        SELECT to_tsvector('{TS_CONFIG}', coalesce(title, '') || ' ' || coalesce(abstract, ''))
          INTO v_head
          FROM literature_document
         WHERE id = p_id;
        IF v_head IS NULL THEN
            RETURN NULL;                       -- no such document
        END IF;
        BEGIN
            SELECT to_tsvector('{TS_CONFIG}', string_agg(content, ' ' ORDER BY chunk_index, id))
              INTO v_body
              FROM document_chunk
             WHERE document_id = p_id
               AND chunk_type IS DISTINCT FROM '{_HEAD_ONLY_CHUNK_TYPE}';
        EXCEPTION WHEN program_limit_exceeded THEN
            v_body := NULL;                    -- full text too large for one tsvector (1 MB)
        END;
        RETURN v_head || coalesce(v_body, ''::tsvector);
    END
    $$
    """,
    """
    CREATE OR REPLACE FUNCTION literev_document_search_doc_trg() RETURNS TRIGGER
    LANGUAGE plpgsql AS $$
    BEGIN
        INSERT INTO document_search (document_id, tsv, stale, updated_at)
        VALUES (NEW.id, literev_document_tsv(NEW.id), FALSE, now())
        ON CONFLICT (document_id) DO UPDATE
            SET tsv = EXCLUDED.tsv, stale = FALSE, updated_at = now();
        RETURN NULL;
    END
    $$
    """,
    f"""
    CREATE OR REPLACE FUNCTION literev_document_search_chunk_trg() RETURNS TRIGGER
    LANGUAGE plpgsql AS $$
    DECLARE
        v_doc  BIGINT;
        v_type TEXT;
    BEGIN
        IF TG_OP = 'DELETE' THEN
            v_doc := OLD.document_id; v_type := OLD.chunk_type;
        ELSE
            v_doc := NEW.document_id; v_type := NEW.chunk_type;
        END IF;
        IF v_type IS DISTINCT FROM '{_HEAD_ONLY_CHUNK_TYPE}' THEN
            IF TG_OP = 'DELETE' THEN
                -- Only mark. Inserting here could race a cascading document delete and
                -- fail its foreign key; a missing row is backfilled by the worker anyway.
                UPDATE document_search SET stale = TRUE WHERE document_id = v_doc;
            ELSE
                INSERT INTO document_search (document_id, tsv, stale)
                SELECT v_doc, NULL, TRUE
                 WHERE EXISTS (SELECT 1 FROM literature_document WHERE id = v_doc)
                ON CONFLICT (document_id) DO UPDATE SET stale = TRUE;
            END IF;
        END IF;
        RETURN NULL;
    END
    $$
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
                       WHERE t.tgname = 'trg_document_search_doc_ins'
                         AND c.relname = 'literature_document') THEN
            CREATE TRIGGER trg_document_search_doc_ins
                AFTER INSERT ON literature_document
                FOR EACH ROW EXECUTE PROCEDURE literev_document_search_doc_trg();
        END IF;
    END
    $$
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
                       WHERE t.tgname = 'trg_document_search_doc_upd'
                         AND c.relname = 'literature_document') THEN
            CREATE TRIGGER trg_document_search_doc_upd
                AFTER UPDATE OF title, abstract ON literature_document
                FOR EACH ROW
                WHEN (OLD.title IS DISTINCT FROM NEW.title OR OLD.abstract IS DISTINCT FROM NEW.abstract)
                EXECUTE PROCEDURE literev_document_search_doc_trg();
        END IF;
    END
    $$
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
                       WHERE t.tgname = 'trg_document_search_chunk_ins'
                         AND c.relname = 'document_chunk') THEN
            CREATE TRIGGER trg_document_search_chunk_ins
                AFTER INSERT ON document_chunk
                FOR EACH ROW EXECUTE PROCEDURE literev_document_search_chunk_trg();
        END IF;
    END
    $$
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
                       WHERE t.tgname = 'trg_document_search_chunk_del'
                         AND c.relname = 'document_chunk') THEN
            CREATE TRIGGER trg_document_search_chunk_del
                AFTER DELETE ON document_chunk
                FOR EACH ROW EXECUTE PROCEDURE literev_document_search_chunk_trg();
        END IF;
    END
    $$
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
                       WHERE t.tgname = 'trg_document_search_chunk_upd'
                         AND c.relname = 'document_chunk') THEN
            CREATE TRIGGER trg_document_search_chunk_upd
                AFTER UPDATE OF content ON document_chunk
                FOR EACH ROW EXECUTE PROCEDURE literev_document_search_chunk_trg();
        END IF;
    END
    $$
    """,
)

TRIGGER_NAMES = (
    "trg_document_search_doc_ins", "trg_document_search_doc_upd",
    "trg_document_search_chunk_ins", "trg_document_search_chunk_del",
    "trg_document_search_chunk_upd",
)


# ── boolean AST → one tsquery ────────────────────────────────────────────────

def ast_to_tsquery_sql(ast, params: dict, idx: list | None = None) -> str | None:
    """Compile main._parse_boolean_ast output into a SQL expression yielding ONE tsquery.

    Leaves bind their term (never interpolated) and become:
      word / "a phrase"   phraseto_tsquery — same stemming as the stored vectors, stop
                          words dropped, a hyphenated word ("sars-cov-2") becoming the
                          same phrase of parts that to_tsvector produced when indexing;
      word*               to_tsquery(quote_literal(word) || ':*') — a prefix match, the
                          PubMed truncation the tokenizer preserves.
    AND / OR / NOT become the tsquery operators && / || / !!, parenthesised, so the
    whole boolean is one index condition regardless of term count.

    A leaf whose term is only stop words compiles to an EMPTY tsquery, which PostgreSQL
    ignores inside && / || (a NOTICE, not an error) and which matches nothing on its
    own — see `stopword_terms`. Returns None for an empty AST: the caller treats that as
    "matches nothing", never as "matches everything".
    """
    if idx is None:
        idx = [0]
    if ast is None:
        return None
    typ = ast[0]
    if typ == "term":
        term = (ast[1] or "").strip()
        if not term:
            return None
        key = f"tq_{idx[0]}"
        idx[0] += 1
        if term.endswith("*") and " " not in term.rstrip("*"):
            stem = term.rstrip("*")
            if not stem:
                return None
            params[key] = stem
            return f"to_tsquery('{TS_CONFIG}', quote_literal(CAST(:{key} AS text)) || ':*')"
        params[key] = term.replace("*", "")
        return f"phraseto_tsquery('{TS_CONFIG}', CAST(:{key} AS text))"
    if typ == "not":
        inner = ast_to_tsquery_sql(ast[1], params, idx)
        return f"(!! {inner})" if inner else None
    if typ in ("and", "or"):
        parts = [p for p in (ast_to_tsquery_sql(ch, params, idx) for ch in ast[1]) if p]
        if not parts:
            return None
        op = " && " if typ == "and" else " || "
        return "(" + op.join(parts) + ")"
    return None


def match_sql(ast, params: dict, alias: str = "s") -> str | None:
    """`<alias>.tsv @@ (<tsquery>)`, or None when the AST has no usable term."""
    q = ast_to_tsquery_sql(ast, params)
    return f"{alias}.tsv @@ {q}" if q else None


def ast_terms(ast) -> list[str]:
    """Every leaf term of an AST, in order (prefix terms keep their trailing '*')."""
    if ast is None:
        return []
    if ast[0] == "term":
        return [ast[1]]
    if ast[0] == "not":
        return ast_terms(ast[1])
    out: list[str] = []
    for ch in ast[1]:
        out += ast_terms(ch)
    return out


# ── runtime state ────────────────────────────────────────────────────────────

_engine = None
_lock = threading.Lock()
_worker_started = False
#: DDL statements the boot could not apply (first line of each), filled by main.py.
#: Non-empty means the search is on the LIKE path for a reason worth reading.
DDL_FAILURES: list[str] = []
_STATE: dict[str, Any] = {
    "ready": False,          # every document has a row → the FTS path answers in full
    "missing": None,         # documents without a row (backfill remaining)
    "stale": None,           # rows whose chunks changed since the vector was computed
    "checked_at": None,      # epoch seconds of the last check of any kind
    "missing_checked_at": None,   # epoch seconds of the last missing-count (the anti-join)
    "last_error": None,
}

#: Seconds a readiness check stays valid for callers that arrive before the worker's
#: first pass (or in processes that never start it, such as tests).
_READY_TTL = 60.0


def configure(engine) -> None:
    global _engine
    _engine = engine


def engine_choice() -> str:
    """LEXICAL_SEARCH_ENGINE: auto (default: full text once backfilled, LIKE before),
    like (force the previous path — the rollback switch), fts (force full text)."""
    v = (os.getenv("LEXICAL_SEARCH_ENGINE") or "auto").strip().lower()
    return v if v in ("auto", "like", "fts") else "auto"


def state() -> dict[str, Any]:
    """For /health: which engine a search would use right now, and why."""
    out = dict(_STATE)
    out["engine"] = "fts" if use_fts(refresh=False) else "like"
    out["choice"] = engine_choice()
    out["ddl_failures"] = list(DDL_FAILURES)
    return out


def _check_state(conn, count_missing: bool = True) -> None:
    """Re-measure. The stale count is an index-only glance; the missing count is an
    anti-join over every document (≈ a few hundred ms on the production corpus), so
    callers in steady state ask for it only every few minutes."""
    from sqlalchemy import text
    now = time.time()
    if count_missing:
        row = conn.execute(text("""
            SELECT (SELECT count(*) FROM literature_document d
                     WHERE NOT EXISTS (SELECT 1 FROM document_search s WHERE s.document_id = d.id)) AS missing,
                   (SELECT count(*) FROM document_search WHERE stale) AS stale
        """)).mappings().first()
        _STATE.update(missing=int(row["missing"]), stale=int(row["stale"]),
                      ready=int(row["missing"]) == 0, checked_at=now, missing_checked_at=now,
                      last_error=None)
    else:
        stale = conn.execute(text("SELECT count(*) FROM document_search WHERE stale")).scalar()
        _STATE.update(stale=int(stale or 0), checked_at=now, last_error=None)


def is_ready(refresh: bool = True) -> bool:
    """True when every document has a row, i.e. the FTS path answers the whole corpus.

    Cheap after the worker's first pass (a cached flag). Before it — a search in the
    first seconds after boot, or a process without the worker — runs the check itself,
    once per `_READY_TTL`. Any failure means "not ready": the LIKE path is the safe side.
    """
    if _engine is None:
        return False
    checked = _STATE["missing_checked_at"]
    if refresh and (checked is None or time.time() - checked > _READY_TTL):
        with _lock:
            checked = _STATE["missing_checked_at"]
            if checked is None or time.time() - checked > _READY_TTL:
                try:
                    with _engine.connect() as c:
                        _check_state(c, count_missing=True)
                except Exception as e:                     # noqa: BLE001 - never raise from a search
                    _STATE.update(ready=False, checked_at=time.time(),
                                  missing_checked_at=time.time(), last_error=str(e)[:200])
    return bool(_STATE["ready"])


def use_fts(refresh: bool = True) -> bool:
    choice = engine_choice()
    if choice == "like":
        return False
    if choice == "fts":
        return _engine is not None
    return is_ready(refresh=refresh)


def stopword_terms(terms: list[str]) -> list[str]:
    """The terms PostgreSQL would silently ignore: only stop words ("the", "it", "of").

    Inside AND/OR an empty tsquery is dropped, so `IT AND infection` means `infection`;
    alone it matches nothing. Best-effort (an error yields []): this is for the log
    line and the person reading it, never for the result. Prefix terms are excluded —
    `to_tsquery` keeps them.
    """
    plain = [t.replace("*", "") for t in terms if not t.endswith("*")]
    if _engine is None or not plain:
        return []
    from sqlalchemy import text
    try:
        with _engine.connect() as c:
            return [r[0] for r in c.execute(text(
                f"SELECT t FROM unnest(CAST(:terms AS text[])) AS t "
                f"WHERE numnode(phraseto_tsquery('{TS_CONFIG}', t)) = 0"),
                {"terms": plain}).all()]
    except Exception:                                      # noqa: BLE001
        return []


# ── the worker: backfill + refresh ───────────────────────────────────────────

_BACKFILL_SQL = """
    INSERT INTO document_search (document_id, tsv, stale, updated_at)
    SELECT d.id, literev_document_tsv(d.id), FALSE, now()
      FROM literature_document d
     WHERE NOT EXISTS (SELECT 1 FROM document_search s WHERE s.document_id = d.id)
     ORDER BY d.id
     LIMIT :n
    ON CONFLICT (document_id) DO NOTHING
"""

# FOR UPDATE SKIP LOCKED: two workers (or a worker and a chunk writer holding the row)
# never wait on each other. A mark set while this runs waits for our commit, then
# lands, and the next pass recomputes — nothing is lost.
_REFRESH_SQL = """
    UPDATE document_search s
       SET tsv = literev_document_tsv(s.document_id), stale = FALSE, updated_at = now()
     WHERE s.document_id IN (SELECT document_id FROM document_search
                              WHERE stale ORDER BY document_id LIMIT :n
                                FOR UPDATE SKIP LOCKED)
"""


def refresh_once(batch: int = 400, max_batches: int = 5, check_missing: bool = True) -> dict[str, Any]:
    """One pass: backfill up to `max_batches` × `batch` missing documents, recompute up
    to as many stale rows, then re-measure. Returns what it did plus the state."""
    from sqlalchemy import text
    if _engine is None:
        raise RuntimeError("lexical_search.configure(engine) has not been called")
    done = {"backfilled": 0, "refreshed": 0}
    backfill = check_missing or _STATE["missing"] is None or (_STATE["missing"] or 0) > 0
    if backfill:
        for _ in range(max_batches):
            with _engine.begin() as c:
                n = c.execute(text(_BACKFILL_SQL), {"n": batch}).rowcount or 0
            done["backfilled"] += n
            if n < batch:
                break
            time.sleep(0.2)                    # leave the database some air
    for _ in range(max_batches):
        with _engine.begin() as c:
            n = c.execute(text(_REFRESH_SQL), {"n": batch}).rowcount or 0
        done["refreshed"] += n
        if n < batch:
            break
    with _engine.connect() as c:
        _check_state(c, count_missing=backfill)
    done.update(_STATE)
    return done


def worker_loop(idle_sleep: float = 30.0, busy_sleep: float = 1.0, missing_every: float = 300.0) -> None:
    """Run forever (daemon thread). Busy while there is work; in steady state one cheap
    stale pass every `idle_sleep` seconds and one missing-count every `missing_every`."""
    was_ready = False
    while True:
        try:
            last = _STATE["missing_checked_at"]
            due = (not _STATE["ready"]) or last is None or time.time() - last > missing_every
            r = refresh_once(check_missing=due)
            busy = bool(r["backfilled"] or r["refreshed"])
            if busy:
                logger.info(f"document_search: +{r['backfilled']} backfilled, "
                            f"{r['refreshed']} refreshed, {r['missing']} missing, {r['stale']} stale")
            if r["ready"] and not was_ready:
                logger.info("document_search: complete — boolean search now uses full text.")
            was_ready = bool(r["ready"])
        except Exception as e:                             # noqa: BLE001 - keep the loop alive
            _STATE.update(last_error=str(e)[:200], ready=False, checked_at=time.time())
            logger.warning(f"document_search worker: {e}")
            busy = False
        time.sleep(busy_sleep if busy else idle_sleep)


def start_worker() -> bool:
    """Start the daemon thread once per process. Returns whether it was started now."""
    global _worker_started
    with _lock:
        if _worker_started or _engine is None:
            return False
        _worker_started = True
    threading.Thread(target=worker_loop, daemon=True, name="document-search").start()
    return True
