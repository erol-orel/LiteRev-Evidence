#!/usr/bin/env python3
"""Compare the CURRENT lexical matching against PostgreSQL full-text search.

Answers the question that decides whether to switch: for your real queries on your real
corpus, does full-text search return MORE papers or FEWER — and *which* ones?

Guessing is worthless here, because the change moves recall in both directions at once:

  gained   stemming unifies word forms: "forecasting" starts matching "forecast",
           "forecasts", "forecasted"; "incidence" matches "incidences".
  lost     substrings stop matching: today `LIKE '%incidence%'` matches "coincidence"
           and `LIKE '%mpox%'` matches inside any longer word. Those disappear — mostly
           a precision win, but it IS a recall change and you should see the sample.
  lost     a quoted phrase becomes a phrase query (`case <-> count`) rather than a
           substring, which is stricter about what sits between the words.

So the script prints the gained and lost titles and lets you judge.

READ-ONLY by default: it computes `to_tsvector` on the fly and creates nothing. That is
slow (a sequential scan per query, minutes on a large corpus) but it cannot alter your
database. `--build` additionally creates two GIN expression indexes so the same
comparison also produces meaningful TIMINGS; `--drop` removes them again. Neither
changes a single row of data — they are indexes on expressions, not schema changes.

Usage
-----
  # results only, nothing created
  python3 scripts/compare_fts.py --query '(mpox OR monkeypox) AND (forecasting OR incidence)'

  # every saved user scenario's query
  python3 scripts/compare_fts.py --from-scenarios --limit 20

  # add the indexes, get timings too, then clean up
  python3 scripts/compare_fts.py --build --from-scenarios
  python3 scripts/compare_fts.py --drop

Reads DB_URL the same way main.py does.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _reexec_in_venv_if_needed() -> None:
    """Re-run under the application's virtualenv when launched with a bare python3.

    `deploy.sh` installs every dependency into /opt/literev-api/.venv, so the obvious
    `python3 scripts/compare_fts.py` dies on `ModuleNotFoundError: fastapi` before doing
    anything. Re-exec instead of lecturing: the command a person naturally types should
    work.
    """
    try:
        import fastapi  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    venv_py = os.path.join(_REPO, ".venv", "bin", "python3")
    if os.path.exists(venv_py) and os.path.realpath(venv_py) != os.path.realpath(sys.executable):
        os.execv(venv_py, [venv_py] + sys.argv)          # replaces this process
    sys.exit(f"This script needs the application's dependencies. Run it with the venv:\n"
             f"  {os.path.join(_REPO, '.venv/bin/python3')} {' '.join(sys.argv)}")


_reexec_in_venv_if_needed()

#: Same text-search configuration in the indexes and in the queries. If these ever
#: diverge the planner silently stops using the index — the exact trap the trigram
#: indexes already hit with a missing COALESCE.
TS_CONFIG = "english"

#: Indexed expressions, built through these two helpers so the DDL and every query are
#: generated from ONE definition. A table alias does not affect how PostgreSQL matches an
#: expression index, but a changed COALESCE or config would — and it would fail silently,
#: exactly as it already can with the trigram indexes.
def doc_tsv(alias: str = "") -> str:
    p = f"{alias}." if alias else ""
    return (f"to_tsvector('{TS_CONFIG}', coalesce({p}title,'') || ' ' "
            f"|| coalesce({p}abstract,''))")


def chunk_tsv(alias: str = "") -> str:
    p = f"{alias}." if alias else ""
    return f"to_tsvector('{TS_CONFIG}', coalesce({p}content,''))"


BUILD_DDL = [
    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_litdoc_fts "
    f"ON literature_document USING gin ({doc_tsv()})",
    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_docchunk_fts "
    f"ON document_chunk USING gin ({chunk_tsv()})",
]
DROP_DDL = [
    "DROP INDEX CONCURRENTLY IF EXISTS ix_litdoc_fts",
    "DROP INDEX CONCURRENTLY IF EXISTS ix_docchunk_fts",
]


# ── boolean AST → tsquery ────────────────────────────────────────────────────

class UnsupportedQuery(Exception):
    """The AST has a shape this comparison cannot express faithfully."""


def ast_to_tsquery_sql(ast, params: dict, idx: list | None = None) -> str:
    """Compile a positive boolean AST into a SQL expression yielding one `tsquery`.

    Each leaf becomes `phraseto_tsquery(...)`, which handles a single word and a quoted
    phrase alike, applies the same stemming as the indexed `to_tsvector`, and drops
    stop words. AND/OR become the `&&` / `||` tsquery operators, so the WHOLE boolean
    expression collapses into ONE index condition regardless of term count — that is
    the entire performance argument for this change.

    Raises `UnsupportedQuery` on a NOT: negation cannot live inside the tsquery without
    breaking per-document semantics (a `!!term` doc-level match would accept a document
    whose title lacks the term while one of its chunks contains it). The caller handles
    negation as a set subtraction instead.
    """
    if idx is None:
        idx = [0]
    if ast is None:
        raise UnsupportedQuery("empty AST")
    typ = ast[0]
    if typ == "term":
        key = f"ts_{idx[0]}"
        idx[0] += 1
        params[key] = ast[1]
        return f"phraseto_tsquery('{TS_CONFIG}', :{key})"
    if typ == "not":
        raise UnsupportedQuery("NOT inside a positive branch")
    if typ in ("and", "or"):
        op = " && " if typ == "and" else " || "
        parts = [ast_to_tsquery_sql(c, params, idx) for c in ast[1]]
        return "(" + op.join(parts) + ")"
    raise UnsupportedQuery(f"unknown node {typ!r}")


def split_positive_negative(ast):
    """(positive_ast, [negated_asts]) for a top-level AND, else (ast, []).

    Mirrors how the current compiler treats `A AND NOT B`, and keeps negation out of the
    tsquery where it would change per-document meaning.
    """
    if ast and ast[0] == "and":
        pos, neg = [], []
        for ch in ast[1]:
            (neg if (ch and ch[0] == "not") else pos).append(ch[1] if (ch and ch[0] == "not") else ch)
        if neg:
            if not pos:
                return None, neg
            return (pos[0] if len(pos) == 1 else ("and", pos)), neg
    if ast and ast[0] == "not":
        return None, [ast[1]]
    return ast, []


def fts_doc_ids_sql(ast, params: dict) -> str:
    """Full boolean AST → SQL returning the matching document ids.

    Two index scans for the positive part (documents, then chunks), minus the same for
    each negated subtree. Compare with the current path, which needs three scans PER
    TERM plus a heap recheck on each.
    """
    pos, negs = split_positive_negative(ast)
    idx = [0]

    def _branch(sub_ast) -> str:
        q = ast_to_tsquery_sql(sub_ast, params, idx)
        return (f"(SELECT ld.id AS did FROM literature_document ld "
                f"WHERE {doc_tsv('ld')} @@ {q}"
                f" UNION SELECT dc.document_id FROM document_chunk dc "
                f"WHERE {chunk_tsv('dc')} @@ {q})")

    out = _branch(pos) if pos is not None else "(SELECT ld.id AS did FROM literature_document ld)"
    for n in negs:
        out = f"({out} EXCEPT {_branch(n)})"
    return out


# ── the comparison ───────────────────────────────────────────────────────────

OUTER = ("d.abstract IS NOT NULL AND length(TRIM(d.abstract)) >= 30 "
         "AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)")


def _old_ids(eng, main, query):
    import sqlalchemy as sa
    params: dict = {}
    frag = main._build_boolean_match_sql_from_query(query, params)
    sql = (f"SELECT d.id FROM literature_document d WHERE ({frag}) AND {OUTER}")
    t = time.perf_counter()
    with eng.connect() as c:
        ids = set(c.execute(sa.text(sql), params).scalars().all())
    return ids, time.perf_counter() - t


def _new_ids(eng, main, query):
    import sqlalchemy as sa
    params: dict = {}
    ast = main._parse_boolean_ast(main._tokenize_boolean(query))
    sub = fts_doc_ids_sql(ast, params)
    sql = (f"SELECT d.id FROM literature_document d JOIN {sub} m ON m.did = d.id "
           f"WHERE {OUTER}")
    t = time.perf_counter()
    with eng.connect() as c:
        ids = set(c.execute(sa.text(sql), params).scalars().all())
    return ids, time.perf_counter() - t


def _titles(eng, ids, n=6):
    import sqlalchemy as sa
    if not ids:
        return []
    with eng.connect() as c:
        return [r[0] for r in c.execute(sa.text(
            "SELECT title FROM literature_document WHERE id = ANY(:ids) LIMIT :n"),
            {"ids": list(ids)[:200], "n": n}).all()]


def compare(eng, main, query, samples):
    print(f"\n{'=' * 78}\nQUERY  {query[:120]}")
    try:
        new, t_new = _new_ids(eng, main, query)
    except UnsupportedQuery as e:
        print(f"  SKIPPED — {e}")
        return None
    old, t_old = _old_ids(eng, main, query)

    gained, lost = new - old, old - new
    pct = (len(new) - len(old)) / len(old) * 100 if old else float("nan")
    print(f"  current (LIKE/trigram) : {len(old):6d} papers   {t_old*1000:8.0f} ms")
    print(f"  full-text  (tsquery)   : {len(new):6d} papers   {t_new*1000:8.0f} ms"
          f"   [{pct:+.1f}%]")
    print(f"  gained {len(gained)}   lost {len(lost)}")
    for label, ids in (("GAINED", gained), ("LOST", lost)):
        for t in _titles(eng, ids, samples):
            print(f"    {label:6s} {(t or '')[:96]}")
    return {"query": query, "old": len(old), "new": len(new),
            "gained": len(gained), "lost": len(lost),
            "ms_old": t_old * 1000, "ms_new": t_new * 1000}


def main_cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", action="append", default=[], help="boolean query (repeatable)")
    ap.add_argument("--from-scenarios", action="store_true",
                    help="use the saved user_scenarios queries")
    ap.add_argument("--limit", type=int, default=10, help="max scenarios to take")
    ap.add_argument("--samples", type=int, default=6, help="titles shown per gained/lost")
    ap.add_argument("--build", action="store_true",
                    help="create the two GIN expression indexes first (no data change)")
    ap.add_argument("--drop", action="store_true", help="drop them and exit")
    args = ap.parse_args()

    import sqlalchemy as sa
    import main as literev  # noqa: N813  (reuses the SAME tokenizer/parser, so the
    #                                      comparison is genuinely apples to apples)

    eng = literev.engine

    if args.drop:
        for ddl in DROP_DDL:
            with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
                c.execute(sa.text(ddl))
            print(f"dropped: {ddl}")
        return 0

    if args.build:
        print("Creating GIN expression indexes (CONCURRENTLY — writes are not blocked).")
        print("This adds indexes only; no column, no row is modified. Minutes on a large "
              "corpus. Remove them again with --drop.\n")
        for ddl in BUILD_DDL:
            t = time.perf_counter()
            with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
                c.execute(sa.text(ddl))
            print(f"  built in {time.perf_counter() - t:6.1f}s: {ddl[:70]}…")
        with eng.connect() as c:
            for name in ("ix_litdoc_fts", "ix_docchunk_fts"):
                size = c.execute(sa.text(
                    "SELECT pg_size_pretty(pg_relation_size(to_regclass(:n)))"),
                    {"n": name}).scalar()
                print(f"  {name}: {size}")
        print()

    queries = list(args.query)
    if args.from_scenarios:
        with eng.connect() as c:
            queries += [r[0] for r in c.execute(sa.text(
                "SELECT DISTINCT query FROM user_scenarios "
                "WHERE query IS NOT NULL AND query <> '' ORDER BY query LIMIT :n"),
                {"n": args.limit}).all()]
    if not queries:
        ap.error("give --query or --from-scenarios")

    rows = [r for r in (compare(eng, literev, q, args.samples) for q in queries) if r]
    if rows:
        tot_old = sum(r["old"] for r in rows)
        tot_new = sum(r["new"] for r in rows)
        print(f"\n{'=' * 78}\nSUMMARY over {len(rows)} queries")
        print(f"  papers: {tot_old} → {tot_new} "
              f"({(tot_new - tot_old) / tot_old * 100:+.1f}%)" if tot_old else "")
        print(f"  gained {sum(r['gained'] for r in rows)}, "
              f"lost {sum(r['lost'] for r in rows)}")
        print(f"  median ms: current {sorted(r['ms_old'] for r in rows)[len(rows)//2]:.0f}"
              f"  full-text {sorted(r['ms_new'] for r in rows)[len(rows)//2]:.0f}")
        if not args.build:
            print("\n  NOTE: without --build there is no FTS index, so the full-text "
                  "timings above are sequential scans and mean nothing. The PAPER COUNTS "
                  "are correct either way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
