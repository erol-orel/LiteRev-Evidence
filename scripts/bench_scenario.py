#!/usr/bin/env python3
"""Measure every read endpoint the scenario page and the search page call, for ONE
scenario: wall time, payload size and (in-process mode) resident memory delta.

Two modes:

  # 1) In-process, against the local database, on a SYNTHETIC corpus of N articles
  #    (seeded on the fly under the id usr-bench-<N>; --cleanup removes it):
  DB_URL=postgresql+psycopg://... python3 scripts/bench_scenario.py --seed 25000

  # 2) Over HTTP against a running API (e.g. production on the server), on an
  #    EXISTING scenario — nothing is written, only GET endpoints are called:
  python3 scripts/bench_scenario.py --base http://127.0.0.1:8000 --scenario usr-xxxx

Endpoints flagged with "*" may start background work on a cold cache (automatic
scoring on the corpus, clustering / knowledge graph computation, LLM generation
of recommended actions or variables); pass --read-only to skip them on a shared
server. Results are sorted by time; anything above --slow-ms (default 2000) or
--big-kb (default 2000) is flagged, and the exit code is 1 when something is
flagged (so the script can gate a CI job).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label, path template, may_start_background_work)
ENDPOINTS: list[tuple[str, str, bool]] = [
    ("health", "/health", False),
    ("scenario list (user)", "/user-scenarios", False),
    ("scenario list (dashboard)", "/gesica/scenarios", False),
    ("activity bar", "/activity", False),
    ("detail (header)", "/user-scenarios/{sid}/detail", False),
    ("counts (banner poll)", "/user-scenarios/{sid}/counts", False),
    ("corpus page (Corpus tab)", "/user-scenarios/{sid}/corpus?limit=200", True),
    ("corpus 10k (search results page)", "/user-scenarios/{sid}/corpus?limit=10000&abstract_chars=600", True),
    ("embedding status", "/user-scenarios/{sid}/embedding-status", False),
    ("prisma", "/user-scenarios/{sid}/prisma", False),
    ("pipeline status", "/user-scenarios/{sid}/pipeline/status", False),
    ("populate status", "/user-scenarios/{sid}/populate/status", False),
    ("screening progress", "/user-scenarios/{sid}/screening-progress", False),
    ("pico stats", "/user-scenarios/{sid}/pico-stats", False),
    ("pico bulk (50)", "/user-scenarios/{sid}/pico-bulk?limit=50&offset=0", False),
    ("double-blind kappa", "/user-scenarios/{sid}/double-blind/kappa", False),
    ("model status", "/user-scenarios/{sid}/model-status", False),
    ("knowledge graph", "/user-scenarios/{sid}/knowledge-graph", True),
    ("clustering (cache)", "/user-scenarios/{sid}/clustering?lang=fr", True),
    ("evidence brief (cache)", "/user-scenarios/{sid}/evidence-brief", False),
    ("settings (threshold)", "/scenarios/{sid}/settings", False),
    ("rerank status", "/scenarios/{sid}/rerank/status", False),
    ("variables", "/scenarios/{sid}/variables?lang=fr", True),
    ("recommended actions", "/scenarios/{sid}/recommended-actions?lang=fr", True),
    ("model data", "/scenarios/{sid}/model/data", False),
    ("model monitor", "/scenarios/{sid}/model/monitor", False),
    ("model spec", "/scenarios/{sid}/model/spec?lang=fr", True),
    ("seir projection", "/scenarios/{sid}/seir/projection", False),
]

WORDS = ("influenza surveillance wastewater forecasting hospital admission emergency "
         "department antimicrobial resistance vaccination coverage outbreak detection "
         "machine learning artificial intelligence early warning epidemic model cohort "
         "randomized trial public health intervention mortality incidence prevalence").split()


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def seed(engine, n: int, sid: str) -> None:
    """Create a synthetic scenario of `n` articles with abstracts, one title_abstract
    chunk each (no embedding), scored links and a fake clustering cache."""
    from sqlalchemy import text
    rnd = random.Random(42)
    with engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :sid"), {"sid": sid})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :sid"), {"sid": sid})
        c.execute(text("DELETE FROM scenario_settings WHERE scenario_id = :sid"), {"sid": sid})
        c.execute(text("""
            INSERT INTO user_scenarios (id, name, query, mode, filters, pinned, article_count,
                                        populate_status, pipeline_status, result_count)
            VALUES (:sid, :name, :q, 'boolean', '{}', TRUE, :n, 'done', 'done', :n)
        """), {"sid": sid, "name": f"Benchmark {n} articles", "q": "influenza AND surveillance", "n": n})
    ids: list[int] = []
    batch = 2000
    # Optional columns differ between a production database and a test database
    # bootstrapped by the suite: insert only the ones that exist.
    with engine.connect() as c:
        colinfo = {r[0]: r[1] for r in c.execute(text(
            "SELECT column_name, column_default FROM information_schema.columns "
            "WHERE table_name = 'literature_document'"))}
        next_id = int(c.execute(text("SELECT COALESCE(MAX(id), 0) + 1 FROM literature_document")).scalar())
    optional = [col for col in ("url", "external_id", "project_context") if col in colinfo]
    # Ids are assigned here (executemany cannot RETURN them, and a test database may
    # lack the id sequence); a sequence, when present, is advanced afterwards.
    cols = ["id", "source", "title", "abstract", "year"] + optional
    insert_sql = (f"INSERT INTO literature_document ({', '.join(cols)}) "
                  f"VALUES ({', '.join(':' + col for col in cols)})")
    for start in range(0, n, batch):
        rows = []
        for i in range(start, min(n, start + batch)):
            words = [rnd.choice(WORDS) for _ in range(230)]          # ~1.5 KB abstract
            rows.append({
                "id": next_id + i,
                "title": f"Benchmark article {i}: " + " ".join(rnd.choice(WORDS) for _ in range(8)),
                "abstract": " ".join(words), "year": 2000 + (i % 26),
                "source": rnd.choice(["pubmed", "europepmc", "openalex", "crossref"]),
                "url": f"https://example.org/bench/{i}", "external_id": f"bench-{i}",
                "project_context": "literev",
            })
        with engine.begin() as c:
            c.execute(text(insert_sql), rows)
            new_ids = [r["id"] for r in rows]
            ids.extend(new_ids)
            c.execute(text("""
                INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, screening_status)
                VALUES (:sid, :did, :score, NULL)
                ON CONFLICT DO NOTHING
            """), [{"sid": sid, "did": d, "score": round(rnd.uniform(0.2, 0.9), 3)} for d in new_ids])
            c.execute(text("""
                INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type)
                SELECT id, 0, left(title || E'\\n\\n' || coalesce(abstract, ''), 2000), 'title_abstract'
                FROM literature_document WHERE id = ANY(CAST(:ids AS bigint[]))
            """), {"ids": new_ids})
        print(f"  seeded {len(ids)}/{n}", file=sys.stderr)
    if colinfo.get("id"):                          # keep the sequence ahead of the explicit ids
        with engine.begin() as c:
            c.execute(text("SELECT setval(pg_get_serial_sequence('literature_document', 'id'), "
                           "(SELECT MAX(id) FROM literature_document))"))
    # Fake clustering cache with one point per article (what the tab downloads).
    clusters = []
    per = max(1, n // 12)
    for k in range(12):
        chunk = ids[k * per:(k + 1) * per] if k < 11 else ids[11 * per:]
        clusters.append({
            "cluster_id": k, "cluster_name": f"Cluster {k + 1}", "is_noise": False, "n_docs": len(chunk),
            "center_x": float(k), "center_y": float(k % 3), "top_words": WORDS[:10],
            "summary": "Résumé synthétique.", "summaries": {"fr": "Résumé synthétique."},
            "representative_doc": {"id": chunk[0] if chunk else 0, "title": "t", "year": 2020, "journal": "j"},
            "points": [{"id": d, "title": f"Benchmark article {j}", "year": 2010,
                        "x": float(k) + rnd.random(), "y": float(k % 3) + rnd.random()}
                       for j, d in enumerate(chunk)],
        })
    payload = {"scenario_id": sid, "n_docs": n, "n_clusters": 12, "method": "hdbscan",
               "embedding_source": "synthetic", "clusters": clusters, "lang": "fr", "from_cache": False}
    with engine.begin() as c:
        c.execute(text("""
            INSERT INTO scenario_settings (scenario_id, clustering_json, clustering_generated_at, updated_at)
            VALUES (:sid, CAST(:p AS jsonb), NOW(), NOW())
            ON CONFLICT (scenario_id) DO UPDATE
            SET clustering_json = CAST(:p AS jsonb), clustering_generated_at = NOW(), updated_at = NOW()
        """), {"sid": sid, "p": json.dumps(payload)})


def cleanup(engine, sid: str) -> None:
    from sqlalchemy import text
    with engine.begin() as c:
        c.execute(text("""
            DELETE FROM literature_document
            WHERE id IN (SELECT document_id FROM article_scenarios WHERE scenario_id = :sid)
              AND title LIKE 'Benchmark article %'
        """), {"sid": sid})
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :sid"), {"sid": sid})
        c.execute(text("DELETE FROM scenario_settings WHERE scenario_id = :sid"), {"sid": sid})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :sid"), {"sid": sid})


def run(get, sid: str, read_only: bool, slow_ms: int, big_kb: int, in_process: bool) -> int:
    rows = []
    for label, tmpl, bg in ENDPOINTS:
        if read_only and bg:
            continue
        path = tmpl.replace("{sid}", sid)
        rss0 = _rss_mb() if in_process else 0.0
        t0 = time.perf_counter()
        try:
            status, nbytes = get(path)
            err = ""
        except Exception as e:                       # noqa: BLE001
            status, nbytes, err = 0, 0, str(e)[:80]
        ms = (time.perf_counter() - t0) * 1000.0
        rss1 = _rss_mb() if in_process else 0.0
        rows.append((ms, nbytes / 1024.0, rss1 - rss0, status, label + (" *" if bg else ""), path, err))
    rows.sort(key=lambda r: -r[0])
    flagged = 0
    print(f"{'ms':>8} {'KB':>9} {'ΔRSS MB':>8} {'st':>4}  endpoint")
    for ms, kb, drss, status, label, path, err in rows:
        flag = ""
        if ms > slow_ms:
            flag += " SLOW"
        if kb > big_kb:
            flag += " BIG"
        if status >= 500 or status == 0:
            flag += " ERROR"
        if flag:
            flagged += 1
        rss_txt = f"{drss:8.1f}" if in_process else f"{'':>8}"
        print(f"{ms:8.0f} {kb:9.1f} {rss_txt} {status:4d}  {label:34s} {path}{'  ' + err if err else ''}{flag}")
    print(f"\n{len(rows)} endpoints, {flagged} flagged (slow > {slow_ms} ms, big > {big_kb} KB, or error).")
    return 1 if flagged else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="API base URL (HTTP mode); omit for in-process mode")
    ap.add_argument("--scenario", help="existing scenario id (required in HTTP mode)")
    ap.add_argument("--seed", type=int, default=0, help="in-process mode: seed a synthetic corpus of N articles")
    ap.add_argument("--cleanup", action="store_true", help="in-process mode: delete the synthetic corpus at the end")
    ap.add_argument("--read-only", action="store_true", help="skip endpoints that may start background work")
    ap.add_argument("--slow-ms", type=int, default=2000)
    ap.add_argument("--big-kb", type=int, default=2000)
    args = ap.parse_args()

    if args.base:
        if not args.scenario:
            ap.error("--scenario is required with --base")
        import urllib.request

        def get(path: str) -> tuple[int, int]:
            req = urllib.request.Request(args.base.rstrip("/") + path, headers={"Accept-Encoding": "identity"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return r.status, len(r.read())
            except urllib.error.HTTPError as e:
                return e.code, len(e.read() or b"")
        return run(get, args.scenario, args.read_only, args.slow_ms, args.big_kb, in_process=False)

    sys.path.insert(0, ROOT)
    os.environ.setdefault("OPENAI_API_KEY", "")
    import main as app_main                                   # noqa: E402  (imports the API)
    from fastapi.testclient import TestClient

    sid = args.scenario or f"usr-bench-{args.seed}"
    if args.seed:
        print(f"Seeding {args.seed} articles into {sid}…", file=sys.stderr)
        t0 = time.perf_counter()
        seed(app_main.engine, args.seed, sid)
        print(f"  seeded in {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    client = TestClient(app_main.app)

    def get(path: str) -> tuple[int, int]:
        r = client.get(path)
        return r.status_code, len(r.content)

    try:
        return run(get, sid, args.read_only, args.slow_ms, args.big_kb, in_process=True)
    finally:
        if args.cleanup and args.seed:
            cleanup(app_main.engine, sid)
            print("cleaned up", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
