#!/usr/bin/env python3
"""Cross-check every number the interface shows for ONE scenario, across endpoints
and caches, and say which ones disagree.

Read-only: only GET endpoints are called (nothing is generated or recomputed).

  # against the running API on the server:
  python3 scripts/audit_scenario.py --base http://127.0.0.1:8000 --scenario usr-xxxx [--lang en]

  # in-process against the local database (DB_URL set):
  python3 scripts/audit_scenario.py --scenario usr-xxxx

Checks (FAIL = numbers that must agree once no pipeline is running; WARN = worth a
look; INFO = state):
  - /counts            server verdict: card count, corpus links, PRISMA screened,
                       above+below threshold
  - /detail, list card corpus_stats.total and article_count vs corpus links; the
                       combined multi-facet expression is present where facets exist
  - /prisma            identification arithmetic (identified − duplicates = unique;
                       unique − removed buckets = screened) and screened = corpus links
  - /embedding-status  corpus_total = corpus links; scored ≤ total
  - /clustering        cache: cluster sizes sum to n_docs; n_docs ≤ n_docs_total ≤
                       corpus links; served language; summaries present
  - /settings          threshold in range; which artefacts are cached
  - /evidence-brief    cached brief present and its language marker
  - /activity          whether this scenario is still running (then mismatches are
                       expected and reported as WARN, not FAIL)
Exit code 1 when a FAIL remains.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Audit:
    def __init__(self, get, sid: str, lang: str):
        self.get, self.sid, self.lang = get, sid, lang
        self.rows: list[tuple[str, str, str]] = []      # (level, check, detail)
        self.running = False

    def add(self, level: str, check: str, detail: str) -> None:
        self.rows.append((level, check, detail))

    def fetch(self, path: str):
        status, body = self.get(path)
        if status != 200:
            self.add("FAIL", path, f"HTTP {status}")
            return None
        return body

    # ── checks ────────────────────────────────────────────────────────────────
    def run(self) -> int:
        sid = self.sid
        activity = self.fetch("/activity") or {}
        running = [a for a in activity.get("running", []) if a.get("scenario_id") == sid]
        self.running = bool(running)
        if running:
            a = running[0]
            self.add("INFO", "activity", f"{a.get('kind')} running, step={a.get('step')} — count mismatches are expected until it ends")
        else:
            self.add("INFO", "activity", "no search or pipeline running for this scenario")

        counts = self.fetch(f"/user-scenarios/{sid}/counts") or {}
        ref = counts.get("corpus_links")
        if counts:
            lvl = "WARN" if self.running else ("OK" if counts.get("consistent") else "FAIL")
            self.add(lvl, "counts (server verdict)",
                     f"corpus_links={ref} article_count={counts.get('article_count')} prisma_screened={counts.get('prisma_screened')} "
                     f"above={counts.get('above_threshold')} below={counts.get('below_threshold')} embedded={counts.get('embedded')} "
                     f"mismatches={counts.get('mismatches') or 'none'}")

        detail = self.fetch(f"/user-scenarios/{sid}/detail") or {}
        if detail:
            total = (detail.get("corpus_stats") or {}).get("total")
            self.add(self._cmp(total, ref), "detail header total", f"corpus_stats.total={total} vs corpus_links={ref}")
            facets = detail.get("facets") or []
            if facets:
                cq = detail.get("combined_query") or ""
                ok = all(f.get("text") in cq for f in facets) and any(op in cq for op in (" AND ", " OR "))
                self.add("OK" if ok else "FAIL", "multi-facet expression",
                         f"{len(facets)} facets, combined_query={cq[:120]!r}")
            else:
                self.add("INFO", "multi-facet expression", "single-query scenario")

        cards = self.fetch("/user-scenarios") or []
        card = next((c for c in cards if c.get("id") == sid), None) if isinstance(cards, list) else None
        if card is None:
            self.add("FAIL", "scenario list card", "scenario not in /user-scenarios")
        else:
            self.add(self._cmp(card.get("article_count"), ref), "scenario list card",
                     f"article_count={card.get('article_count')} vs corpus_links={ref}; "
                     f"populate={card.get('populate_status')} pipeline={card.get('pipeline_status')}")

        prisma = self.fetch(f"/user-scenarios/{sid}/prisma") or {}
        ident = prisma.get("identification") or {}
        if ident:
            identified = ident.get("total_records") or ident.get("records_identified")
            dups, unique = ident.get("duplicates_removed"), ident.get("unique_records")
            buckets = [ident.get(k) or 0 for k in ("removed_no_abstract", "removed_not_matching", "removed_other_reasons")]
            screened = ident.get("records_screened")
            added, gone = int(ident.get("added_after_search") or 0), int(ident.get("removed_after_search") or 0)
            arith_ok = (identified is not None and dups is not None and unique is not None
                        and identified - dups == unique and unique - sum(buckets) + added - gone == screened)
            self.add("OK" if arith_ok else "FAIL", "prisma arithmetic",
                     f"identified={identified} − duplicates={dups} = unique={unique}; unique − removed{buckets}"
                     + (f" + added_after={added}" if added else "") + (f" − removed_after={gone}" if gone else "")
                     + f" = screened={screened} (figures_from={ident.get('figures_from')}, federation_incomplete={ident.get('federation_incomplete')})")
            if added or gone:
                self.add("INFO", "prisma drift since the search",
                         f"the corpus changed after the search that produced the figures: +{added} / −{gone} documents "
                         f"(a rebuild, later pages of a source, duplicates marked since); re-run the search for one consistent run")
            self.add(self._cmp(screened, ref), "prisma screened = corpus", f"records_screened={screened} vs corpus_links={ref}")
        else:
            self.add("WARN", "prisma", "no identification figures (scenario older than the PRISMA accounting, or never searched)")

        emb = self.fetch(f"/user-scenarios/{sid}/embedding-status") or {}
        if emb:
            ct = emb.get("corpus_total")
            rk = emb.get("ranking") or {}
            self.add(self._cmp(ct, ref), "embedding status corpus", f"corpus_total={ct} vs corpus_links={ref}")
            scored, rtotal = rk.get("scored"), rk.get("total")
            if scored is not None and rtotal is not None:
                self.add("OK" if scored <= rtotal else "FAIL", "scoring progress",
                         f"scored={scored}/{rtotal} semantic_ready={(emb.get('score_availability') or {}).get('semantic')}")

        cl = self.fetch(f"/user-scenarios/{sid}/clustering?lang={self.lang}") or {}
        if cl.get("status") == "running":
            self.add("INFO", "clustering", "not cached in this language (a background job was just started)")
        elif cl.get("clusters"):
            n = cl.get("n_docs"); nt = cl.get("n_docs_total", n)
            sizes = sum(c.get("n_docs", 0) for c in cl["clusters"])
            ok = sizes == n and (nt is None or n <= nt) and (ref is None or nt is None or nt <= ref)
            self.add("OK" if ok else "FAIL", "clustering sizes",
                     f"clusters sum={sizes} n_docs={n} n_docs_total={nt} corpus_links={ref} method={cl.get('method')}")
            self.add("OK" if cl.get("lang") == self.lang else "FAIL", "clustering language",
                     f"served lang={cl.get('lang')} requested={self.lang}")
            dense = [c for c in cl["clusters"] if not c.get("is_noise")]
            missing = [c["cluster_id"] for c in dense if not (c.get("summary") or "").strip()]
            self.add("OK" if not missing else "WARN", "cluster summaries",
                     "all dense clusters summarised" if not missing else f"{len(missing)} clusters without summary (cache from the pipeline; regenerated on first open)")
            ps, pt = cl.get("points_shown"), cl.get("points_total")
            if ps is not None and pt is not None:
                self.add("OK" if ps <= pt else "FAIL", "clustering points", f"points_shown={ps} points_total={pt}")
        else:
            self.add("INFO", "clustering", cl.get("message") or "no clustering cache")

        ps = self.fetch(f"/user-scenarios/{sid}/pico-stats") or {}
        if ps and ps.get("total") is not None:
            wp, wo = ps.get("with_pico") or 0, ps.get("without_pico") or 0
            self.add(self._cmp(ps.get("total"), ref), "pico stats corpus", f"total={ps.get('total')} vs corpus_links={ref}")
            self.add("OK" if wp + wo == ps.get("total") else "FAIL", "pico coverage arithmetic",
                     f"with_pico={wp} + without_pico={wo} = total={ps.get('total')} ({ps.get('coverage_pct')}%)")

        st = self.fetch(f"/scenarios/{sid}/settings") or {}
        if st:
            thr = st.get("similarity_threshold")
            self.add("OK" if thr is not None and 0 <= float(thr) <= 1 else "FAIL", "settings threshold", f"similarity_threshold={thr}")
            cached = st.get("cached") or {}
            self.add("INFO", "cached artefacts", ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in cached.items()) or "unknown")

        brief = self.fetch(f"/user-scenarios/{sid}/evidence-brief") or {}
        if brief:
            self.add("INFO", "evidence brief (GET)", f"keys={sorted(k for k in brief.keys() if not k.startswith('_'))[:8]}")

        return self.report()

    def _cmp(self, value, ref) -> str:
        if value is None or ref is None:
            return "WARN"
        if int(value) == int(ref):
            return "OK"
        return "WARN" if self.running else "FAIL"

    def report(self) -> int:
        width = max(len(c) for _, c, _ in self.rows) + 2
        fails = 0
        for level, check, detail in self.rows:
            fails += level == "FAIL"
            print(f"{level:5s} {check:{width}s} {detail}")
        verdict = "ALL NUMBERS AGREE" if not fails else f"{fails} CHECK(S) FAIL"
        print(f"\n{verdict} — scenario {self.sid}" + (" (still running: recheck when it ends)" if self.running else ""))
        return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, help="user scenario id (usr-…)")
    ap.add_argument("--base", help="API base URL; omit for in-process mode against DB_URL")
    ap.add_argument("--lang", default="fr", choices=["fr", "en"], help="language to audit the caches in")
    args = ap.parse_args()

    if args.base:
        import urllib.request

        def get(path: str):
            req = urllib.request.Request(args.base.rstrip("/") + path, headers={"Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.status, json.loads(r.read().decode("utf-8") or "null")
            except urllib.error.HTTPError as e:
                return e.code, None
    else:
        sys.path.insert(0, ROOT)
        os.environ.setdefault("OPENAI_API_KEY", "")
        import main as app_main                            # noqa: E402
        from fastapi.testclient import TestClient
        client = TestClient(app_main.app)

        def get(path: str):
            r = client.get(path)
            return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else None)

    return Audit(get, args.scenario, args.lang).run()


if __name__ == "__main__":
    sys.exit(main())
