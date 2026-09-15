#!/usr/bin/env python3
"""Preflight for a demo: is the API ready to present, right now?

Runs against a live API — on the server (default --base http://127.0.0.1:8000) or from
a laptop (--base https://literev-scenario.com/api) — and prints one line per check,
OK / WARN / FAIL:

  health         database, schema, full-text engine, memory, uptime, rate limits
  OpenAI         one cheap REAL call (a search-strategy translation): the key works and
                 has quota; without it the briefs, variables and actions do not generate
  Cohere         configured or not (the rerank is optional)
  per scenario   the numbers the page shows agree (scripts/audit_scenario.py); every
                 artefact a tab shows is cached and served in the requested language —
                 clustering, knowledge graph, evidence brief, LLM brief, variables,
                 recommended actions, model spec — the missing ones are generated when
                 an API key is given (WRITE_API_KEY or --api-key) and awaited (--wait);
                 and the time of every read endpoint (above --slow-ms is a WARN)

Exit code 1 when a check FAILs. Run it the evening before and again an hour before,
after warming the clustering tab once if the API restarted in between (its first
computation compiles UMAP, about 30 s).

  WRITE_API_KEY=… python3 scripts/preflight_demo.py --scenario usr-aaa --scenario usr-bbb --lang en
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENERATING = ("generating", "running", "starting")


class Preflight:
    def __init__(self, get, post, sids: list[str], lang: str = "fr", api_key: str | None = None,
                 wait_s: int = 300, slow_ms: int = 2000, sleep=time.sleep, audit=None):
        self.get, self.post = get, post
        self.sids, self.lang, self.api_key = sids, lang, api_key
        self.wait_s, self.slow_ms, self.sleep, self.audit = wait_s, slow_ms, sleep, audit
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level: str, check: str, detail: str) -> None:
        self.rows.append((level, check, detail))
        print(f"{level:5s} {check:34s} {detail}", flush=True)

    def timed_get(self, path: str):
        t0 = time.perf_counter()
        status, body = self.get(path)
        ms = (time.perf_counter() - t0) * 1000.0
        return status, body, ms

    def read(self, check: str, path: str, ok=lambda b: "") -> dict | None:
        """GET a read endpoint: FAIL on a non-200, WARN when slow, OK with a detail."""
        status, body, ms = self.timed_get(path)
        if status != 200:
            self.add("FAIL", check, f"HTTP {status} on {path}")
            return None
        detail = ok(body or {}) if isinstance(body, (dict, list)) else ""
        level = "WARN" if ms > self.slow_ms else "OK"
        self.add(level, check, f"{ms:.0f} ms" + (f" · {detail}" if detail else "") + (" · SLOW" if level == "WARN" else ""))
        return body

    def wait_until(self, path: str, ready, label: str) -> dict | None:
        """Poll a GET until `ready(body)` or the wait budget is spent."""
        deadline = time.monotonic() + self.wait_s
        while True:
            status, body = self.get(path)
            if status == 200 and isinstance(body, dict):
                if str(body.get("status", "")) == "error":
                    self.add("FAIL", label, f"generation failed: {body.get('error') or body.get('message') or '?'}")
                    return None
                if ready(body):
                    return body
            if time.monotonic() >= deadline:
                self.add("WARN", label, f"still generating after {self.wait_s} s: check it again before the session")
                return None
            self.sleep(5)

    # ── global checks ────────────────────────────────────────────────────────
    def check_health(self) -> None:
        status, body, ms = self.timed_get("/health")
        if status != 200 or not isinstance(body, dict):
            self.add("FAIL", "health", f"HTTP {status}")
            return
        if body.get("database") != "ok":
            self.add("FAIL", "health: database", str(body.get("database")))
        else:
            self.add("OK", "health: database", f"ok ({ms:.0f} ms)")
        schema = body.get("schema") or {}
        if schema.get("ok") is False:
            self.add("FAIL", "health: schema", f"missing={schema.get('missing_tables')} ddl_failures={schema.get('ddl_failures')}")
        else:
            self.add("OK", "health: schema", "complete")
        lex = body.get("lexical_search") or {}
        engine = lex.get("engine") or lex.get("mode") or "?"
        self.add("OK" if engine == "fts" else "WARN", "health: full-text engine",
                 f"{engine}" + ("" if engine == "fts" else " — boolean searches take the slow LIKE path"))
        proc = body.get("process") or {}
        rss, peak, up = proc.get("rss_mb"), proc.get("rss_peak_mb"), proc.get("uptime_s")
        if rss is not None:
            hours = (up or 0) / 3600.0
            level = "WARN" if (rss or 0) > 2000 else "OK"
            self.add(level, "health: process", f"rss {rss:.0f} MB (peak {peak:.0f} MB), {proc.get('threads')} threads, up {hours:.1f} h"
                     + (" — a restart before the session would free memory" if level == "WARN" else ""))
            if hours < 0.5:
                self.add("WARN", "health: uptime", "the API restarted less than 30 minutes ago: open one clustering tab to warm UMAP")
        rl = body.get("rate_limit")
        if rl:
            self.add("OK", "health: rate limits", f"{rl.get('general_per_min')}/min general, {rl.get('expensive_per_min')}/min on search, ask and pipeline — per IP")

    def check_openai(self) -> None:
        query = f"ambulance demand forecasting preflight {int(time.time())}"
        t0 = time.perf_counter()
        status, body = self.post("/search-strategy", {"query": query})
        ms = (time.perf_counter() - t0) * 1000.0
        if status != 200 or not isinstance(body, dict):
            self.add("FAIL", "OpenAI", f"HTTP {status} on /search-strategy")
        elif body.get("degraded"):
            self.add("FAIL", "OpenAI", "unavailable (no key, quota exhausted or an error): briefs, variables and actions will not generate")
        else:
            self.add("OK", "OpenAI", f"a real translation answered in {ms:.0f} ms")

    def check_llm_usage(self) -> None:
        if not self.api_key:
            return
        status, body = self.get("/llm-usage?hours=24")
        if status == 200 and isinstance(body, dict):
            top = (body.get("by_purpose") or [{}])[0]
            who = f", biggest: {top.get('purpose') or top.get('usage') or top.get('caller') or '?'} ({top.get('total_tokens', 0)} tokens)" if top else ""
            self.add("OK", "OpenAI usage (24 h)",
                     f"{body.get('total_calls', 0)} calls, {body.get('total_tokens', 0)} tokens{who} — check the budget cap (runbook §4)")

    # ── per scenario ─────────────────────────────────────────────────────────
    def check_scenario(self, sid: str) -> None:
        lang = self.lang
        print(f"\n── scenario {sid} ({lang}) ──", flush=True)
        detail = self.read(f"{sid}: detail", f"/user-scenarios/{sid}/detail?lang={lang}",
                           lambda b: f"« {str(b.get('title', ''))[:50]} », {(b.get('corpus_stats') or {}).get('total')} articles")
        if detail is None:
            return
        counts = self.read(f"{sid}: counts", f"/user-scenarios/{sid}/counts",
                           lambda b: "consistent" if b.get("consistent") else ("still running" if b.get("in_progress") else f"DISAGREE {b.get('mismatches') or b.get('details') or ''}"))
        if counts and not counts.get("consistent"):
            self.add("WARN" if counts.get("in_progress") else "FAIL", f"{sid}: counts", "the header, the list and the PRISMA disagree")
        if self.audit:
            rc, out = self.audit(sid)
            for line in out.splitlines():
                print("      " + line)
            self.add("OK" if rc == 0 else "FAIL", f"{sid}: audit", "all numbers agree" if rc == 0 else "see the lines above")

        # clustering: served from cache in the requested language, or computed now
        status, body, ms = self.timed_get(f"/user-scenarios/{sid}/clustering?lang={lang}")
        if status != 200 or not isinstance(body, dict):
            self.add("FAIL", f"{sid}: clustering", f"HTTP {status}")
        else:
            if str(body.get("status", "")) in GENERATING:
                body = self.wait_until(f"/user-scenarios/{sid}/clustering/status?lang={lang}",
                                       lambda b: b.get("status") == "done" or bool(b.get("clusters")), f"{sid}: clustering")
            if body and body.get("clusters") is not None:
                n = len(body.get("clusters") or [])
                served = body.get("lang")
                level = "OK" if (served in (None, lang)) else "WARN"
                self.add(level, f"{sid}: clustering", f"{n} clusters on {body.get('n_docs')} docs, cached={bool(body.get('from_cache'))}, lang={served}" + ("" if level == "OK" else f" (asked {lang})"))
        self.read(f"{sid}: knowledge graph", f"/user-scenarios/{sid}/knowledge-graph",
                  lambda b: f"{len(b.get('nodes') or [])} nodes, {len(b.get('edges') or b.get('links') or [])} edges")
        self.read(f"{sid}: evidence brief", f"/user-scenarios/{sid}/evidence-brief",
                  lambda b: f"{(b.get('corpus_stats') or {}).get('total', '?')} articles in the brief")
        self.artifact(sid, "LLM brief", f"/scenarios/{sid}/evidence-brief/llm?lang={lang}",
                      f"/scenarios/{sid}/evidence-brief/generate?lang={lang}")
        self.artifact(sid, "variables", f"/scenarios/{sid}/variables?lang={lang}",
                      f"/scenarios/{sid}/variables/generate?lang={lang}")
        self.artifact(sid, "recommended actions", f"/scenarios/{sid}/recommended-actions?lang={lang}", None,
                      ready=lambda b: b.get("status") == "ready")
        self.read(f"{sid}: model spec", f"/scenarios/{sid}/model/spec?lang={lang}",
                  lambda b: f"status={b.get('status', 'ready')}")
        self.read(f"{sid}: PRISMA", f"/user-scenarios/{sid}/prisma",
                  lambda b: f"{((b.get('identification') or {}).get('total_records') or (b.get('identification') or {}).get('total_records_identified'))} records")
        self.read(f"{sid}: PICO stats", f"/user-scenarios/{sid}/pico-stats",
                  lambda b: f"{b.get('with_pico', b.get('extracted', '?'))} with PICO")
        emb = self.read(f"{sid}: embedding status", f"/user-scenarios/{sid}/embedding-status",
                        lambda b: (f"{b.get('status_label') or b.get('status')}, "
                                   f"{(b.get('title_abstract_chunks') or {}).get('embedded_docs', '?')}/{b.get('corpus_total', '?')} docs embedded, "
                                   f"{(b.get('ranking') or {}).get('scored', '?')} scored, "
                                   f"cohere={'on' if (b.get('score_availability') or {}).get('cohere_configured') else 'off'}"))
        if emb is not None and not (emb.get("score_availability") or {}).get("cohere_configured"):
            self.add("WARN", f"{sid}: Cohere", "no COHERE_API_KEY: the relevance order is the cosine score only")
        self.read(f"{sid}: screening", f"/user-scenarios/{sid}/screening-progress")
        self.read(f"{sid}: model monitor", f"/scenarios/{sid}/model/monitor?lang={lang}",
                  lambda b: f"status={b.get('status', '?')}")
        self.read(f"{sid}: SEIR", f"/scenarios/{sid}/seir/projection",
                  lambda b: "applicable" if b.get("applicable", b.get("ok", True)) else f"not applicable ({b.get('reason_code') or b.get('reason')})")

    def artifact(self, sid: str, name: str, get_path: str, generate_path: str | None,
                 ready=lambda b: b.get("status") not in ("empty", "generating", "error", "running")) -> None:
        """A cached LLM artefact: OK when served, generated and awaited when missing."""
        label = f"{sid}: {name}"
        status, body, ms = self.timed_get(get_path)
        if status != 200 or not isinstance(body, dict):
            self.add("FAIL", label, f"HTTP {status}")
            return
        st = str(body.get("status", ""))
        if ready(body):
            self.add("OK", label, f"cached ({ms:.0f} ms)" + (f", lang={body.get('lang') or body.get('_lang') or lang_of(body)}" if lang_of(body) else ""))
            return
        if st == "error":
            self.add("FAIL", label, f"last generation failed: {body.get('message') or body.get('error')}")
            return
        if st not in GENERATING:
            if generate_path is None:
                pass                                        # the GET itself starts the generation
            elif not self.api_key:
                self.add("WARN", label, "not generated: open the tab once in the interface, or rerun with an API key to generate it now")
                return
            else:
                gs, gb = self.post(generate_path, {})
                if gs not in (200, 202):
                    self.add("FAIL", label, f"generate: HTTP {gs} {str(gb)[:80]}")
                    return
        body = self.wait_until(get_path, ready, label)
        if body is not None:
            self.add("OK", label, "generated now and cached")

    def report(self) -> int:
        fails = sum(1 for lvl, _, _ in self.rows if lvl == "FAIL")
        warns = sum(1 for lvl, _, _ in self.rows if lvl == "WARN")
        print(f"\n{len(self.rows)} checks: {fails} FAIL, {warns} WARN")
        print("READY TO PRESENT" if not fails else "NOT READY — fix the FAIL lines")
        return 1 if fails else 0

    def run(self) -> int:
        self.check_health()
        self.check_openai()
        self.check_llm_usage()
        for sid in self.sids:
            self.check_scenario(sid)
        return self.report()


def lang_of(body: dict) -> str | None:
    for key in ("lang", "_lang", "language"):
        if isinstance(body.get(key), str):
            return body[key]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL (from a laptop: https://literev-scenario.com/api)")
    ap.add_argument("--scenario", action="append", required=True, help="scenario id to check (repeatable)")
    ap.add_argument("--lang", default="fr", choices=["fr", "en"], help="language the session will use")
    ap.add_argument("--api-key", default=os.environ.get("WRITE_API_KEY"), help="write key: generates the missing artefacts and reads the LLM usage")
    ap.add_argument("--wait", type=int, default=300, help="seconds to wait for a generation")
    ap.add_argument("--slow-ms", type=int, default=2000)
    ap.add_argument("--no-audit", action="store_true", help="skip scripts/audit_scenario.py")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    headers = {"Accept-Encoding": "identity"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    def _call(method: str, path: str, payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={**headers, **({"Content-Type": "application/json"} if data is not None else {})})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read().decode("utf-8")
                return r.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, raw
        except Exception as e:                                # noqa: BLE001 — network
            return 0, str(e)

    def get(path):
        return _call("GET", path)

    def post(path, payload):
        return _call("POST", path, payload)

    def audit(sid: str):
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "audit_scenario.py"),
               "--base", base, "--scenario", sid, "--lang", args.lang]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return proc.returncode, (proc.stdout + proc.stderr).strip()

    pf = Preflight(get, post, args.scenario, args.lang, args.api_key, args.wait, args.slow_ms,
                   audit=None if args.no_audit else audit)
    return pf.run()


if __name__ == "__main__":
    sys.exit(main())
