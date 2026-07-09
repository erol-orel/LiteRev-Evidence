#!/usr/bin/env python3
"""Live-verify every LiteRev data source + connector — RUN ON PROD (has egress).

    cd /opt/literev-api && .venv/bin/python tools/smoke_test_sources_full.py

(Use the app venv — the connector + parser checks import `main`/`data_connectors`.)

For each of the 13 literature sources it makes a real API call and prints: reachable?,
the source's own TOTAL count for the query, and a sample title. Then it exercises every
data connector, and finally shows the boolean parser on your exact PubMed query.

No DB needed. Keys read from the environment (NCBI_API_KEY, SEMANTIC_SCHOLAR_API_KEY,
CORE_API_KEY) exactly as the app does.
"""
import os
import sys
import xml.etree.ElementTree as ET

import requests

# Run as `.venv/bin/python tools/smoke_test_sources_full.py`: sys.path[0] is tools/, so add
# the app root (parent dir) so the connector/parser checks can `import main`/`data_connectors`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UA = {"User-Agent": "LiteRev-smoke/1.0 (mailto:literev@gesica.ch)"}
TIMEOUT = 30

# Your reported query (PubMed syntax) + a plain-text version for the keyword APIs.
PUBMED_Q = ('("Public Health Schools"[MeSH Terms] OR "public health schools"[Title/Abstract] '
            'OR "schools of public health"[Title/Abstract] OR "public health education"[Title/Abstract]) '
            'AND (ranking[Title/Abstract] OR criteria[Title/Abstract] OR "ranking criteria"[Title/Abstract] '
            'OR "ranking system"[Title/Abstract] OR evaluation[Title/Abstract] OR assessment[Title/Abstract]) '
            'AND (global[Title/Abstract] OR worldwide[Title/Abstract] OR international[Title/Abstract] '
            'OR "in the world"[Title/Abstract])')
PLAIN_Q = "public health schools ranking criteria global evaluation"


def line(name, ok, total, sample):
    flag = "OK " if ok else "FAIL"
    tot = f"total={total}" if total is not None else "total=?"
    smp = f" | {str(sample)[:70]}" if sample else ""
    print(f"  [{flag}] {name:24s} {tot}{smp}")


def _try(fn, name):
    try:
        fn()
    except Exception as e:
        line(name, False, None, f"{type(e).__name__}: {str(e)[:80]}")


# ── 13 literature sources ─────────────────────────────────────────────────────
def src_pubmed():
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    p = {"db": "pubmed", "term": PUBMED_Q, "retmode": "json", "retmax": 0}
    if os.getenv("NCBI_API_KEY"):
        p["api_key"] = os.getenv("NCBI_API_KEY")
    r = requests.get(f"{base}/esearch.fcgi", params=p, headers=UA, timeout=TIMEOUT)
    n = int(r.json()["esearchresult"]["count"])
    line("PubMed (YOUR query)", r.ok, n, "← compare to the scenario's PubMed count")


def src_openalex():
    r = requests.get("https://api.openalex.org/works",
                     params={"search": PLAIN_Q, "per_page": 1, "mailto": "literev@gesica.ch"},
                     headers=UA, timeout=TIMEOUT)
    j = r.json()
    line("OpenAlex", r.ok, j.get("meta", {}).get("count"), (j.get("results") or [{}])[0].get("title"))


def src_crossref():
    r = requests.get("https://api.crossref.org/works",
                     params={"query": PLAIN_Q, "rows": 1, "mailto": "literev@gesica.ch"},
                     headers=UA, timeout=TIMEOUT)
    m = r.json().get("message", {})
    t = (m.get("items") or [{}])[0].get("title", [""])
    line("Crossref", r.ok, m.get("total-results"), t[0] if t else "")


def src_europepmc():
    r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     params={"query": PLAIN_Q, "format": "json", "pageSize": 1, "resultType": "lite"},
                     headers=UA, timeout=TIMEOUT)
    j = r.json()
    line("Europe PMC", r.ok, j.get("hitCount"),
         (j.get("resultList", {}).get("result") or [{}])[0].get("title"))


def src_preprints():
    r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     params={"query": f"({PLAIN_Q}) AND (SRC:PPR)", "format": "json", "pageSize": 1},
                     headers=UA, timeout=TIMEOUT)
    line("Preprints (EPMC PPR)", r.ok, r.json().get("hitCount"), None)


def src_semantic_scholar():
    h = {**UA}
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        h["x-api-key"] = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                     params={"query": PLAIN_Q, "limit": 1, "fields": "title"}, headers=h, timeout=TIMEOUT)
    j = r.json()
    line("Semantic Scholar", r.ok, j.get("total"), (j.get("data") or [{}])[0].get("title"))


def src_doaj():
    import urllib.parse as u
    r = requests.get(f"https://doaj.org/api/search/articles/{u.quote(PLAIN_Q, safe='')}",
                     params={"pageSize": 1}, headers=UA, timeout=TIMEOUT)
    line("DOAJ", r.ok, r.json().get("total"), None)


def src_clinicaltrials():
    r = requests.get("https://clinicaltrials.gov/api/v2/studies",
                     params={"query.term": PLAIN_Q, "pageSize": 1, "countTotal": "true"},
                     headers=UA, timeout=TIMEOUT)
    line("ClinicalTrials.gov", r.ok, r.json().get("totalCount"), None)


def src_core():
    key = os.getenv("CORE_API_KEY")
    if not key:
        line("CORE", False, None, "CORE_API_KEY not set → source is SKIPPED in populate")
        return
    r = requests.post("https://api.core.ac.uk/v3/search/works",
                      headers={"Authorization": f"Bearer {key}", **UA},
                      json={"q": PLAIN_Q, "limit": 1}, timeout=TIMEOUT)
    line("CORE", r.ok, r.json().get("totalHits"), None)


def src_arxiv():
    r = requests.get("http://export.arxiv.org/api/query",
                     params={"search_query": f"all:{PLAIN_Q}", "max_results": 1}, headers=UA, timeout=TIMEOUT)
    root = ET.fromstring(r.text)
    ns = {"os": "http://a9.com/-/spec/opensearch/1.1/"}
    line("arXiv", r.ok, root.findtext("os:totalResults", namespaces=ns), None)


def src_openaire():
    r = requests.get("https://api.openaire.eu/search/publications",
                     params={"keywords": PLAIN_Q, "size": 1, "format": "json"}, headers=UA, timeout=TIMEOUT)
    tot = r.json().get("response", {}).get("header", {}).get("total", {}).get("$")
    line("OpenAIRE", r.ok, tot, None)


def src_biorxiv_medrxiv():
    from datetime import date, timedelta
    to = date.today(); win = f"{(to - timedelta(days=45)).isoformat()}/{to.isoformat()}"
    for server in ("biorxiv", "medrxiv"):
        r = requests.get(f"https://api.biorxiv.org/details/{server}/{win}/0/json", headers=UA, timeout=TIMEOUT)
        n = int((r.json().get("messages") or [{}])[0].get("total", 0) or 0)
        line(f"{server} (45d window)", r.ok, n, "no keyword search — window scan + local filter")


# ── data connectors ───────────────────────────────────────────────────────────
def connectors():
    print("\nData connectors (data_connectors.fetch_series):")
    try:
        import data_connectors as dc
    except Exception as e:
        print(f"  import failed: {e}")
        return
    cases = [
        ("open-meteo-weather", {"region": "geneva", "start_date": "2024-01-01", "end_date": "2024-01-07"}),
        ("open-meteo-air-quality", {"region": "geneva", "start_date": "2024-01-01", "end_date": "2024-01-07"}),
        ("eawag-wastewater", {"region": "geneva"}),
        ("foph-wastewater", {}),
    ]
    for cid, params in cases:
        try:
            rows = dc.fetch_series(cid, params)
            cols = sorted({k for r in rows[:200] for k in r})
            print(f"  [OK ] {cid:24s} rows={len(rows)} cols={cols}")
            if cid == "foph-wastewater" and rows:
                print(f"        sample={rows[0]}")
        except Exception as e:
            print(f"  [FAIL] {cid:24s} {type(e).__name__}: {str(e)[:80]}")


# ── boolean parser (the 109→5 fix) — no network/DB ────────────────────────────
def parser_demo():
    print("\nBoolean parser on YOUR PubMed query (the fix — no DB needed):")
    try:
        import main
        ast = main._parse_boolean_ast(main._tokenize_boolean(PUBMED_Q))
        groups = ast[1] if ast and ast[0] == "and" else [ast]
        sizes = [len(g[1]) if g and g[0] in ("and", "or") else 1 for g in groups]
        print(f"  top-level: '{ast[0] if ast else None}' of {len(groups)} groups (sizes {sizes})")
        params: dict = {}
        main._build_boolean_match_sql_from_query(PUBMED_Q, params)
        phrases = sorted({v.strip('%') for v in params.values()})
        print(f"  phrases matched ({len(phrases)}): {phrases}")
        junk = [p for p in phrases if p in ("titleabstract", "meshterms", "mesh", "terms")]
        print("  → expect 3 groups (4/6/4), NO junk. junk found:", junk or "none")
    except Exception as e:
        print(f"  parser check failed: {e}")


if __name__ == "__main__":
    print("=" * 78)
    print("LiteRev live source verification")
    print(f"keys: NCBI={bool(os.getenv('NCBI_API_KEY'))} "
          f"S2={bool(os.getenv('SEMANTIC_SCHOLAR_API_KEY'))} CORE={bool(os.getenv('CORE_API_KEY'))}")
    print("=" * 78)
    print("13 literature sources (each source's OWN total for the query):")
    for fn, nm in [
        (src_pubmed, "PubMed"), (src_openalex, "OpenAlex"), (src_crossref, "Crossref"),
        (src_europepmc, "Europe PMC"), (src_preprints, "Preprints"),
        (src_semantic_scholar, "Semantic Scholar"), (src_doaj, "DOAJ"),
        (src_clinicaltrials, "ClinicalTrials.gov"), (src_core, "CORE"),
        (src_arxiv, "arXiv"), (src_openaire, "OpenAIRE"), (src_biorxiv_medrxiv, "bioRxiv/medRxiv"),
    ]:
        _try(fn, nm)
    connectors()
    parser_demo()
    print("\nDone. Paste this back if any source shows FAIL or an unexpected total.")
