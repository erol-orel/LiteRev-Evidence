#!/usr/bin/env python3
"""Same boolean query → every live source → count comparison. RUN ON PROD (has egress).

    cd /opt/literev-api && .venv/bin/python tools/boolean_counts.py
    cd /opt/literev-api && .venv/bin/python tools/boolean_counts.py '("public health") AND (ranking OR criteria)'

Unlike smoke_test_sources_full.py (which sends bare keywords just to check REACHABILITY),
this sends the SAME boolean to every source using the EXACT per-source translation the app
uses in populate — main._strip_field_tags / _boolean_to_arxiv / _boolean_to_s2 / the portable
boolean — and prints each source's own total for that query. That is the apples-to-apples
view: the corpus is the SOURCE-UNION of these boolean hits (minus records with no abstract).

Only Crossref (relevance-only upstream) and bioRxiv/medRxiv (no keyword API — date-window scan)
can't take the boolean; every other source does. Pass a natural-language query and it is first
translated via the app's LLM strategy (needs OPENAI_API_KEY); pass a boolean and it's used as-is
(no LLM call). Uses the app venv — it imports `main`. No DB writes.
"""
import logging
import os
import sys
import urllib.parse as _url
import xml.etree.ElementTree as ET

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Load the same env files the service loads (OPENAI / NCBI / S2 / CORE keys) before import.
def _load_env_file(path):
    try:
        with open(path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                if _k and _k not in os.environ:
                    os.environ[_k] = _v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass


for _ep in ["/opt/literev-api/.env", "/etc/literev/env", "/etc/literev-api.env",
            "/etc/literev/secrets", "/opt/literev-api/secrets.env"]:
    _load_env_file(_ep)

logging.disable(logging.INFO)   # hush main.py's import-time DDL "vérifiées/créées" chatter
import main  # noqa: E402  — reuse the app's OWN translators so we send what the app sends

UA = {"User-Agent": "LiteRev-boolcount/1.0 (mailto:literev@gesica.ch)"}
TIMEOUT = 30

QUERY = " ".join(sys.argv[1:]).strip() or (
    '("public health schools" OR "schools of public health" OR "public health education") '
    'AND (ranking OR criteria OR evaluation OR assessment) '
    'AND (global OR worldwide OR international)')

# ── replicate main.py populate routing EXACTLY (main.py ~7841-7895, 8328) ──────
if main._looks_boolean(QUERY):
    _boolean = _pubmed_q = QUERY                     # boolean given → use as-is (no LLM)
else:
    _strat = main._generate_search_strategy(QUERY)   # natural → app LLM strategy
    _boolean = (_strat.get("general") or QUERY) if isinstance(_strat, dict) else QUERY
    _pubmed_q = (_strat.get("pubmed") or _boolean) if isinstance(_strat, dict) else _boolean

_plain_q = main._plain_keywords(_boolean) or main._plain_keywords(QUERY) or QUERY
_portable_bool = main._strip_field_tags(_boolean).strip()
_bool_is_real = bool(_portable_bool) and main._looks_boolean(_portable_bool)
_send_bool = _bool_is_real and len(_portable_bool) <= 1200
_bool_query = _portable_bool if _send_bool else _plain_q          # OpenAlex / DOAJ / CORE / CT
_oa_q = _portable_bool if _bool_is_real else _plain_q             # OpenAIRE Graph v2

_arxiv_q, _arxiv_native = f"all:{_plain_q}", False
if _bool_is_real:
    try:
        _ax = main._boolean_to_arxiv(main._parse_boolean_ast(main._tokenize_boolean(_portable_bool)))
        if _ax and len(_ax) <= 1200:
            _arxiv_q, _arxiv_native = _ax, True
    except Exception:
        pass

_s2_bool = None
if _bool_is_real:
    try:
        _s2_bool = main._boolean_to_s2(main._parse_boolean_ast(main._tokenize_boolean(_portable_bool)))
    except Exception:
        _s2_bool = None


def line(name, sent_as, ok, total, sent):
    flag = "OK " if ok else "FAIL"
    tot = f"{total:>9}" if isinstance(total, int) else f"{'?':>9}"
    print(f"  [{flag}] {name:20s} {sent_as:14s} {tot}   {str(sent)[:60]}")


# ── per-source COUNT with the query the APP sends that source ──────────────────
def c_pubmed():
    p = {"db": "pubmed", "term": _pubmed_q, "retmode": "json", "retmax": 0}
    if os.getenv("NCBI_API_KEY"):
        p["api_key"] = os.getenv("NCBI_API_KEY")
    n = int(requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                         params=p, headers=UA, timeout=TIMEOUT).json()["esearchresult"]["count"])
    line("PubMed", "pubmed-bool", True, n, _pubmed_q)


def c_europepmc():
    n = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     params={"query": _boolean, "format": "json", "pageSize": 1},
                     headers=UA, timeout=TIMEOUT).json().get("hitCount")
    line("Europe PMC", "general-bool", True, n, _boolean)


def c_preprints():
    q = f"({_boolean}) AND (SRC:PPR)"
    n = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     params={"query": q, "format": "json", "pageSize": 1},
                     headers=UA, timeout=TIMEOUT).json().get("hitCount")
    line("Preprints EPMC", "general-bool", True, n, q)


def c_openalex():
    n = requests.get("https://api.openalex.org/works",
                     params={"search": _bool_query, "per_page": 1, "mailto": "literev@gesica.ch"},
                     headers=UA, timeout=TIMEOUT).json().get("meta", {}).get("count")
    line("OpenAlex", "portable-bool" if _send_bool else "plain", True, n, _bool_query)


def c_doaj():
    n = requests.get(f"https://doaj.org/api/search/articles/{_url.quote(_bool_query, safe='')}",
                     params={"pageSize": 1}, headers=UA, timeout=TIMEOUT).json().get("total")
    line("DOAJ", "portable-bool" if _send_bool else "plain", True, n, _bool_query)


def c_clinicaltrials():
    n = requests.get("https://clinicaltrials.gov/api/v2/studies",
                     params={"query.term": _bool_query, "pageSize": 1, "countTotal": "true"},
                     headers=UA, timeout=TIMEOUT).json().get("totalCount")
    line("ClinicalTrials", "portable-bool" if _send_bool else "plain", True, n, _bool_query)


def c_core():
    key = os.getenv("CORE_API_KEY")
    if not key:
        line("CORE", "portable-bool", False, None, "CORE_API_KEY not set → SKIPPED in populate")
        return
    n = requests.post("https://api.core.ac.uk/v3/search/works",
                      headers={"Authorization": f"Bearer {key}", **UA},
                      json={"q": _bool_query, "limit": 1}, timeout=TIMEOUT).json().get("totalHits")
    line("CORE", "portable-bool" if _send_bool else "plain", True, n, _bool_query)


def c_arxiv():
    root = ET.fromstring(requests.get("http://export.arxiv.org/api/query",
                         params={"search_query": _arxiv_q, "max_results": 1},
                         headers=UA, timeout=TIMEOUT).text)
    n = root.findtext("os:totalResults", namespaces={"os": "http://a9.com/-/spec/opensearch/1.1/"})
    line("arXiv", "arxiv-dialect" if _arxiv_native else "plain",
         True, int(n) if n and n.isdigit() else n, _arxiv_q)


def c_semantic_scholar():
    h = {**UA}
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        h["x-api-key"] = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if _s2_bool:                                          # bulk endpoint (what the app uses)
        j = requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",
                         params={"query": _s2_bool, "fields": "title"}, headers=h, timeout=TIMEOUT).json()
        line("Semantic Scholar", "s2-bulk", True, j.get("total"), _s2_bool)
    else:                                                 # relevance endpoint + plain
        j = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                         params={"query": _plain_q, "limit": 1, "fields": "title"}, headers=h, timeout=TIMEOUT).json()
        line("Semantic Scholar", "plain", True, j.get("total"), _plain_q)


def c_openaire():
    j = requests.get("https://api.openaire.eu/graph/v2/researchProducts",
                     params={"search": _oa_q, "pageSize": 1},
                     headers={"Accept": "application/json", **UA}, timeout=TIMEOUT).json()
    hdr = j.get("header") or {}
    n = hdr.get("numFound") or hdr.get("total") or len(j.get("results") or [])
    line("OpenAIRE", "portable-bool" if _bool_is_real else "plain", True, n, _oa_q)


def c_crossref():
    # Crossref has NO boolean — relevance-only. Shown for contrast; app sends plain keywords.
    n = requests.get("https://api.crossref.org/works",
                     params={"query": _plain_q, "rows": 0, "mailto": "literev@gesica.ch"},
                     headers=UA, timeout=TIMEOUT).json().get("message", {}).get("total-results")
    line("Crossref", "plain (n/a)", True, n, _plain_q)


SOURCES = [c_pubmed, c_europepmc, c_preprints, c_openalex, c_doaj, c_clinicaltrials,
           c_core, c_arxiv, c_semantic_scholar, c_openaire, c_crossref]

if __name__ == "__main__":
    print("=" * 96)
    print("SAME boolean → every source (each in the dialect the APP sends). Counts are per-source totals.")
    print("=" * 96)
    print(f"input query   : {QUERY[:88]}")
    print(f"general bool  : {_boolean[:88]}")
    print(f"portable bool : {_portable_bool[:88]}   (real booléen: {_bool_is_real}, sent: {_send_bool})")
    print(f"arxiv dialect : {_arxiv_q[:88]}   (native: {_arxiv_native})")
    print(f"s2 dialect    : {(_s2_bool or '(falls back to plain)')[:88]}")
    print(f"plain keywords: {_plain_q[:88]}   (Crossref + fallback only)")
    print("-" * 96)
    print(f"  {'source':21s} {'sent-as':14s} {'total':>9}   query")
    print("-" * 96)
    for fn in SOURCES:
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {fn.__name__[2:]:20s} {'':14s} {'?':>9}   {type(e).__name__}: {str(e)[:44]}")
    print("-" * 96)
    print("READ: 'portable-bool' = the SAME boolean (PubMed field-tags stripped) sent verbatim.")
    print("PubMed gets the MeSH-tagged dialect; arXiv/S2 get their own syntax; EPMC gets the general")
    print("boolean; only Crossref (relevance-only) gets plain keywords. The corpus is the UNION of")
    print("these boolean hits — so it should be close to the SUM of the boolean-capable rows, not the")
    print("tiny local re-match that produced '11 at the end' before the fix.")
