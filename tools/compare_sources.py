#!/usr/bin/env python3
"""Per-source comparison: DIRECT API results vs what LiteRev's corpus filter KEEPS.

    cd /opt/literev-api && .venv/bin/python tools/compare_sources.py

(Use the app venv — this imports `main`, so it needs the service's dependencies.)

For a natural query it translates exactly like the app (main._generate_search_strategy),
then per source fetches a sample and applies the SAME local boolean match that defines
corpus membership (substring match of the general boolean against title+abstract). The
gap between "direct total" and "passes local boolean" is why the corpus is far smaller
than the raw source totals — and it's biggest for PubMed (MeSH matches with no literal
phrase in the abstract). No DB writes.
"""
import os
import sys
import xml.etree.ElementTree as ET

import requests

# Run as `.venv/bin/python tools/compare_sources.py`: sys.path[0] is tools/, so add the
# app root (parent dir) so `import main` resolves to /opt/literev-api/main.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # reuse the app's translator + boolean parser (same egress as the service)

NL_QUERY = ("Est-ce qu'il existe des critères pour établir un ranking des écoles "
            "en santé publique dans le monde?")
UA = {"User-Agent": "LiteRev-compare/1.0 (mailto:literev@gesica.ch)"}
TIMEOUT = 30
SAMPLE = 200   # docs to pull per source to measure the local-boolean pass rate

# ── translate exactly like the app ───────────────────────────────────────────
_strat = main._generate_search_strategy(NL_QUERY)
GENERAL = (_strat.get("general") or NL_QUERY) if isinstance(_strat, dict) else NL_QUERY
PUBMED_Q = (_strat.get("pubmed") or GENERAL) if isinstance(_strat, dict) else GENERAL
PLAIN = main._plain_keywords(GENERAL) or main._plain_keywords(NL_QUERY) or NL_QUERY
_AST = main._parse_boolean_ast(main._tokenize_boolean(GENERAL))


def local_pass(title, abstract):
    """Replicates _search_local_doc_ids boolean mode: substring match of the general
    boolean against lowercased title+abstract — the exact corpus-membership test."""
    blob = f"{title or ''} {abstract or ''}".lower()

    def m(a):
        if a is None:
            return True
        t = a[0]
        if t == "term":
            return a[1] in blob
        if t == "not":
            return not m(a[1])
        if t == "and":
            return all(m(c) for c in a[1])
        if t == "or":
            return any(m(c) for c in a[1])
        return True
    return m(_AST)


def _oa_abstract(inv):
    if not inv:
        return ""
    words = {}
    for w, pos in inv.items():
        for p in pos:
            words[p] = w
    return " ".join(words[i] for i in sorted(words))


# ── per-source: (direct_total, [(title, abstract), ...]) ─────────────────────
def s_pubmed():
    b = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    p = {"db": "pubmed", "term": PUBMED_Q, "retmode": "json", "retmax": 0}
    if os.getenv("NCBI_API_KEY"):
        p["api_key"] = os.getenv("NCBI_API_KEY")
    total = int(requests.get(f"{b}/esearch.fcgi", params=p, headers=UA, timeout=TIMEOUT)
                .json()["esearchresult"]["count"])
    ep = {"db": "pubmed", "term": PUBMED_Q, "retmode": "json", "retmax": SAMPLE, "sort": "pub_date"}
    if os.getenv("NCBI_API_KEY"):
        ep["api_key"] = os.getenv("NCBI_API_KEY")
    ids = requests.get(f"{b}/esearch.fcgi", params=ep, headers=UA, timeout=TIMEOUT).json()["esearchresult"]["idlist"]
    docs = []
    if ids:
        fp = {"db": "pubmed", "id": ",".join(ids), "rettype": "xml", "retmode": "xml"}
        if os.getenv("NCBI_API_KEY"):
            fp["api_key"] = os.getenv("NCBI_API_KEY")
        root = ET.fromstring(requests.post(f"{b}/efetch.fcgi", data=fp, timeout=90).content)
        for a in root.findall(".//PubmedArticle"):
            te = a.find(".//ArticleTitle")
            title = "".join(te.itertext()).strip() if te is not None else ""
            ab = " ".join("".join(n.itertext()).strip() for n in a.findall(".//Abstract/AbstractText"))
            docs.append((title, ab))
    return total, docs


def s_openalex():
    r = requests.get("https://api.openalex.org/works",
                     params={"search": PLAIN, "per_page": min(SAMPLE, 200), "mailto": "literev@gesica.ch"},
                     headers=UA, timeout=TIMEOUT).json()
    total = r.get("meta", {}).get("count", 0)
    docs = [(w.get("title") or "", _oa_abstract(w.get("abstract_inverted_index"))) for w in r.get("results", [])]
    return total, docs


def s_crossref():
    r = requests.get("https://api.crossref.org/works",
                     params={"query": PLAIN, "rows": 100, "mailto": "literev@gesica.ch"},
                     headers=UA, timeout=TIMEOUT).json().get("message", {})
    total = r.get("total-results", 0)
    docs = [((i.get("title") or [""])[0], i.get("abstract") or "") for i in r.get("items", [])]
    return total, docs


def s_europepmc():
    r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     params={"query": GENERAL, "format": "json", "pageSize": 100, "resultType": "core"},
                     headers=UA, timeout=TIMEOUT).json()
    total = r.get("hitCount", 0)
    docs = [(x.get("title") or "", x.get("abstractText") or "")
            for x in r.get("resultList", {}).get("result", [])]
    return total, docs


def s_semantic_scholar():
    h = {**UA}
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        h["x-api-key"] = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                     params={"query": PLAIN, "limit": 100, "fields": "title,abstract"},
                     headers=h, timeout=TIMEOUT).json()
    total = r.get("total", 0)
    docs = [(d.get("title") or "", d.get("abstract") or "") for d in r.get("data", [])]
    return total, docs


def s_biorxiv_medrxiv():
    # bioRxiv + medRxiv are STILL live (native, source="biorxiv"/"medrxiv"), separate from
    # the Europe PMC "preprint" facet. They have NO keyword search — the API only serves a
    # date window — so "total" here is ALL preprints in the last 45 days, not query hits;
    # the fetcher keeps those matching >= 1/3 of the terms, then the corpus re-match applies.
    from datetime import date, timedelta
    to = date.today()
    win = f"{(to - timedelta(days=45)).isoformat()}/{to.isoformat()}"
    total, docs = 0, []
    for server in ("biorxiv", "medrxiv"):
        r = requests.get(f"https://api.biorxiv.org/details/{server}/{win}/0/json",
                         headers=UA, timeout=TIMEOUT).json()
        total += int((r.get("messages") or [{}])[0].get("total", 0) or 0)
        for it in (r.get("collection") or [])[:100]:
            docs.append((it.get("title") or "", it.get("abstract") or ""))
    return total, docs


SOURCES = [
    ("PubMed", "pubmed boolean", s_pubmed),
    ("Europe PMC", "general boolean", s_europepmc),
    ("OpenAlex", "plain keywords", s_openalex),
    ("Crossref", "plain keywords", s_crossref),
    ("Semantic Scholar", "plain keywords", s_semantic_scholar),
    ("bioRxiv+medRxiv", "45d window scan", s_biorxiv_medrxiv),   # native, NOT the EPMC facet
]

if __name__ == "__main__":
    print("=" * 92)
    print("DIRECT vs LiteRev corpus filter — per source")
    print(f"NL query : {NL_QUERY}")
    print(f"general  : {GENERAL[:110]}")
    print(f"plain    : {PLAIN}")
    print("=" * 92)
    print(f"{'Source':18} {'query':16} {'direct':>8} {'sampled':>8} {'pass local-bool':>16} {'pass%':>7}")
    print("-" * 92)
    for name, qkind, fn in SOURCES:
        try:
            total, docs = fn()
            passed = sum(1 for t, a in docs if local_pass(t, a))
            rate = f"{100*passed/len(docs):.0f}%" if docs else "—"
            print(f"{name:18} {qkind:16} {total:>8} {len(docs):>8} {passed:>16} {rate:>7}")
        except Exception as e:
            print(f"{name:18} {qkind:16} {'FAIL':>8}  {type(e).__name__}: {str(e)[:40]}")
    print("-" * 92)
    print("READ: 'direct' = the source's own total for the query. 'pass local-bool' = how many")
    print("of the SAMPLED docs survive the OLD local boolean re-match (title+abstract substring).")
    print("The gap is the discrepancy; for PubMed it's papers matched via MeSH with no literal")
    print("phrase in the abstract.")
    print("AFTER the source-union fix: PubMed / Europe PMC / EPMC-preprints bypass that re-match")
    print("(their whole 'direct' set enters the corpus, minus no-abstract records); the KEYWORD")
    print("APIs (OpenAlex/Crossref/S2/…) still keep only 'pass local-bool' — their plain-query")
    print("results are a loose ranking that the boolean SHOULD filter.")
