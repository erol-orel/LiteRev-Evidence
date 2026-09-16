"""Live literature sources: fetchers, parsers, federated search, direct ingestion.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import os
import re
import threading as _threading_ncbi
from typing import Any

from fastapi import Depends, Query
from sqlalchemy import text

from .core import RELIEFWEB_APPNAME, app, engine, logger, require_api_key
from .documents import _normalize_doi, _normalize_title, sanitize_db_text
from .scenario_store import _get_scenario_threshold, _get_user_scenario_or_404
from .search import _plain_keywords

_NCBI_LOCK = _threading_ncbi.Lock()
_NCBI_LAST = [0.0]
_NCBI_MIN_INTERVAL = 0.4  # ~2.5 req/s, sous la limite de 3/s


def _ncbi_get(url: str, params: dict, timeout: int = 12):
    """GET eutils throttlé (verrou global) avec un petit retry sur 429/erreur."""
    import requests as _req
    import time as _time
    key = os.getenv("NCBI_API_KEY")
    if key:
        params = {**params, "api_key": key}
    # Avec une clé API, NCBI autorise 10 req/s (vs 3 sans) : on resserre l'espacement
    # pour réduire la sérialisation du verrou global sur le trio PubMed/PROSPERO/Cochrane.
    min_interval = 0.11 if key else _NCBI_MIN_INTERVAL
    r = None
    last_exc: Exception | None = None
    for attempt in range(3):
        with _NCBI_LOCK:
            wait = min_interval - (_time.time() - _NCBI_LAST[0])
            if wait > 0:
                _time.sleep(wait)
            try:
                r = _req.get(url, params=params, timeout=timeout)
            except Exception as _e:  # timeout / ConnectionError : transitoire → retry (cf. docstring)
                last_exc = _e
                r = None
            finally:
                _NCBI_LAST[0] = _time.time()
        if r is None:
            _time.sleep(0.6 * (attempt + 1))
            continue
        if r.status_code == 429:
            _time.sleep(0.6 * (attempt + 1))
            continue
        return r
    if r is None:
        raise last_exc if last_exc else RuntimeError("NCBI request a échoué (3 tentatives)")
    return r


def _live_fetch_pubmed(query: str, max_results: int) -> tuple[list[dict], int]:
    """Fetch from PubMed eSearch+eSummary.

    Renvoie (résultats, hitcount_réel). `hitcount_réel` = esearchresult.count = le
    NOMBRE VRAI d'enregistrements PubMed correspondant à la requête - le même
    compteur que sur pubmed.ncbi.nlm.nih.gov - qui peut dépasser de loin les
    `max_results` réellement rapatriés dans le panneau. Surfacé pour que le badge
    de source affiche le vrai total trouvé, et non les seules lignes récupérées
    (cause du « 306 sur PubMed mais 35 ici »)."""
    results: list[dict] = []
    total = 0
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        r = _ncbi_get(f"{base}/esearch.fcgi", {
            "db": "pubmed", "term": query, "retmax": max_results,
            "retmode": "json", "tool": "literev", "email": "api@literev.app"
        })
        _esr = r.json().get("esearchresult", {})
        total = int(_esr.get("count", 0) or 0)          # vrai hitcount (comme le site)
        ids = _esr.get("idlist", [])
        if not ids:
            return [], total
        r2 = _ncbi_get(f"{base}/esummary.fcgi", {
            "db": "pubmed", "id": ",".join(ids), "retmode": "json",
            "tool": "literev", "email": "api@literev.app"
        })
        res2 = r2.json().get("result", {})
        for uid in res2.get("uids", []):
            item = res2.get(uid, {})
            results.append({
                "title": item.get("title", ""),
                "abstract": None,
                "doi": next((a["value"] for a in item.get("articleids", []) if a.get("idtype") == "doi"), None),
                "year": int(item.get("pubdate", "")[:4]) if item.get("pubdate", "")[:4].isdigit() else None,
                "authors": [a.get("name", "") for a in item.get("authors", [])],
                "journal": item.get("source", None),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                "external_id": f"pmid:{uid}",
                "source_name": "PubMed",
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_pubmed error: {_e}")
    return results, total


def _live_fetch_openalex(query: str, max_results: int) -> list[dict]:
    import requests as _req
    results = []
    try:
        # OpenAlex `search` n'accepte pas la syntaxe booléenne (renvoie HTTP 400)
        # → on lui passe des mots-clés simples.
        r = _req.get("https://api.openalex.org/works", params={
            "search": _plain_keywords(query), "per-page": min(max_results, 50),
            "select": "id,title,abstract_inverted_index,doi,publication_year,authorships,primary_location,open_access"
        }, headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
        for item in r.json().get("results", []):
            doi = item.get("doi", "")
            if doi and doi.startswith("https://doi.org/"):
                doi = doi[len("https://doi.org/"):]
            loc = item.get("primary_location") or {}
            source = loc.get("source") or {}
            results.append({
                "title": item.get("title", ""),
                "abstract": None,
                "doi": doi or None,
                "year": item.get("publication_year"),
                "authors": [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])[:5]],
                "journal": source.get("display_name"),
                "url": item.get("id"),
                "external_id": item.get("id"),
                "source_name": "OpenAlex",
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_openalex error: {_e}")
    return results


def _live_fetch_crossref(query: str, max_results: int) -> list[dict]:
    import requests as _req
    results = []
    try:
        r = _req.get("https://api.crossref.org/works", params={
            "query": query, "rows": min(max_results, 50),
            "select": "DOI,title,abstract,published,author,container-title"
        }, headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
        for item in r.json().get("message", {}).get("items", []):
            pub = item.get("published", {}).get("date-parts", [[None]])[0]
            year = pub[0] if pub else None
            results.append({
                "title": (item.get("title") or [""])[0],
                "abstract": item.get("abstract"),
                "doi": item.get("DOI"),
                "year": year,
                "authors": [f"{a.get('family', '')} {a.get('given', '')}".strip() for a in item.get("author", [])[:5]],
                "journal": (item.get("container-title") or [None])[0],
                "url": f"https://doi.org/{item.get('DOI')}" if item.get("DOI") else None,
                "external_id": item.get("DOI"),
                "source_name": "Crossref",
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_crossref error: {_e}")
    return results


def _live_fetch_europepmc(query: str, max_results: int) -> list[dict]:
    import requests as _req
    results = []
    try:
        # NB : ne PAS passer sort=RELEVANCE - c'est une valeur invalide pour
        # EuropePMC qui renvoie alors une liste vide. Sans 'sort', l'API trie
        # par pertinence par défaut.
        r = _req.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={
            "query": query, "resultType": "lite", "pageSize": min(max_results, 50),
            "format": "json"
        }, headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
        for item in r.json().get("resultList", {}).get("result", []):
            results.append({
                "title": item.get("title", ""),
                "abstract": item.get("abstractText"),
                "doi": item.get("doi"),
                "year": int(item["pubYear"]) if item.get("pubYear", "").isdigit() else None,
                "authors": item.get("authorString", "").split(", ")[:5] if item.get("authorString") else [],
                "journal": item.get("journalTitle"),
                "url": f"https://europepmc.org/article/{item.get('source','')}/{item.get('id','')}",
                "external_id": item.get("id"),
                "source_name": "EuropePMC",
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_europepmc error: {_e}")
    return results


def _live_fetch_preprints(query: str, max_results: int) -> list[dict]:
    """Préprints (bioRxiv, medRxiv, Research Square, …) via Europe PMC (filtre SRC:PPR).
    Europe PMC indexe les préprints AVEC recherche plein-texte par mots-clés - au
    contraire de l'API biorxiv (dates/DOI uniquement) qu'utilisaient les anciens
    scanners medRxiv/bioRxiv (peu/pas de résultats)."""
    import requests as _req
    results = []
    try:
        r = _req.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={
            "query": f"({query}) AND (SRC:PPR)", "resultType": "lite",
            "pageSize": min(max_results, 50), "format": "json",
        }, headers={"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}, timeout=10)
        for item in r.json().get("resultList", {}).get("result", []):
            results.append({
                "title": item.get("title", ""),
                "abstract": item.get("abstractText"),
                "doi": item.get("doi"),
                "year": int(item["pubYear"]) if item.get("pubYear", "").isdigit() else None,
                "authors": item.get("authorString", "").split(", ")[:5] if item.get("authorString") else [],
                "journal": item.get("journalTitle") or "Preprint",
                "url": f"https://europepmc.org/article/{item.get('source','PPR')}/{item.get('id','')}",
                "external_id": item.get("id"),
                "source_name": "Preprints",
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_preprints error: {_e}")
    return results


def _live_fetch_pubmed_term(term: str, source_name: str, id_prefix: str, max_results: int) -> list[dict]:
    """Helper PubMed générique (esearch+esummary) avec un terme/filtre arbitraire.
    Sert de proxy pour les sources sans API libre (PROSPERO, Cochrane)."""
    results = []
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        r = _ncbi_get(f"{base}/esearch.fcgi", {
            "db": "pubmed", "term": term, "retmax": max_results,
            "retmode": "json", "tool": "literev", "email": "api@literev.app"
        })
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        r2 = _ncbi_get(f"{base}/esummary.fcgi", {
            "db": "pubmed", "id": ",".join(ids), "retmode": "json",
            "tool": "literev", "email": "api@literev.app"
        })
        res = r2.json().get("result", {})
        for uid in res.get("uids", []):
            item = res.get(uid, {})
            results.append({
                "title": item.get("title", ""),
                "abstract": None,
                "doi": next((a["value"] for a in item.get("articleids", []) if a.get("idtype") == "doi"), None),
                "year": int(item.get("pubdate", "")[:4]) if item.get("pubdate", "")[:4].isdigit() else None,
                "authors": [a.get("name", "") for a in item.get("authors", [])],
                "journal": item.get("source", None),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                "external_id": f"{id_prefix}:{uid}",
                "source_name": source_name,
            })
    except Exception as _e:
        logger.warning(f"_live_fetch_pubmed_term({source_name}) error: {_e}")
    return results


def _live_fetch_preprint_server(server: str, source_name: str, query: str, max_results: int) -> list[dict]:
    """Récupère les prépublications récentes (biorxiv/medrxiv API) puis filtre par
    correspondance de mots-clés de la requête (pas d'API plein-texte côté serveur)."""
    import requests as _req
    import datetime as _dt
    results = []
    try:
        # Mots-clés significatifs (booléen nettoyé). Les 2 premiers sont les
        # termes "primaires" (concept central) : on EXIGE qu'au moins un soit
        # présent, plus un nombre minimal de correspondances totales - sinon le
        # filtre laisse passer n'importe quel preprint contenant 2 mots courants.
        words = _plain_keywords(query, max_words=12).split()
        primary = words[:2]
        min_hits = min(3, len(words)) if len(words) >= 3 else 1
        date_to = _dt.date.today()
        date_from = date_to - _dt.timedelta(days=180)
        cursor = 0
        scanned = 0
        # Plafond resserré : l'API biorxiv ne fait pas de recherche plein-texte, on
        # filtre côté client par mots-clés → le taux de correspondance est faible et
        # scanner 300 prépublications (3 pages × 10s) consommait quasi tout le budget
        # de 30s de la fédération pour ~0 résultat. 120 + timeout court suffit.
        max_scan = 120  # plafond pour rester dans le budget temps de la fédération
        _hdrs = {"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}
        while scanned < max_scan and len(results) < max_results:
            url = (f"https://api.biorxiv.org/details/{server}/"
                   f"{date_from.isoformat()}/{date_to.isoformat()}/{cursor}/json")
            r = _req.get(url, timeout=6, headers=_hdrs)
            if not r.ok:
                break
            payload = r.json()
            coll = payload.get("collection", []) or []
            if not coll:
                break
            for item in coll:
                scanned += 1
                hay = (item.get("title", "") + " " + item.get("abstract", "")).lower()
                if words:
                    hits = sum(1 for w in words if w in hay)
                    has_primary = any(p in hay for p in primary) if primary else True
                    if not (has_primary and hits >= min_hits):
                        continue
                yr = None
                d = item.get("date", "")
                if len(d) >= 4 and d[:4].isdigit():
                    yr = int(d[:4])
                doi = item.get("doi")
                results.append({
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract"),
                    "doi": doi,
                    "year": yr,
                    "authors": [a.strip() for a in (item.get("authors", "") or "").split(";")[:5] if a.strip()],
                    "journal": source_name,
                    "url": f"https://doi.org/{doi}" if doi else None,
                    "external_id": doi,
                    "source_name": source_name,
                })
                if len(results) >= max_results:
                    break
            total = int(payload.get("messages", [{}])[0].get("total", 0) or 0)
            cursor += len(coll)
            if cursor >= total:
                break
    except Exception as _e:
        logger.warning(f"_live_fetch_preprint_server({server}) error: {_e}")
    return results


def _live_fetch_medrxiv(query: str, max_results: int) -> list[dict]:
    return _live_fetch_preprint_server("medrxiv", "medRxiv", query, max_results)


def _live_fetch_biorxiv(query: str, max_results: int) -> list[dict]:
    return _live_fetch_preprint_server("biorxiv", "bioRxiv", query, max_results)


def _federated_live_search(
    query: str,
    max_per_source: int = 50,
    pubmed_query: str | None = None,
    general_query: str | None = None,
) -> tuple[list[dict], list[str], dict[str, int], dict[str, dict]]:
    """Interroge les sources externes en parallèle, déduplique (par DOI puis
    titre normalisé), marque in_local_db, et score chaque résultat
    (sémantique cosinus + lexical + hybride). Réutilisé par /search (fédéré)
    et par la recherche live des scénarios.

    Retourne (results triés par hybrid_score desc, sources_queried,
    raw_counts par source avant déduplication)."""
    import concurrent.futures
    pubmed_query = pubmed_query or query
    general_query = general_query or query
    # PubMed RECALL : la requête MeSH générée (pubmed_query) est souvent BEAUCOUP plus
    # étroite que le booléen portable (general_query) - p.ex. 35 vs 306 pour le même
    # booléen collé sur le site PubMed. On interroge PubMed avec l'UNION des deux (même
    # correctif que le populate) pour retrouver le rappel du site. Repli : booléen seul.
    if (pubmed_query and general_query and pubmed_query != general_query
            and len(pubmed_query) + len(general_query) <= 1900):
        _pubmed_q = f"({pubmed_query}) OR ({general_query})"
    else:
        _pubmed_q = general_query or pubmed_query

    source_fns = [
        ("PubMed", _live_fetch_pubmed, _pubmed_q),
        ("OpenAlex", _live_fetch_openalex, general_query),
        ("Crossref", _live_fetch_crossref, general_query),
        ("EuropePMC", _live_fetch_europepmc, general_query),
        # Préprints (bioRxiv/medRxiv/…) via Europe PMC (SRC:PPR) : recherche par
        # mots-clés réelle. Remplace les scanners medRxiv/bioRxiv (dates seules) et
        # les proxys PubMed PROSPERO/Cochrane (trompeurs et redondants avec PubMed).
        ("Preprints", _live_fetch_preprints, general_query),
    ]

    import time as _t_fed
    _t0_fed = _t_fed.time()
    all_results: list[dict] = []
    sources_queried: list[str] = []
    raw_counts: dict[str, int] = {name: 0 for name, _, _ in source_fns}
    # Statut par source pour le diagnostic ("not working / slow") : ok / empty /
    # error / timeout, + latence. Les sources non complétées dans le délai
    # restent "timeout" (auparavant silencieusement absentes de la réponse).
    source_status: dict[str, dict[str, Any]] = {
        name: {"status": "timeout", "count": 0, "latency_ms": None} for name, _, _ in source_fns
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fn, q, max_per_source): name for name, fn, q in source_fns}
        try:
            for future in concurrent.futures.as_completed(futures, timeout=30):
                name = futures[future]
                sources_queried.append(name)
                _ms = round((_t_fed.time() - _t0_fed) * 1000)
                try:
                    _res = future.result()
                    # Un fetcher renvoie soit une liste, soit (liste, total_trouvé_réel).
                    # PubMed fournit son vrai hitcount (esearchresult.count), qui peut
                    # dépasser les lignes rapatriées → le badge affiche le vrai total.
                    if isinstance(_res, tuple):
                        items, _found = _res
                    else:
                        items, _found = _res, None
                    _found = _found if (_found is not None and _found >= len(items)) else len(items)
                    raw_counts[name] = _found
                    all_results.extend(items)
                    source_status[name] = {
                        "status": "ok" if items else "empty",
                        "count": _found, "fetched": len(items), "latency_ms": _ms,
                    }
                except Exception as _fe:
                    logger.warning(f"federated source {name} error: {_fe}")
                    source_status[name] = {
                        "status": "error", "count": 0, "latency_ms": _ms,
                        "error": str(_fe)[:200],
                    }
        except concurrent.futures.TimeoutError:
            logger.warning("federated search: certaines sources ont dépassé le délai")

    # Marquage in_local_db - literature_document n'a pas de colonne doi dédiée et
    # external_id est hétérogène selon la source (DOI brut pour Crossref/EuropePMC,
    # "pmid:<id>" pour PubMed/PROSPERO/Cochrane, URL pour OpenAlex). On compare donc
    # l'external_id stocké à la fois aux DOIs ET aux external_id des résultats
    # (auparavant : DOI seul → PubMed/PROSPERO/Cochrane/OpenAlex jamais reconnus).
    keys: set[str] = set()
    for r in all_results:
        if r.get("doi"):
            keys.add(r["doi"].lower())
        if r.get("external_id"):
            keys.add(str(r["external_id"]).lower())
    in_db_keys: set[str] = set()
    if keys:
        try:
            with engine.connect() as conn:
                # Compare les clés (DOI + external_id des résultats) à la fois à la
                # colonne external_id ET à la colonne doi stockées : un article déjà
                # ingéré depuis PubMed (external_id='pmid:…', doi renseigné) doit être
                # reconnu par le DOI d'un résultat Crossref/OpenAlex (sinon new_count
                # surestimé → ré-ingestion redondante).
                rows_db = conn.execute(text(
                    "SELECT LOWER(external_id), LOWER(doi) FROM literature_document "
                    "WHERE (LOWER(external_id) = ANY(:keys) OR LOWER(doi) = ANY(:keys)) "
                    "AND project_context = 'literev'"
                ), {"keys": list(keys)}).fetchall()
                for _r in rows_db:
                    if _r[0]:
                        in_db_keys.add(_r[0])
                    if _r[1]:
                        in_db_keys.add(_r[1])
        except Exception as _dbe:
            logger.warning(f"federated DB check error: {_dbe}")
    for r in all_results:
        _doi = (r.get("doi") or "").lower()
        _eid = str(r.get("external_id") or "").lower()
        r["in_local_db"] = bool((_doi and _doi in in_db_keys) or (_eid and _eid in in_db_keys))

    # Déduplication par DOI (sinon titre normalisé), suivi des sources
    import re as _re2
    def _norm_title(t: str) -> str:
        return _re2.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()
    deduped: dict[str, dict] = {}
    for r in all_results:
        key = ("doi:" + r["doi"].lower()) if r.get("doi") else ("ttl:" + _norm_title(r.get("title", "")))
        if not key or key in ("ttl:", "doi:"):
            key = "id:" + str(id(r))
        if key in deduped:
            existing = deduped[key]
            srcs = existing.setdefault("also_in_sources", [])
            if r.get("source_name") and r["source_name"] not in srcs and r["source_name"] != existing.get("source_name"):
                srcs.append(r["source_name"])
            for f in ("abstract", "year", "url", "doi"):
                if not existing.get(f) and r.get(f):
                    existing[f] = r[f]
            existing["in_local_db"] = existing.get("in_local_db") or r.get("in_local_db")
        else:
            r.setdefault("also_in_sources", [])
            deduped[key] = r
    deduped_list = list(deduped.values())

    # Scoring sémantique + lexical + hybride
    def _lexical_overlap(q_words: set[str], text_blob: str) -> float:
        if not q_words:
            return 0.0
        hay = set(_re2.findall(r"[a-z0-9]{3,}", (text_blob or "").lower()))
        if not hay:
            return 0.0
        return min(1.0, len(q_words & hay) / max(1, len(q_words)))

    q_words = set(_re2.findall(r"[a-z0-9]{3,}", (query or "").lower()))
    openai_key = os.getenv("OPENAI_API_KEY")
    q_emb = None
    res_embs: list[list[float] | None] = [None] * len(deduped_list)
    if openai_key and deduped_list:
        # Borne anti-latence : le scoring sémantique est sur le chemin de la
        # requête (l'utilisateur attend la réponse). On ne ré-embedde donc QUE les
        # N meilleurs résultats par recouvrement lexical (retrieve-then-rerank) ;
        # au-delà, semantic_score reste 0 (ces résultats sont déjà peu pertinents).
        # Évite d'embedder ~400 résultats par requête. Timeout court + garde par
        # lot pour qu'un appel lent/échoué ne fige pas ni n'annule tout le scoring.
        SEM_SCORE_CAP = 60
        cand_idx = sorted(
            range(len(deduped_list)),
            key=lambda i: _lexical_overlap(
                q_words,
                (deduped_list[i].get("title", "") or "") + " " + (deduped_list[i].get("abstract") or "")),
            reverse=True,
        )[:SEM_SCORE_CAP]
        try:
            from llm_usage import MeteredOpenAI as _OAI
            _client = _OAI(api_key=openai_key, timeout=8.0)
            q_emb = _client.embeddings.create(
                input=[(query or "").replace("\n", " ").strip()],
                model="text-embedding-3-small",
            ).data[0].embedding
            texts = [((deduped_list[i].get("title", "") or "") + ". " + (deduped_list[i].get("abstract") or "")).replace("\n", " ").strip()[:2000]
                     for i in cand_idx]
            for b in range(0, len(texts), 256):
                try:
                    emb_resp = _client.embeddings.create(input=texts[b:b + 256], model="text-embedding-3-small")
                    for j, d in enumerate(emb_resp.data):
                        res_embs[cand_idx[b + j]] = d.embedding
                except Exception as _be:
                    logger.warning(f"federated scoring batch error: {_be}")
        except Exception as _ee:
            logger.warning(f"federated scoring embed error: {_ee}")
            q_emb = None

    def _cosine(a, b) -> float:
        if not a or not b:
            return 0.0
        import math as _m
        dot = sum(x * y for x, y in zip(a, b))
        na = _m.sqrt(sum(x * x for x in a)); nb = _m.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    for i, r in enumerate(deduped_list):
        blob = (r.get("title", "") or "") + " " + (r.get("abstract") or "")
        r["lexical_score"] = round(_lexical_overlap(q_words, blob), 4)
        sem = max(0.0, _cosine(q_emb, res_embs[i])) if (q_emb and res_embs[i]) else 0.0
        r["semantic_score"] = round(sem, 4)

    # ④ Fusion HYBRIDE par Reciprocal Rank Fusion (RRF) au lieu d'une somme pondérée :
    # on classe INDÉPENDAMMENT par lexical puis par sémantique et on somme 1/(k+rang).
    # Sans paramètre à régler, et robuste au fait que seuls les SEM_SCORE_CAP premiers
    # résultats ont un score sémantique (les autres, sem=0, ne comptent que par le
    # lexical) - une somme pondérée mélangeait ces échelles hétérogènes. k=60 (usuel).
    # Un résultat n'obtient de crédit d'un signal que si ce signal est > 0 (sinon tous
    # les ex-æquo à 0 en bas du classement pollueraient le score).
    _RRF_K = 60
    _rrf = [0.0] * len(deduped_list)
    _order_lex = sorted(range(len(deduped_list)),
                        key=lambda i: deduped_list[i]["lexical_score"], reverse=True)
    for _rank, _i in enumerate(_order_lex):
        if deduped_list[_i]["lexical_score"] > 0:
            _rrf[_i] += 1.0 / (_RRF_K + _rank + 1)
    _order_sem = sorted(range(len(deduped_list)),
                        key=lambda i: deduped_list[i]["semantic_score"], reverse=True)
    for _rank, _i in enumerate(_order_sem):
        if deduped_list[_i]["semantic_score"] > 0:
            _rrf[_i] += 1.0 / (_RRF_K + _rank + 1)
    # Normalisation en [0,1] pour l'affichage (le front affiche hybrid_score.toFixed(2)
    # comme badge) : on divise par le meilleur score RRF → 1er résultat ≈ 1.00, l'ORDRE
    # reste identique au RRF brut.
    _rrf_max = (max(_rrf) if _rrf else 0.0) or 1.0
    for _i, r in enumerate(deduped_list):
        r["hybrid_score"] = round(_rrf[_i] / _rrf_max, 4)

    deduped_list.sort(key=lambda r: r.get("hybrid_score", 0.0), reverse=True)
    return deduped_list, sources_queried, raw_counts, source_status


@app.post("/user-scenarios/{scenario_id}/search/live")
def search_live(
    scenario_id: str,
    max_per_source: int = 50,
    lang: str | None = Query(None),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Live federated search across the external sources in parallel.

    `lang` : langue de l'interface qui lance la recherche. Cinquième point d'entrée qui
    construit un corpus, il l'ignorait : la suite (résumés de clusters, brief, variables,
    actions) était donc écrite en français quel que soit le toggle, et un redémarrage
    reprenait le scénario en français puisque `pipeline_lang` avait été fixé à 'fr'."""
    from .scenarios import _launch_populate_job  # lazy: scenarios is loaded after this module
    row = _get_user_scenario_or_404(scenario_id)
    query = row["query"]
    strategy = row.get("search_strategy") or {}
    pubmed_query = strategy.get("pubmed", query) if isinstance(strategy, dict) else query
    general_query = strategy.get("general", query) if isinstance(strategy, dict) else query

    all_results, sources_queried, raw_counts, source_status = _federated_live_search(
        query, max_per_source, pubmed_query=pubmed_query, general_query=general_query
    )
    new_count = sum(1 for r in all_results if not r["in_local_db"])

    # Compteur RÉEL du corpus du scénario (identique à l'onglet Corpus) pour que
    # le panneau « recherche en direct » soit cohérent avec le corpus. Le bloc
    # fédéré ci-dessus n'interroge que les APIs externes (plafonné) ; il ne
    # reflète PAS la correspondance locale réelle.
    _thr = _get_scenario_threshold(scenario_id)
    corpus_total = 0
    corpus_above = 0
    try:
        with engine.connect() as _cc:
            _cr = _cc.execute(text("""
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                           AND (COALESCE(ars.screening_status, d.screening_status) = 'included'
                                OR COALESCE(ars.similarity_score, 0) >= :thr)) AS above
                FROM article_scenarios ars
                JOIN literature_document d ON d.id = ars.document_id
                WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            """), {"sid": scenario_id, "thr": _thr}).mappings().first()
        corpus_total = int(_cr["total"] or 0)
        corpus_above = int(_cr["above"] or 0)
    except Exception as _ce:
        logger.warning(f"search_live corpus count {scenario_id}: {_ce}")

    # Background ingest of new papers - via le lanceur verrouillé pour ne jamais
    # démarrer un populate concurrent (sinon les nettoyages post-ingestion se
    # marchent dessus : compteurs corrompus, liens supprimés par l'autre job).
    ingesting_background = False
    if new_count > 0:
        try:
            status = _launch_populate_job(scenario_id, query, row.get("filters") or {}, 200, lang=lang)
            ingesting_background = (status == "started")
        except Exception as _be:
            logger.warning(f"search_live background ingest error: {_be}")

    return {
        "results": all_results,
        "total": len(all_results),
        "new_count": new_count,
        "corpus_total": corpus_total,
        "corpus_above_threshold": corpus_above,
        "threshold": _thr,
        "sources_queried": sources_queried,
        "source_raw_counts": raw_counts,
        "source_status": source_status,
        "ingesting_background": ingesting_background,
    }


@app.get("/sources/health")
def sources_health(query: str = "cardiac arrest", timeout: int = 12) -> dict[str, Any]:
    """Diagnostic des sources externes (live search).

    Interroge en parallèle les SIX sources sondées (PubMed, OpenAlex, Crossref,
    Europe PMC, bioRxiv/medRxiv, ReliefWeb) avec une requête minimale et renvoie, par
    source, le statut HTTP, la latence (ms), un compteur de résultats et l'erreur
    éventuelle. Permet de diagnostiquer « sources lentes / ne répondent plus »
    directement en production (où l'accès réseau sortant diffère du sandbox).

    ATTENTION : les autres fetchers de la fédération (Semantic Scholar, DOAJ,
    ClinicalTrials.gov, CORE, arXiv, OpenAIRE) ne sont PAS sondés. « 6 joignables sur 6 »
    ne signifie donc pas que toute la fédération est saine ; la réponse porte
    `probed` et `not_probed` pour que ce soit explicite.
    Lecture seule, aucune écriture, aucune clé requise.
    """
    import concurrent.futures
    import time as _t
    from datetime import datetime as _dtm, timezone as _tz
    import requests as _req

    ua = {"User-Agent": "LiteRev/1.0 (mailto:api@literev.app)"}
    ncbi_key = os.getenv("NCBI_API_KEY")
    eutils_params = {"db": "pubmed", "term": query, "retmax": 1, "retmode": "json",
                     "tool": "literev", "email": "api@literev.app"}
    if ncbi_key:
        eutils_params["api_key"] = ncbi_key

    # (nom, url, params, headers, extracteur de compteur depuis le JSON)
    probes = [
        ("PubMed (eutils)", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
         eutils_params, ua,
         lambda j: len(j.get("esearchresult", {}).get("idlist", []))),
        ("OpenAlex", "https://api.openalex.org/works",
         {"search": _plain_keywords(query), "per-page": 1, "select": "id,title"}, ua,
         lambda j: j.get("meta", {}).get("count")),
        ("Crossref", "https://api.crossref.org/works",
         {"query": query, "rows": 1, "select": "DOI,title"}, ua,
         lambda j: j.get("message", {}).get("total-results")),
        ("EuropePMC", "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
         {"query": query, "resultType": "lite", "pageSize": 1, "format": "json"}, ua,
         lambda j: j.get("hitCount")),
        ("bioRxiv/medRxiv", "https://api.biorxiv.org/details/biorxiv/2024-01-01/2024-01-07/0/json",
         None, ua,
         lambda j: (j.get("messages", [{}])[0] or {}).get("total")),
        # ReliefWeb (flux SÉPARÉ, littérature grise) : sondé ici pour qu'une panne, un
        # appname non approuvé ou un quota épuisé soit VISIBLE dans le diagnostic de
        # production, et pas seulement au moment d'une ingestion.
        ("ReliefWeb (situation reports)", "https://api.reliefweb.int/v2/reports",
         {"appname": RELIEFWEB_APPNAME or "unconfigured", "limit": 0,
          "filter[field]": "disaster_type.name", "filter[value]": "Epidemic"}, ua,
         lambda j: j.get("totalCount")),
    ]

    def _probe(name: str, url: str, params, headers, count_fn) -> dict[str, Any]:
        t0 = _t.time()
        try:
            r = _req.get(url, params=params, headers=headers, timeout=timeout)
            ms = round((_t.time() - t0) * 1000)
            count = None
            if r.status_code == 200:
                try:
                    count = count_fn(r.json())
                except Exception:
                    count = None
            return {"source": name, "ok": r.status_code == 200, "http": r.status_code,
                    "latency_ms": ms, "count": count,
                    "error": None if r.status_code == 200 else (r.text or "")[:200]}
        except Exception as e:
            return {"source": name, "ok": False, "http": None,
                    "latency_ms": round((_t.time() - t0) * 1000), "count": None,
                    "error": f"{type(e).__name__}: {e}"[:200]}

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(probes)) as ex:
        futs = {ex.submit(_probe, *p): p[0] for p in probes}
        try:
            for f in concurrent.futures.as_completed(futs, timeout=timeout + 5):
                results.append(f.result())
        except concurrent.futures.TimeoutError:
            done = {r["source"] for r in results}
            for name in futs.values():
                if name not in done:
                    results.append({"source": name, "ok": False, "http": None,
                                    "latency_ms": None, "count": None,
                                    "error": "probe timed out"})
    results.sort(key=lambda r: r["source"])
    # Les fetchers de la fédération que ce diagnostic NE sonde PAS. Les nommer dans la
    # réponse évite de lire « 6 joignables sur 6 » comme « toute la fédération est saine ».
    not_probed = ["Semantic Scholar", "DOAJ", "ClinicalTrials.gov", "CORE", "arXiv",
                  "OpenAIRE"]
    return {
        "query": query,
        "checked_at": _dtm.now(_tz.utc).isoformat(),
        "sources": results,
        "reachable": sum(1 for r in results if r["ok"]),
        "total": len(results),
        "probed": [r["source"] for r in results],
        "not_probed": not_probed,
        "coverage_note": (
            f"{len(results)} sources sondées sur {len(results) + len(not_probed)} "
            f"fetchers ; les autres ne sont pas testées par cet endpoint."
        ),
        "config": {
            "ncbi_api_key": bool(ncbi_key),
            "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
        },
    }


# ── Parseurs de sources littéraires (PURS / testables) ───────────────────────
# Chaque parseur transforme la réponse brute d'une API en une liste de docs
# {title, abstract, year, url, external_id, doi, source_type}. Le fetcher (closure
# dans le populate) ne fait que le HTTP + la pagination puis délègue ici - de sorte
# que la forme de chaque réponse est vérifiée par des tests SANS réseau.
def _parse_semantic_scholar(payload: dict) -> list[dict]:
    """Semantic Scholar Graph API /paper/search → docs. external_id = s2:<paperId>."""
    out: list[dict] = []
    for it in (payload.get("data") or []) if isinstance(payload, dict) else []:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        pid = it.get("paperId")
        if not title or not pid:
            continue
        ext = it.get("externalIds") or {}
        doi = ext.get("DOI") or ext.get("doi")
        year = it.get("year")
        out.append({
            "title": title, "abstract": (it.get("abstract") or None),
            "year": int(year) if isinstance(year, int) else None,
            "url": it.get("url") or (f"https://doi.org/{doi}" if doi else None),
            "doi": doi, "external_id": f"s2:{pid}", "source_type": "article",
        })
    return out


def _parse_doaj(payload: dict) -> list[dict]:
    """DOAJ /api/search/articles → docs. external_id = doaj:<id>."""
    out: list[dict] = []
    for it in (payload.get("results") or []) if isinstance(payload, dict) else []:
        bj = (it.get("bibjson") or {}) if isinstance(it, dict) else {}
        title = (bj.get("title") or "").strip()
        did = it.get("id") if isinstance(it, dict) else None
        if not title or not did:
            continue
        doi = None
        for ident in (bj.get("identifier") or []):
            if isinstance(ident, dict) and str(ident.get("type", "")).lower() == "doi":
                doi = ident.get("id"); break
        url = None
        for lk in (bj.get("link") or []):
            if isinstance(lk, dict) and lk.get("url"):
                url = lk.get("url"); break
        try:
            year = int(bj.get("year")) if bj.get("year") else None
        except (TypeError, ValueError):
            year = None
        out.append({
            "title": title, "abstract": (bj.get("abstract") or None), "year": year,
            "url": url or (f"https://doi.org/{doi}" if doi else None), "doi": doi,
            "external_id": f"doaj:{did}", "source_type": "article",
        })
    return out


def _parse_clinicaltrials(payload: dict) -> list[dict]:
    """ClinicalTrials.gov API v2 /studies → docs. external_id = nct:<id>."""
    out: list[dict] = []
    for st in (payload.get("studies") or []) if isinstance(payload, dict) else []:
        ps = (st.get("protocolSection") or {}) if isinstance(st, dict) else {}
        idm = ps.get("identificationModule") or {}
        nct = idm.get("nctId")
        title = (idm.get("briefTitle") or idm.get("officialTitle") or "").strip()
        if not nct or not title:
            continue
        desc = ps.get("descriptionModule") or {}
        year = None
        date = ((ps.get("statusModule") or {}).get("startDateStruct") or {}).get("date") or ""
        if len(date) >= 4 and date[:4].isdigit():
            year = int(date[:4])
        out.append({
            "title": title,
            "abstract": (desc.get("briefSummary") or desc.get("detailedDescription") or None),
            "year": year, "url": f"https://clinicaltrials.gov/study/{nct}",
            "doi": None, "external_id": f"nct:{nct}", "source_type": "clinical_trial",
        })
    return out


def _parse_core(payload: dict) -> list[dict]:
    """CORE API v3 /search/works → docs. external_id = core:<id>."""
    out: list[dict] = []
    for it in (payload.get("results") or []) if isinstance(payload, dict) else []:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        cid = it.get("id")
        if not title or cid is None:
            continue
        doi = it.get("doi")
        try:
            year = int(it.get("yearPublished")) if it.get("yearPublished") else None
        except (TypeError, ValueError):
            year = None
        out.append({
            "title": title, "abstract": (it.get("abstract") or None), "year": year,
            "url": it.get("downloadUrl") or (f"https://doi.org/{doi}" if doi else None),
            "doi": doi, "external_id": f"core:{cid}", "source_type": "article",
        })
    return out


def _parse_arxiv(xml_text: str) -> list[dict]:
    """arXiv Atom feed (export.arxiv.org/api/query) → docs. external_id = arxiv:<id>."""
    import xml.etree.ElementTree as _ET
    _ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    out: list[dict] = []
    try:
        root = _ET.fromstring(xml_text)
    except Exception:
        return out
    for e in root.findall("a:entry", _ns):
        title = (e.findtext("a:title", default="", namespaces=_ns) or "").strip()
        aid_url = (e.findtext("a:id", default="", namespaces=_ns) or "").strip()
        if not title or not aid_url:
            continue
        aid = aid_url.rsplit("/abs/", 1)[-1] if "/abs/" in aid_url else aid_url.rsplit("/", 1)[-1]
        summary = (e.findtext("a:summary", default="", namespaces=_ns) or "").strip() or None
        published = e.findtext("a:published", default="", namespaces=_ns) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        out.append({
            "title": " ".join(title.split()), "abstract": summary, "year": year,
            "url": aid_url, "doi": e.findtext("arxiv:doi", default=None, namespaces=_ns),
            "external_id": f"arxiv:{aid}", "source_type": "preprint",
        })
    return out


def _parse_biorxiv(payload: dict, terms: list[str], server: str) -> list[dict]:
    """bioRxiv/medRxiv details API collection → docs FILTRÉS par mots-clés.
    L'API n'offre pas de recherche : le fetcher scanne une fenêtre de dates RÉCENTE
    et le filtrage lexical se fait ici (garde un preprint si ≥ un TIERS des termes
    ≥4 car. apparaissent dans titre+résumé). external_id = <server>:<doi>."""
    out: list[dict] = []
    coll = payload.get("collection") if isinstance(payload, dict) else None
    kw = [t for t in (terms or []) if len(t) >= 4]
    need = max(1, (len(kw) + 2) // 3) if kw else 0
    for it in (coll or []):
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        doi = it.get("doi")
        if not title or not doi:
            continue
        abstract = it.get("abstract") or None
        hay = f"{title} {abstract or ''}".lower()
        if kw and sum(1 for t in kw if t in hay) < need:
            continue
        date = it.get("date") or ""
        out.append({
            "title": title, "abstract": abstract,
            "year": int(date[:4]) if date[:4].isdigit() else None,
            "url": f"https://doi.org/{doi}", "doi": doi,
            "external_id": f"{server}:{doi}", "source_type": "preprint",
        })
    return out


def _openaire_text(v: Any) -> str | None:
    """Extrait une chaîne d'un champ OpenAIRE (str, dict {"$":…}, ou liste)."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("$")
    if isinstance(v, list):
        for x in v:
            t = _openaire_text(x)
            if t:
                return t
    return None


def _parse_openaire_graph(payload: dict) -> list[dict]:
    """OpenAIRE Graph API v2 (/graph/v2/researchProducts) → docs. DÉFENSIF : les champs
    (descriptions, pids, publicationDate) varient en forme. external_id = openaire:<id>.
    Remplace l'ancien /search/publications (retiré le 2026-05-31)."""
    out: list[dict] = []
    results = (payload.get("results") if isinstance(payload, dict) else None) or []
    for r in results:
        if not isinstance(r, dict):
            continue
        title = (r.get("mainTitle") or _openaire_text(r.get("title")) or "").strip()
        oid = r.get("id")
        if not title or not oid:
            continue
        _desc = r.get("descriptions") or r.get("description")
        if isinstance(_desc, list):
            _desc = " ".join((d.get("value") if isinstance(d, dict) else str(d)) or "" for d in _desc)
        elif isinstance(_desc, dict):
            _desc = _desc.get("value")
        abstract = (_desc or "").strip() or None
        _pd = str(r.get("publicationDate") or r.get("publicationYear") or "")
        year = int(_pd[:4]) if _pd[:4].isdigit() else None
        doi = None
        for p in (r.get("pids") or r.get("pid") or []):
            if isinstance(p, dict):
                _sch = str(p.get("scheme") or p.get("type") or p.get("@classid") or "").lower()
                _val = p.get("value") or p.get("id") or p.get("$")
                if _sch == "doi" and _val:
                    doi = _val
                    break
        doi = _normalize_doi(doi)
        out.append({
            "title": title, "abstract": abstract, "year": year,
            "url": (f"https://doi.org/{doi}" if doi else None),
            "doi": doi, "external_id": f"openaire:{oid}", "source_type": "article",
        })
    return out


def _parse_openaire(payload: dict) -> list[dict]:
    """OpenAIRE search/publications (format=json) → docs. BEST-EFFORT : structure
    profondément imbriquée et variable ($-wrapped, dict-ou-liste). external_id = doi:<…>
    ou openaire:<objIdentifier>."""
    out: list[dict] = []
    results = (((payload or {}).get("response") or {}).get("results") or {}).get("result") if isinstance(payload, dict) else None
    if isinstance(results, dict):
        results = [results]
    for r in (results or []):
        if not isinstance(r, dict):
            continue
        res = ((r.get("metadata") or {}).get("oaf:entity") or {}).get("oaf:result")
        if not isinstance(res, dict):
            continue
        title = _openaire_text(res.get("title"))
        if not title:
            continue
        doi = None
        pids = res.get("pid")
        if isinstance(pids, dict):
            pids = [pids]
        for p in (pids or []):
            if isinstance(p, dict) and str(p.get("@classid", "")).lower() == "doi":
                doi = p.get("$"); break
        oid = _openaire_text((r.get("header") or {}).get("dri:objIdentifier")) or _openaire_text(res.get("objIdentifier"))
        eid = (f"doi:{doi}" if doi else (f"openaire:{oid}" if oid else None))
        if not eid:
            continue
        date = _openaire_text(res.get("dateofacceptance")) or ""
        out.append({
            "title": " ".join(title.split()), "abstract": _openaire_text(res.get("description")),
            "year": int(date[:4]) if date[:4].isdigit() else None,
            "url": (f"https://doi.org/{doi}" if doi else None), "doi": doi,
            "external_id": eid, "source_type": "article",
        })
    return out


def _ingest_doc_direct(
    source: str,
    title: str,
    abstract: str | None,
    year: int | None,
    url: str | None,
    external_id: str | None,
    doi: str | None,
    authors: str | None = None,
    journal: str | None = None,
    source_type: str = "article",
    project_context: str = "literev",
) -> tuple[int | None, bool]:
    """INSERT SQL direct d'un document + chunk title_abstract.
    Évite les appels HTTP à l'API locale (POST /documents + POST /chunks).
    Retourne (id_document, is_new) : is_new=True si la ligne vient d'être INSÉRÉE,
    False si le document existait déjà (dédup pré-SELECT, ou course entre fetchers
    parallèles résolue par ON CONFLICT). Le compteur « ingested » ne s'incrémente
    que sur les vrais INSERT - sinon il surcompte les doublons inter-sources.
    """
    # Même classe de bug que le texte intégral, sur le chemin d'ingestion PRINCIPAL :
    # une API JSON peut renvoyer la séquence d'échappement u+0000, que json.loads décode
    # en NUL RÉEL. L'INSERT échoue alors et le document est perdu pour cette recherche.
    # On nettoie à l'entrée, avant `title_norm` (la dédup doit voir le titre déjà
    # normalisé) et avant la construction du chunk title_abstract.
    title = sanitize_db_text(title)
    abstract = sanitize_db_text(abstract)
    authors = sanitize_db_text(authors)
    journal = sanitize_db_text(journal)
    doi = _normalize_doi(doi)
    title_norm = _normalize_title(title)
    content_text = f"{title}\n\n{abstract or ''}".strip()

    # PMID dérivé de l'external_id (« pmid:123 » - format PubMed unifié, cf.
    # _live_fetch_pubmed et le populate) ou d'un external_id purement numérique
    # d'une source PubMed. Renseigne la colonne `pmid` (jusqu'ici NULL par ce
    # chemin) pour que la dédup PMID (maintenance corpus / _softdedup) opère -
    # sans dépendre du format d'external_id. Aucune signature d'appelant modifiée.
    _pmid: str | None = None
    if external_id:
        _eid_s = external_id.strip()
        _mp = re.match(r"^pmid:(\d+)$", _eid_s, re.I)
        if _mp:
            _pmid = _mp.group(1)
        elif source == "pubmed" and _eid_s.isdigit():
            _pmid = _eid_s

    # Dédup en UN SEUL aller-retour (au lieu de deux SELECT séparés) : par external_id,
    # sinon par DOI, sinon par TITRE normalisé - capte les doublons sans DOI (préprints,
    # essais) ou dont le DOI diffère selon la source. Titres < 20 car. normalisés ignorés
    # (faux positifs). Le try/except couvre le cas d'une colonne title_norm absente (migration).
    _has_tn = bool(title_norm and len(title_norm) >= 20)
    try:
        with engine.connect() as _c:
            existing = _c.execute(text("""
                SELECT id FROM literature_document
                WHERE project_context = :ctx AND (
                    external_id = :eid
                    OR (:has_tn AND title_norm = :tn)
                    OR (:has_doi AND doi = :doi))
                ORDER BY (external_id = :eid) DESC, (:has_doi AND doi = :doi) DESC
                LIMIT 1
            """), {"ctx": project_context, "eid": external_id,
                   "tn": title_norm, "has_tn": _has_tn,
                   "doi": doi, "has_doi": bool(doi)}).scalar()
    except Exception:
        with engine.connect() as _c:   # repli : dédup external_id seul (colonne title_norm absente)
            existing = _c.execute(text(
                "SELECT id FROM literature_document "
                "WHERE external_id = :eid AND project_context = :ctx LIMIT 1"
            ), {"eid": external_id, "ctx": project_context}).scalar()
    if existing:
        return (existing, False)   # dédup pré-SELECT (external_id / titre normalisé / DOI)

    # INSERT document + chunk dans UNE SEULE transaction : sinon un crash entre
    # les deux laisse un document sans chunk (jamais indexable/cherchable) - c'est
    # l'origine des documents orphelins observés en production.
    with engine.begin() as _c:
        # ON CONFLICT DO NOTHING SANS cible : capte un conflit sur N'IMPORTE quel
        # index unique - uq_literature_document_doi (DOI) ET uq_litdoc_title_norm
        # (titre normalisé, len≥20). Ferme la course entre fetchers parallèles qui,
        # avant l'index unique sur le titre, pouvaient tous deux passer le pré-SELECT
        # puis INSÉRER deux lignes pour le même article sans DOI (préprint, essai).
        doc_id = _c.execute(text("""
            INSERT INTO literature_document (
                source, title, abstract, year, url, external_id,
                project_context, source_type, doi, authors, journal, title_norm, pmid
            ) VALUES (
                :source, :title, :abstract, :year, :url, :external_id,
                :project_context, :source_type, :doi, :authors, :journal, :title_norm, :pmid
            )
            ON CONFLICT DO NOTHING
            RETURNING id
        """), {
            "source": source, "title": title, "abstract": abstract,
            "year": year, "url": url, "external_id": external_id,
            "project_context": project_context, "source_type": source_type,
            "doi": doi, "authors": authors, "journal": journal,
            "title_norm": (title_norm or None), "pmid": _pmid,
        }).scalar()
        is_new = doc_id is not None
        if doc_id is None:
            # Un INSERT concurrent a gagné la course (conflit DOI ou titre normalisé).
            # On récupère la ligne canonique via les MÊMES clés que le pré-SELECT, au
            # lieu de supposer que le conflit portait sur le DOI (qui peut être NULL).
            doc_id = _c.execute(text("""
                SELECT id FROM literature_document
                WHERE project_context = :ctx AND (
                    external_id = :eid
                    OR (:has_tn AND title_norm = :tn)
                    OR (:has_doi AND doi = :doi))
                ORDER BY (external_id = :eid) DESC, (:has_doi AND doi = :doi) DESC
                LIMIT 1
            """), {"ctx": project_context, "eid": external_id,
                   "tn": title_norm, "has_tn": _has_tn,
                   "doi": doi, "has_doi": bool(doi)}).scalar()

        # INSERT du chunk title_abstract, idempotent : ne crée PAS de second chunk
        # si le document en a déjà un (cas d'un doc atteint via dédup DOI avec un
        # external_id différent - l'origine des chunks dupliqués observés).
        if doc_id is not None and len(content_text) >= 30:
            _c.execute(text("""
                INSERT INTO document_chunk (
                    document_id, chunk_index, content, chunk_type,
                    token_count, chunk_weight, metadata_json
                )
                SELECT :doc_id, 0, :content, 'title_abstract', :token_count, 1.0, '{}'
                WHERE NOT EXISTS (
                    SELECT 1 FROM document_chunk
                    WHERE document_id = :doc_id AND chunk_type = 'title_abstract'
                )
            """), {
                "doc_id": doc_id,
                "content": content_text,
                "token_count": len(content_text.split()),
            })
    # title_norm est désormais inséré directement (colonne présente) → plus d'UPDATE séparé.
    return (doc_id, is_new)
