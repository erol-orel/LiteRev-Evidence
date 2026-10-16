"""Corpus build (populate) and the full enrichment pipeline of a scenario.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import Depends, Query
from sqlalchemy import text, bindparam

from .core import (EXHAUSTIVE_FEDERATION_BUDGET, POPULATE_FEDERATION_BUDGET, _norm_lang, app,
                   engine, logger, redact_keys, require_api_key)
from .documents import (
    METADATA_UPDATE_SQL,
    _llm_lang_directive,
    _metadata_update_params,
    _normalize_doi,
    _strategy_is_degraded,
    _truncate_to_tokens,
    sanitize_db_text,
)
from .scenario_store import (_get_scenario_threshold, _get_user_scenario_or_404, has_abstract_sql,
                             pipeline_enrich_scope, scenario_scope_sql, scenario_threshold_info)
from .search import (
    DOAJ_MAX_QUERY_CHARS,
    EXHAUSTIVE_MAX_PER_SOURCE,
    LIVE_MAX_PER_SOURCE,
    OPENAIRE_MAX_OPERATORS,
    OPENALEX_BISECT_DEPTH,
    OPENALEX_MAX_OPERATORS,
    OPENALEX_SPLIT_MAX_PARTS,
    RANKED_PASS_MAX,
    SOURCE_OUTCOMES_COUNTED,
    S2_MAX_QUERY_ENCODED,
    URL_QUERY_MAX_ENCODED,
    _url_encoded_len,
    _boolean_corpus_ids,
    _bisect_boolean,
    _boolean_to_arxiv,
    _boolean_to_s2,
    _count_operators,
    _clean_boolean,
    _dedup_scenario_links,
    _facet_ops,
    _facets_intersect,
    _generate_search_strategy,
    _ids_within_criteria,
    _kept_link_counts,
    _looks_boolean,
    excludes_preprints,
    preprint_sql,
    _coverage_caveat,
    _multi_query_corpus_ids,
    _normalize_sub_queries,
    _outcome_summary,
    _parse_boolean_ast,
    _plain_keywords,
    _prisma_identification_figures,
    _search_local_doc_ids,
    _set_scenario_corpus,
    _shorten_boolean,
    _source_label,
    _split_boolean,
    _store_prisma_identification,
    _strategy_to_store,
    source_record_keys,
    _strip_field_tags,
    _tokenize_boolean,
    _widen_boolean_for_or_facets,
)
from .sources import (
    EPMC_SEARCH_URL,
    NCBI_ESEARCH_CEILING,
    epmc_send,
    _NCBI_LAST,
    _NCBI_LOCK,
    _NCBI_MIN_INTERVAL,
    _ingest_doc_direct,
    _ncbi_get,
    epmc_query,
    openalex_params,
    pubmed_esearch_by_years,
    s2_headers_paced,
    s2_key,
    pubmed_year_bounds,
    pubmed_years_term,
    _parse_arxiv,
    _parse_arxiv_total,
    _parse_clinicaltrials,
    _parse_core,
    _parse_doaj,
    _parse_openaire_graph,
    _parse_semantic_scholar,
)
from .gesica import _gesica_title, _get_db_gesica_scenario_or_404
from .clustering import (
    _build_clusters_payload,
    _cluster_core,
    _clustering_docs,
    _persist_clustering_result,
    _run_clustering_background,
)
from .knowledge_graph import _precompute_user_kg
from llm_usage import model_for as _model


#: Combien de temps le pipeline ATTEND son étape d'extraction avant de passer à la suite.
#: Elle ne s'arrête pas pour autant : passé ce budget elle continue en arrière-plan, et
#: l'étape le dit (`still_running`). Sans borne, un corpus de plusieurs milliers
#: d'articles pertinents retiendrait le pipeline des heures ; sans attente du tout, le
#: brief et les variables liraient une extraction qui n'a pas encore écrit une ligne.
#: `PIPELINE_EXTRACTION_WAIT_S=0` pour ne pas attendre.
EXTRACTION_WAIT_S = 3600


def _extraction_wait_budget_s() -> float:
    """Le budget d'attente, en secondes, lu dans l'environnement. Une valeur illisible
    ou negative garde le defaut. Pur, hors la lecture de l'environnement."""
    raw = (os.getenv("PIPELINE_EXTRACTION_WAIT_S") or "").strip()
    if not raw:
        return float(EXTRACTION_WAIT_S)
    try:
        value = float(raw)
    except ValueError:
        return float(EXTRACTION_WAIT_S)
    return value if value >= 0 else float(EXTRACTION_WAIT_S)



def _enrich_gate() -> str:
    """Le périmètre des étapes d'enrichissement du pipeline, en SQL.

    `all` par défaut, soit exactement la sélection d'avant : hors doublons et hors
    articles écartés, ce qui n'a jamais été une restriction mais la même règle que
    partout. `PIPELINE_ENRICH_SCOPE=relevant` la réduit au sous-ensemble pertinent,
    pour un jour où le budget de modèle doit être tenu."""
    return scenario_scope_sql(pipeline_enrich_scope(), doc="ld", link="asn")


def _auto_pipeline_after_search() -> bool:
    """Le pipeline complet d'enrichissement enchaîne-t-il automatiquement après une
    recherche (corpus construit ET scoré) ? Oui par défaut ; AUTO_PIPELINE_AFTER_SEARCH=0
    pour s'en tenir à la recherche (les onglets calculent alors à l'ouverture)."""
    return os.getenv("AUTO_PIPELINE_AFTER_SEARCH", "1").strip().lower() not in ("0", "false", "no", "off")


# ── Une page qui échoue se réessaie, un nombre BORNÉ de fois ─────────────────────
# Neuf boucles de pagination sur onze s'arrêtaient au premier délai dépassé, au premier
# 429 ou 5xx d'une page N : la source finissait en `error` avec les pages d'avant N, et
# tout ce qui suivait était perdu ; un lot efetch de PubMed (200 notices) en échec était
# sauté. Trois tentatives, 2 s puis 4 s d'attente (8 s pour une quatrième, si on la
# règle), un `Retry-After` respecté jusqu'à FETCH_RETRY_AFTER_CAP_S, et jamais une
# attente au-delà du budget de fédération. Après la dernière, la source reste en
# `error`, nommée sur la carte, comme avant. Lus à l'appel : un test les règle à zéro.
FETCH_RETRY_ATTEMPTS = max(1, int(os.getenv("FETCH_RETRY_ATTEMPTS", "3") or 3))
FETCH_RETRY_BACKOFF_S = (2.0, 4.0, 8.0)
FETCH_RETRY_AFTER_CAP_S = 30.0
#: Les attentes d'un 429 quand l'appel porte la clé du serveur (Semantic Scholar). Avec la clé
#: et l'espacement d'une requête par seconde, la recherche de l'exposition du 13 octobre 2026
#: s'arrêtait encore à la deuxième page (200 notices sur 10 257) après les quatre attentes de
#: 2 à 8 s faites pour le quota anonyme, qu'insister durcit. Une clé ne l'est pas pour avoir
#: attendu : huit tentatives, quatre minutes au plus, jamais au-delà du budget de fédération.
#: Lues à l'appel : un test les raccourcit.
KEYED_429_WAITS_S = (5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0)


def _pubmed_year(article_elem) -> int | None:
    """L'année d'une notice PubMed (efetch XML) : PubDate/Year, sinon les quatre premiers
    chiffres de PubDate/MedlineDate (« 1998 Spring », « 1998-1999 », « Winter 2017 »),
    sinon ArticleDate/Year. Le MedlineDate n'était pas lu : ces notices entraient sans
    année, et toute recherche bornée en années les perdait. PUR/testable."""
    year_text = article_elem.findtext(".//PubDate/Year") or ""
    if not year_text[:4].isdigit():
        _m = re.search(r"(\d{4})", article_elem.findtext(".//PubDate/MedlineDate") or "")
        year_text = _m.group(1) if _m else ""
    if not year_text[:4].isdigit():
        year_text = article_elem.findtext(".//ArticleDate/Year") or ""
    return int(year_text[:4]) if year_text[:4].isdigit() else None


def _pubmed_authors(article_elem) -> str | None:
    """TOUS les auteurs d'une notice PubMed (efetch XML), « Nom Prénom » séparés par « ; ».
    La liste était coupée aux six premiers, sans « et al. », si bien que les exports (RIS,
    BibTeX) et les citations perdaient les autres. L'affichage abrégé (six puis « et al. »)
    est celui du rapport (report.format_reference), pas celui du stockage. PUR/testable."""
    names = []
    for author in article_elem.findall(".//AuthorList/Author"):
        last = author.findtext("LastName") or ""
        first = author.findtext("ForeName") or ""
        if last:
            names.append(f"{last} {first}".strip())
    return "; ".join(names) if names else None


def _crossref_year(item: dict) -> int | None:
    """L'année de publication d'une notice Crossref : `issued` (la date de publication
    que Crossref retient), puis `published-print`, `published-online`, `published`, et
    `created` en dernier. `created` seul était lu : c'est la date du DÉPÔT du DOI, si
    bien qu'un article de 1995 déposé en 2009 entrait daté de 2009 et que les filtres
    d'années le plaçaient mal. PUR/testable."""
    for key in ("issued", "published-print", "published-online", "published", "created"):
        parts = ((item or {}).get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            try:
                return int(parts[0][0])
            except (TypeError, ValueError):
                continue
    return None


# ── Les appels NCBI de l'étape « texte intégral » ────────────────────────────────
# Ils partaient par un `requests.get` à eux, sans le throttle partagé, sans la clé d'API
# et sans le passage en POST, pendant que cinq fils les lançaient en parallèle : au-delà
# des 3 requêtes par seconde que NCBI accepte sans clé, des 429, et des textes intégraux
# PMC perdus. Ils passent désormais par `sources._ncbi_get`, comme l'esearch du populate.
_NCBI_EUTILS_FT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_PMC_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


def _ft_ncbi_get(url: str, params: dict, timeout: int = 20, use_key: bool = True):
    """Un appel NCBI de l'étape texte intégral, par le helper throttlé ; None sur échec
    (le contrat de `_ft_get`, que ces appels empruntaient)."""
    try:
        if use_key:
            return _ncbi_get(url, params, timeout=timeout)
        return _ncbi_get(url, params, timeout=timeout, use_key=False)
    except Exception as _e:                                  # noqa: BLE001
        logger.debug(f"_ft_ncbi_get {url}: {_e}")
        return None


def _ft_resolve_pmcid(ext_id: str, pmid_val, doi_val: str, email: str) -> str | None:
    """Le PMCID d'un document : lu dans son identifiant externe, sinon demandé à NCBI
    (esummary par PMID, puis le convertisseur d'identifiants PMC par DOI)."""
    if ext_id and "PMC" in ext_id.upper():
        _m = re.search(r"(\d{5,10})", ext_id)
        if _m:
            return f"PMC{_m.group(1)}"
    if ext_id and re.match(r"^PMC\d+$", ext_id.upper()):
        return ext_id.upper()
    _cand = None
    if ext_id and ext_id.isdigit():
        _cand = ext_id
    elif pmid_val:
        _cand = str(pmid_val).replace("PMID:", "").strip()
    if _cand:
        _r2 = _ft_ncbi_get(f"{_NCBI_EUTILS_FT}/esummary.fcgi",
                           {"db": "pubmed", "id": _cand, "retmode": "json"})
        if _r2 is not None and _r2.status_code == 200:
            try:
                _aids = _r2.json().get("result", {}).get(_cand, {}).get("articleids", [])
                for _aid in _aids:
                    if _aid.get("idtype") == "pmcid":
                        _pmc = _aid.get("value", "").replace("PMC", "")
                        if _pmc:
                            return f"PMC{_pmc}"
            except Exception:
                pass
    if doi_val:
        # Le convertisseur PMC n'est pas eutils : sa requête reste celle d'avant, sans la
        # clé d'eutils, mais espacée par le même verrou.
        _r3 = _ft_ncbi_get(_PMC_IDCONV_URL,
                           {"ids": doi_val, "format": "json", "tool": "literev", "email": email},
                           use_key=False)
        if _r3 is not None and _r3.status_code == 200:
            try:
                _recs = _r3.json().get("records", [])
                if _recs and _recs[0].get("pmcid"):
                    return _recs[0]["pmcid"]
            except Exception:
                pass
    return None


def _ft_fetch_pmc_xml(pmcid: str):
    """Le texte intégral XML d'un article PMC, par efetch (db=pmc) ; None sur échec."""
    return _ft_ncbi_get(f"{_NCBI_EUTILS_FT}/efetch.fcgi",
                        {"db": "pmc", "id": pmcid.replace("PMC", ""), "rettype": "full", "retmode": "xml"},
                        timeout=30)


# ── Le texte intégral en accès ouvert d'un article ──────────────────────────────
# Il vivait dans une fermeture de `_run_user_scenario_full_pipeline` : seul le pipeline
# complet savait télécharger un texte, et sur TOUT le corpus d'un scénario (17 510
# articles pour HPAI_last, dont 1 075 pertinents), pendant que le lot « texte intégral »
# du panneau d'enrichissement ne trouvait que des LIENS. Les mêmes fonctions servent
# désormais aux deux (`download_fulltext`), sur la portée que l'appelant choisit.
_EPMC_BASE_FT = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_CHUNK_SIZE_FT = 4000
_CHUNK_OVERLAP_FT = 400
#: Les articles traités en parallèle : les appels NCBI restent espacés par le verrou partagé.
FULLTEXT_WORKERS = 5


def _unpaywall_email() -> str:
    return os.getenv("UNPAYWALL_EMAIL", "literev@gesica.ch")


def _ft_get(url, params=None, timeout=20, headers=None):
    import time as _time
    import requests as _requests
    for _att in range(3):
        try:
            _r = _requests.get(url, params=params, timeout=timeout,
                               headers={"User-Agent": "LiteRev-Evidence/1.0", **(headers or {})})
            if _r.status_code == 429:
                _time.sleep(int(_r.headers.get("Retry-After", 10)))
                continue
            return _r
        except Exception as _fe:
            logger.debug(f"_ft_get {url}: attempt {_att+1} failed: {_fe}")
            _time.sleep(1)
    logger.debug(f"_ft_get {url}: all retries exhausted")
    return None


def _parse_pmc_xml_ft(xml_str):
    import xml.etree.ElementTree as _ET_ft
    _skip = {"ref-list", "ack", "fn-group", "glossary", "app-group", "notes", "bio", "author-notes"}
    try:
        _root = _ET_ft.fromstring(xml_str)
    except _ET_ft.ParseError:
        xml_str = re.sub(r"&(?!amp;|lt;|gt;|apos;|quot;)", "&amp;", xml_str)
        try:
            _root = _ET_ft.fromstring(xml_str)
        except Exception:
            return None
    _parts = []

    def _walk(n):
        _tag = n.tag.split("}")[-1] if "}" in n.tag else n.tag
        if _tag in _skip:
            return
        if n.text and n.text.strip():
            _parts.append(n.text.strip())
        for _ch in n:
            _walk(_ch)
        if n.tail and n.tail.strip():
            _parts.append(n.tail.strip())
    _walk(_root)
    _txt = re.sub(r"\s+", " ", " ".join(_parts)).strip()
    return _txt if len(_txt) > 50 else None


def _extract_pdf_text_ft(pdf_url):
    import subprocess as _subprocess
    import tempfile as _tempfile
    import requests as _requests
    try:
        # 60 s pour le téléchargement, 120 s pour pdftotext : à 30 s chacun, les
        # gros PDF (suppléments, rapports) perdaient leur texte.
        _r = _requests.get(pdf_url, timeout=60, stream=True,
                           headers={"User-Agent": "LiteRev-Evidence/1.0"})
        if _r.status_code != 200:
            return None
        _ct = _r.headers.get("content-type", "")
        if "pdf" not in _ct.lower() and not pdf_url.lower().endswith(".pdf"):
            _txt = re.sub(r"<[^>]+>", " ", _r.text)
            _txt = sanitize_db_text(re.sub(r"\s+", " ", _txt)).strip()
            return _txt if len(_txt) > 500 else None
        with _tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as _f:
            for _chunk in _r.iter_content(chunk_size=8192):
                _f.write(_chunk)
            _tmp = _f.name
        try:
            _res = _subprocess.run(["pdftotext", "-layout", _tmp, "-"],
                                   capture_output=True, text=True, timeout=120)
        finally:
            os.unlink(_tmp)
        if _res.returncode == 0 and _res.stdout.strip():
            # pdftotext émet des NUL sur certains PDF mal formés ; sans ce
            # nettoyage l'INSERT du chunk échoue et l'article perd son texte.
            _txt = sanitize_db_text(re.sub(r"\s+", " ", _res.stdout)).strip()
            return _txt if len(_txt) > 500 else None
    except Exception:
        pass
    return None


def _chunk_text_ft(text_str):
    """Découpage SÉMANTIQUE par phrases : on coupe aux frontières de phrases (et non au
    milieu d'un mot ou d'une idée), puis on fusionne en chunks d'environ _CHUNK_SIZE_FT
    caractères avec un recouvrement au niveau de la phrase."""
    text_str = re.sub(r"\s+", " ", text_str).strip()
    if not text_str:
        return []
    _sentences = re.split(r"(?<=[.!?])\s+", text_str)
    _chunks: list[str] = []
    _cur: list[str] = []
    _cur_len = 0
    for _s in _sentences:
        if _cur and _cur_len + len(_s) + 1 > _CHUNK_SIZE_FT:
            _chunks.append(" ".join(_cur).strip())
            _ov: list[str] = []
            _olen = 0
            for _p in reversed(_cur):
                if _olen + len(_p) <= _CHUNK_OVERLAP_FT:
                    _ov.insert(0, _p)
                    _olen += len(_p)
                else:
                    break
            _cur = _ov
            _cur_len = sum(len(_p) + 1 for _p in _cur)
        _cur.append(_s)
        _cur_len += len(_s) + 1
    if _cur:
        _chunks.append(" ".join(_cur).strip())
    # Phrase unique trop longue (> 1.5x la cible) : re-découpe en fenêtre mot.
    _final: list[str] = []
    _max = int(_CHUNK_SIZE_FT * 1.5)
    for _c in _chunks:
        if len(_c) <= _max:
            _final.append(_c)
            continue
        _st = 0
        while _st < len(_c):
            _en = min(_st + _CHUNK_SIZE_FT, len(_c))
            if _en < len(_c):
                _cut = _c.rfind(" ", _st, _en)
                if _cut > _st:
                    _en = _cut
            _final.append(_c[_st:_en].strip())
            _st = _en
    return [c for c in _final if len(c) > 50]


def _insert_fulltext_chunks_ft(doc_id, chunks, source_label, emb_client=None):
    """Insère les chunks fulltext_section et les embedde immédiatement si possible."""
    import time as _time
    with engine.begin() as _c:
        _c.execute(text(
            "DELETE FROM document_chunk WHERE document_id = :did "
            "AND chunk_type IN ('fulltext_section', 'full_text')"
        ), {"did": doc_id})
        for _i, _chunk_text in enumerate(chunks):
            _meta = json.dumps({"source": source_label, "chunk_index": _i})
            # `sanitize_db_text` est un filet de sécurité au POINT D'ÉCRITURE : l'extracteur
            # nettoie déjà, mais le texte vient de plusieurs provenances (PMC, Europe PMC,
            # PDF) et un seul NUL fait échouer toute la transaction, DELETE compris.
            _c.execute(text("""
                INSERT INTO document_chunk
                    (document_id, content, chunk_index, chunk_type, chunk_weight, metadata_json)
                VALUES (:did, :content, :idx, 'fulltext_section', 1.0, CAST(:meta AS jsonb))
            """), {"did": doc_id, "content": sanitize_db_text(_chunk_text),
                   "idx": _i, "meta": _meta})
        _c.execute(text(
            "UPDATE literature_document SET has_fulltext = true, open_access = true WHERE id = :did"
        ), {"did": doc_id})
    # Embedder les nouveaux chunks immédiatement si client OpenAI disponible
    if emb_client and chunks:
        try:
            with engine.connect() as _c2:
                _new_chunks = _c2.execute(text("""
                    SELECT id, content FROM document_chunk
                    WHERE document_id = :did AND chunk_type = 'fulltext_section'
                      AND embedding IS NULL ORDER BY chunk_index
                """), {"did": doc_id}).mappings().fetchall()
            for _bi in range(0, len(_new_chunks), 50):
                _batch = _new_chunks[_bi:_bi + 50]
                _texts = [_truncate_to_tokens(r["content"]) for r in _batch]
                _emb_resp = emb_client.embeddings.create(model=_model("embedding"), input=_texts)
                for _k, _ed in enumerate(_emb_resp.data):
                    _vec = "[" + ",".join(str(x) for x in _ed.embedding) + "]"
                    with engine.begin() as _c3:
                        _c3.execute(text(
                            "UPDATE document_chunk SET embedding = CAST(:vec AS vector) WHERE id = :cid"
                        ), {"vec": _vec, "cid": _batch[_k]["id"]})
                _time.sleep(0.1)
        except Exception as _emb_e:
            logger.warning(f"Fulltext embed doc {doc_id}: {_emb_e}")
    return len(chunks)


def _note_oa_link(doc_id: int, url: str) -> None:
    """Le lien d'accès ouvert qu'Unpaywall donne, gardé même quand son PDF ne se lit pas :
    c'est ce que le lot « texte intégral » rangeait seul avant de savoir télécharger."""
    try:
        with engine.begin() as _c:
            _c.execute(text("""
                UPDATE literature_document
                SET oa_url = COALESCE(oa_url, :url), oa_url_found_at = COALESCE(oa_url_found_at, NOW()),
                    url = COALESCE(NULLIF(TRIM(COALESCE(url, '')), ''), :url)
                WHERE id = :did
            """), {"url": url, "did": doc_id})
    except Exception as _e:                                  # noqa: BLE001
        logger.debug(f"oa_url doc {doc_id}: {_e}")


#: Le lien de téléchargement de CORE, que sa notice porte en `url` : le PDF en accès ouvert.
_CORE_DOWNLOAD_RE = re.compile(r"^https?://core\.ac\.uk/download/", re.I)
#: L'API de CORE (v3), que la recherche interroge déjà avec la même clé (`CORE_API_KEY`).
_CORE_API = "https://api.core.ac.uk/v3"


def _core_fulltext(ext_id: str, doi: str) -> tuple[str | None, str | None]:
    """Le texte que CORE a déjà extrait de sa copie en dépôt (`fullText`), lu par son API avec
    la clé de la recherche : par son identifiant pour une notice venue de CORE, sinon par le DOI,
    et un résultat dont le DOI diffère n'est pas pris. Rend (texte, raison d'un échec)."""
    key = (os.getenv("CORE_API_KEY") or "").strip()
    if not key:
        return None, "core_api:no_key"
    headers = {"Authorization": f"Bearer {key}"}
    cid = ext_id[5:] if ext_id.startswith("core:") else ""
    if cid.isdigit():
        r = _ft_get(f"{_CORE_API}/works/{cid}", headers=headers, timeout=30)
    elif doi:
        r = _ft_get(f"{_CORE_API}/search/works", params={"q": f'doi:"{doi}"', "limit": 3},
                    headers=headers, timeout=30)
    else:
        return None, "core_api:no_identifier"
    if r is None:
        return None, "core_api:timeout"
    if r.status_code != 200:
        return None, f"core_api:http_{r.status_code}"
    try:
        body = r.json()
    except ValueError:
        return None, "core_api:not_json"
    items = [body] if cid.isdigit() else (body.get("results") or []) if isinstance(body, dict) else []
    for it in items:
        if not isinstance(it, dict) or (not cid.isdigit()
                                        and (_normalize_doi(it.get("doi") or "") or "").lower() != doi.lower()):
            continue                                             # un DOI ne distingue pas la casse
        text = str(it.get("fullText") or "").strip()
        if len(text) > 500:
            return text, None
    return None, "core_api:no_text"


def _fulltext_of(row) -> tuple[str | None, str | None, list[str]]:
    """Le texte intégral en accès ouvert d'un article, sa provenance, et pourquoi chaque
    source a échoué : PMC (Europe PMC puis NCBI, en XML), Unpaywall, bioRxiv/medRxiv,
    Semantic Scholar, OpenAlex, puis la copie de CORE, son texte par son API ou son PDF (les
    PDF lus par pdftotext)."""
    _ext_id = row["external_id"] or ""
    _doi = _normalize_doi(row["doi"] or "") or ""
    _pmid = row["pmid"] or ""
    _fulltext = None
    _source_used = None
    _fail: list[str] = []
    _email = _unpaywall_email()

    # Source 1 : PMC (les appels NCBI par le helper throttlé, cf. _ft_ncbi_get)
    _pmcid = _ft_resolve_pmcid(_ext_id, _pmid, _doi, _email)
    if _pmcid:
        _r_pmc = _ft_get(f"{_EPMC_BASE_FT}/{_pmcid}/fullTextXML", timeout=30)
        if _r_pmc and _r_pmc.status_code == 200 and _r_pmc.text.strip().startswith("<"):
            _fulltext = _parse_pmc_xml_ft(_r_pmc.text)
            if _fulltext and len(_fulltext) > 500:
                _source_used = f"europepmc:{_pmcid}"
            else:
                _fulltext = None
                _fail.append(f"europepmc:{_pmcid}:xml_parse_empty")
        elif _r_pmc:
            _fail.append(f"europepmc:{_pmcid}:http_{_r_pmc.status_code}")
        if not _fulltext:
            _r_ncbi = _ft_fetch_pmc_xml(_pmcid)
            if _r_ncbi and _r_ncbi.status_code == 200 and _r_ncbi.text.strip().startswith("<"):
                _fulltext = _parse_pmc_xml_ft(_r_ncbi.text)
                if _fulltext and len(_fulltext) > 500:
                    _source_used = f"pmc:{_pmcid}"
                else:
                    _fulltext = None
                    _fail.append(f"pmc:{_pmcid}:xml_parse_empty")
            elif _r_ncbi:
                _fail.append(f"pmc:{_pmcid}:http_{_r_ncbi.status_code}")
    else:
        _fail.append("pmcid:not_resolved")

    # Source 2 : Unpaywall
    if not _fulltext and _doi and _doi.startswith("10."):
        _r_uw = _ft_get(f"https://api.unpaywall.org/v2/{_doi}", params={"email": _email})
        if _r_uw and _r_uw.status_code == 200:
            try:
                _uw_data = _r_uw.json()
                _best = _uw_data.get("best_oa_location") or {}
                _pdf_url = _best.get("url_for_pdf") or _best.get("url")
                if not _pdf_url:
                    for _loc in _uw_data.get("oa_locations", []):
                        if _loc.get("url_for_pdf"):
                            _pdf_url = _loc["url_for_pdf"]
                            break
                if _pdf_url:
                    _note_oa_link(row["id"], _pdf_url)
                    _fulltext = _extract_pdf_text_ft(_pdf_url)
                    if _fulltext and len(_fulltext) > 500:
                        _source_used = "unpaywall"
                    else:
                        _fulltext = None
                        _fail.append("unpaywall:pdf_empty")
                else:
                    _fail.append(f"unpaywall:not_oa(is_oa={_uw_data.get('is_oa')})")
            except Exception as _uw_e:
                _fail.append(f"unpaywall:parse_error:{_uw_e}")
        elif _r_uw:
            _fail.append(f"unpaywall:http_{_r_uw.status_code}")
        else:
            _fail.append("unpaywall:timeout")
    elif not _doi:
        _fail.append("unpaywall:skipped_no_doi")

    # Source 3 : bioRxiv/medRxiv
    if not _fulltext and _doi and _doi.startswith("10.1101/"):
        for _srv in ["biorxiv", "medrxiv"]:
            _r_bx = _ft_get(f"https://api.biorxiv.org/details/{_srv}/{_doi}/na/json")
            if _r_bx and _r_bx.status_code == 200:
                try:
                    _coll = _r_bx.json().get("collection", [])
                    if _coll:
                        _fulltext = _extract_pdf_text_ft(f"https://www.{_srv}.org/content/{_doi}.full.pdf")
                        if _fulltext and len(_fulltext) > 500:
                            _source_used = _srv
                            break
                        _fulltext = None
                        _fail.append(f"{_srv}:pdf_empty")
                    else:
                        _fail.append(f"{_srv}:not_found")
                except Exception as _bx_e:
                    _fail.append(f"{_srv}:parse_error:{_bx_e}")

    # Source 4 : Semantic Scholar
    if not _fulltext:
        # Une notice venue de Semantic Scholar sans DOI ni PMID se cherche par son propre
        # identifiant (`s2:<paperId>`), que l'API lit tel quel.
        _ss_id = (f"DOI:{_doi}" if _doi else f"PMID:{_pmid}" if _pmid
                  else f"PMCID:{_ext_id}" if _ext_id and _ext_id.upper().startswith("PMC")
                  else _ext_id[3:] if _ext_id.startswith("s2:") and len(_ext_id) > 3 else None)
        if _ss_id:
            # La clé de Semantic Scholar quand le serveur en a une, comme la recherche :
            # sans elle, l'appel puise dans le quota anonyme, souvent épuisé (429). Avec
            # elle, l'appel attend son tour (une requête par seconde pour tout le serveur).
            _r_ss = _ft_get(f"https://api.semanticscholar.org/graph/v1/paper/{_ss_id}",
                            params={"fields": "openAccessPdf,abstract"},
                            headers=s2_headers_paced() or None)
            if _r_ss and _r_ss.status_code == 200:
                try:
                    _oa_pdf = _r_ss.json().get("openAccessPdf")
                    if _oa_pdf and _oa_pdf.get("url"):
                        _fulltext = _extract_pdf_text_ft(_oa_pdf["url"])
                        if _fulltext and len(_fulltext) > 500:
                            _source_used = "semanticscholar"
                        else:
                            _fulltext = None
                            _fail.append("semanticscholar:pdf_empty")
                    else:
                        _fail.append("semanticscholar:no_oa_pdf")
                except Exception as _ss_e:
                    _fail.append(f"semanticscholar:parse_error:{_ss_e}")
            elif _r_ss:
                _fail.append(f"semanticscholar:http_{_r_ss.status_code}")
        else:
            _fail.append("semanticscholar:no_identifier")

    # Source 5 : OpenAlex
    if not _fulltext and (_ext_id.startswith("W") or _doi):
        _oa_work_url = (f"https://api.openalex.org/works/{_ext_id}" if _ext_id.startswith("W")
                        else f"https://api.openalex.org/works/doi:{_doi}")
        _r_oa = _ft_get(_oa_work_url, params=openalex_params({"select": "open_access"}))
        if _r_oa and _r_oa.status_code == 200:
            try:
                _oa_info = _r_oa.json().get("open_access", {})
                _oa_url = _oa_info.get("oa_url")
                if _oa_url:
                    _fulltext = _extract_pdf_text_ft(_oa_url)
                    if _fulltext and len(_fulltext) > 500:
                        _source_used = "openalex_oa"
                    else:
                        _fulltext = None
                        _fail.append("openalex:pdf_empty")
                else:
                    _fail.append(f"openalex:not_oa(is_oa={_oa_info.get('is_oa')})")
            except Exception as _oa_e:
                _fail.append(f"openalex:parse_error:{_oa_e}")
        elif _r_oa:
            _fail.append(f"openalex:http_{_r_oa.status_code}")

    # Source 6 : la copie que CORE a moissonnée. CORE agrège des dépôts en accès ouvert, et son
    # API rend le texte qu'il en a déjà extrait : par son identifiant pour une notice venue de
    # CORE (elles n'ont souvent ni DOI ni PMID : 41 des 536 articles pertinents sans texte de
    # HPAI_last, le 12 octobre 2026), par le DOI pour les autres. Son lien de téléchargement,
    # le PDF lui-même, ne se lisait pas depuis le serveur (1 sur 41) : il reste le dernier recours.
    if not _fulltext and (_ext_id.startswith("core:") or _doi):
        _core_text, _core_why = _core_fulltext(_ext_id, _doi)
        if _core_text:
            _fulltext, _source_used = _core_text, "core"
        elif _core_why != "core_api:no_key":
            _fail.append(_core_why)
    _own_url = str((row.get("url") if hasattr(row, "get") else None) or "")
    if not _fulltext and _CORE_DOWNLOAD_RE.match(_own_url):
        _fulltext = _extract_pdf_text_ft(_own_url)
        if _fulltext and len(_fulltext) > 500:
            _source_used = "core"
        else:
            _fulltext = None
            _fail.append("core:pdf_empty")
    return (_fulltext if _source_used else None), _source_used, _fail


def _fetch_and_store_fulltext(row, emb_client=None) -> tuple[bool, str | None]:
    """Un article : son texte téléchargé, découpé et rangé ; (succès, raison d'un échec)."""
    try:
        _fulltext, _source_used, _fail = _fulltext_of(row)
        if _fulltext and _source_used:
            _chunks_ft = _chunk_text_ft(_fulltext)
            if _chunks_ft:
                _insert_fulltext_chunks_ft(row["id"], _chunks_ft, _source_used, emb_client)
                return True, None
            logger.warning(f"Fulltext doc {row['id']}: text retrieved but produced 0 chunks (source={_source_used})")
            return False, "0_chunks"
        logger.info(f"Fulltext unavailable doc {row['id']} (ext_id={row['external_id'] or ''}, doi={row['doi'] or ''}): "
                    f"{' | '.join(_fail) or 'no_sources_tried'}")
        return False, "not_found"
    except Exception as _ft_e:
        logger.warning(f"Fulltext doc {row['id']}: {_ft_e}")
        return False, str(_ft_e)


#: Un article SANS texte intégral, c'est un article sans chunk de texte intégral : la
#: même question que l'extraction se pose (`has_fulltext_chunk`). Le drapeau `has_fulltext`
#: ne la tranche pas : un lien trouvé l'a longtemps posé sans qu'aucun texte soit rangé, et
#: un article ainsi marqué n'était plus jamais téléchargé.
NO_FULLTEXT_CHUNK_SQL = ("NOT EXISTS (SELECT 1 FROM document_chunk c WHERE c.document_id = ld.id"
                         " AND c.chunk_type IN ('fulltext_section', 'full_text'))")


def rows_without_fulltext(scenario_id: str, scope: str, limit: int | None = None) -> list[dict]:
    """Les articles d'un scénario dont aucun texte intégral n'est rangé, dans la portée
    donnée (`relevant` : le sous-ensemble pertinent, `all` : tout le scénario)."""
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(f"""
            SELECT ld.id, ld.external_id, ld.doi, ld.pmid, ld.source, ld.url
            FROM literature_document ld
            JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
            WHERE ld.project_context = 'literev'
              AND ({scenario_scope_sql(scope, doc="ld", link="asn")})
              AND {NO_FULLTEXT_CHUNK_SQL}
            ORDER BY ld.id
            LIMIT :lim
        """), {"sid": scenario_id, "lim": limit}).mappings().fetchall()]


def download_fulltext(rows: list[dict], progress=None) -> dict[str, int]:
    """Télécharge le texte intégral en accès ouvert de ces articles, en parallèle, et le
    range en chunks (embeddés si une clé OpenAI est réglée). `progress(done, total,
    failed)` est appelé au fil de l'eau. Un article qui gagne ainsi un texte est relu par
    la prochaine extraction (`extraction._needs_extraction`)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    emb_client = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            from llm_usage import MeteredOpenAI as _OAI_ft
            emb_client = _OAI_ft(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:                                    # noqa: BLE001
            emb_client = None
    total, fetched, failed, done = len(rows), 0, 0, 0
    lock = threading.Lock()
    if not rows:
        return {"total": 0, "fetched": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=FULLTEXT_WORKERS) as executor:
        futures = [executor.submit(_fetch_and_store_fulltext, row, emb_client) for row in rows]
        for future in as_completed(futures):
            try:
                ok, _ = future.result()
            except Exception as e:                           # noqa: BLE001
                ok = False
                logger.error(f"Error in fulltext worker: {e}")
            with lock:
                done += 1
                fetched += 1 if ok else 0
                failed += 0 if ok else 1
                if progress and (done % 10 == 0 or done == total):
                    try:
                        progress(done, total, failed)
                    except Exception:                        # noqa: BLE001
                        pass
    return {"total": total, "fetched": fetched, "failed": failed}


# ── Cache des réponses des sources (par fetcher et par requête) ──────────────────
# Une recherche relancée quelques heures après la première redemandait à chaque source
# tout son résultat, jusqu'au plafond par source, pour ne rien insérer (les documents
# étaient déjà en base) : trois minutes de fédération pour rien. On mémorise, par fetcher
# et par requête (requête, filtres, plafond), les documents qu'il a renvoyés quand sa
# pagination est allée au bout sans erreur. Une relance dans le délai SOURCE_CACHE_TTL_S
# rejoue ces liens (mêmes compteurs PRISMA, même source-union) sans appel réseau ; les
# fetchers coupés par le budget ou en erreur ne sont pas mémorisés. force_live=True sur
# /populate ignore le cache.
SOURCE_CACHE_TTL_S = int(os.getenv("SOURCE_CACHE_TTL_S", str(12 * 3600)) or 0)


def _ensure_source_query_cache() -> None:
    try:
        with engine.begin() as _c:
            _c.execute(text("""
                CREATE TABLE IF NOT EXISTS source_query_cache (
                    fetcher     VARCHAR(40) NOT NULL,
                    query_hash  VARCHAR(40) NOT NULL,
                    query       TEXT,
                    links       JSONB NOT NULL,
                    n_records   INTEGER NOT NULL DEFAULT 0,
                    fetched_at  TIMESTAMP NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (fetcher, query_hash)
                )
            """))
            # Le total que la source avait annoncé, rejoué avec ses liens : sans lui, une
            # recherche servie depuis le cache ne savait plus qu'une source avait été
            # plafonnée, et la carte PRISMA de HPAI_last n'en nommait aucune alors que
            # quatre l'étaient (Europe PMC 1 996 sur 13 810).
            _c.execute(text("ALTER TABLE source_query_cache ADD COLUMN IF NOT EXISTS totals JSONB"))
            # Ce que la carte doit encore dire d'une réponse rejouée : les tranches d'années
            # de PubMed au-delà de 10 000 notices. Sans elles, la même recherche relancée
            # depuis le cache affichait plus de 10 000 notices PubMed sans dire comment.
            _c.execute(text("ALTER TABLE source_query_cache ADD COLUMN IF NOT EXISTS meta JSONB"))
    except Exception as _e:                                  # noqa: BLE001 - never blocks boot
        logger.warning(f"_ensure_source_query_cache: {_e}")


try:
    _ensure_source_query_cache()
except Exception as _e:                                      # noqa: BLE001
    logger.warning(f"_ensure_source_query_cache: {_e}")


# ── Où une source COUPÉE par le budget s'était arrêtée de lire ───────────────────────
# Une recherche exhaustive demande à chaque base TOUT ce qu'elle apparie. Europe PMC en
# annonce 72 847 sur la question COVID exposition, OpenAlex 66 395 : cela ne tient pas dans
# le budget d'une fédération, donc la source est coupée en route et la carte le dit.
# Jusqu'ici la reprise repartait du DÉBUT (curseur « * », décalage 0) : elle relisait les
# mêmes pages, le budget s'épuisait au même endroit, et la quatrième reprise sur ce
# scénario n'a rapporté AUCUNE notice nouvelle. Autrement dit, enchaîner les reprises ne
# convergeait pas, et « exhaustif » ne pouvait pas se terminer au-delà d'une certaine
# taille. La position de lecture est donc gardée, par scénario, par source et par requête,
# et la reprise continue là où la précédente s'est arrêtée.
def _ensure_source_fetch_cursor() -> None:
    try:
        with engine.begin() as _c:
            _c.execute(text("""
                CREATE TABLE IF NOT EXISTS source_fetch_cursor (
                    scenario_id VARCHAR(64) NOT NULL,
                    fetcher     VARCHAR(40) NOT NULL,
                    query_hash  VARCHAR(40) NOT NULL,
                    position    TEXT,
                    fetched     BIGINT NOT NULL DEFAULT 0,
                    rounds      INTEGER NOT NULL DEFAULT 0,
                    updated_at  TIMESTAMP NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (scenario_id, fetcher, query_hash)
                )
            """))
    except Exception as _e:                                  # noqa: BLE001 - never blocks boot
        logger.warning(f"_ensure_source_fetch_cursor: {_e}")


try:
    _ensure_source_fetch_cursor()
except Exception as _e:                                      # noqa: BLE001
    logger.warning(f"_ensure_source_fetch_cursor: {_e}")


def cursor_action(outcome: str, has_position: bool, resumed: bool) -> str | None:
    """Que faire de la position de lecture d'une source, selon son issue.

    `save` : coupée en route, on garde où elle en était pour que la prochaine reprise
    continue. `clear` : elle a fini de lire (plus rien à reprendre), ou elle est repartie
    d'une position mémorisée et n'a pas pu lire une seule page, ce qui arrive quand l'API
    a périmé le curseur : l'oublier fait repartir le tour suivant du début plutôt que de
    rester bloqué. `None` : rien à changer (une source rejouée depuis le cache, écartée
    faute de clé, ou en erreur sans reprise en cours).
    """
    if outcome == "cut_by_budget":
        return "save" if has_position else None
    if outcome in ("ok", "empty"):
        return "clear"
    if outcome == "error" and resumed and not has_position:
        return "clear"
    return None


def _load_fetch_cursors(scenario_id: str, query_hash: str) -> dict[str, dict]:
    """{fetcher: {position, fetched, rounds}} pour CETTE requête et CE scénario.

    La clé porte le hash de la requête : changer la requête invalide la position, puisque
    le curseur d'une base ne veut rien dire pour une autre question."""
    out: dict[str, dict] = {}
    try:
        with engine.connect() as _c:
            rows = _c.execute(text(
                "SELECT fetcher, position, fetched, rounds FROM source_fetch_cursor "
                "WHERE scenario_id = :s AND query_hash = :h"
            ), {"s": scenario_id, "h": query_hash}).mappings().all()
        for r in rows:
            out[r["fetcher"]] = {"position": r["position"], "fetched": int(r["fetched"] or 0),
                                 "rounds": int(r["rounds"] or 0)}
    except Exception as _e:                                  # noqa: BLE001
        logger.warning(f"_load_fetch_cursors: {_e}")
    return out


def _save_fetch_cursor(scenario_id: str, fetcher: str, query_hash: str,
                       position: Any, fetched: int, rounds: int) -> None:
    try:
        with engine.begin() as _c:
            _c.execute(text("""
                INSERT INTO source_fetch_cursor (scenario_id, fetcher, query_hash, position,
                                                 fetched, rounds, updated_at)
                VALUES (:s, :f, :h, :p, :n, :r, NOW())
                ON CONFLICT (scenario_id, fetcher, query_hash) DO UPDATE
                SET position = :p, fetched = :n, rounds = :r, updated_at = NOW()
            """), {"s": scenario_id, "f": fetcher, "h": query_hash,
                   "p": None if position is None else str(position)[:2000],
                   "n": int(fetched or 0), "r": int(rounds or 0)})
    except Exception as _e:                                  # noqa: BLE001
        logger.warning(f"_save_fetch_cursor {fetcher}: {_e}")


def _clear_fetch_cursor(scenario_id: str, fetcher: str | None = None,
                        query_hash: str | None = None) -> None:
    """Oublier la position : la source a fini de lire, ou la requête a changé."""
    try:
        where = ["scenario_id = :s"]
        params: dict[str, Any] = {"s": scenario_id}
        if fetcher:
            where.append("fetcher = :f")
            params["f"] = fetcher
        if query_hash:
            where.append("query_hash = :h")
            params["h"] = query_hash
        with engine.begin() as _c:
            _c.execute(text("DELETE FROM source_fetch_cursor WHERE " + " AND ".join(where)), params)
    except Exception as _e:                                  # noqa: BLE001
        logger.warning(f"_clear_fetch_cursor: {_e}")


def _source_query_hash(query: str, filters: dict | None, max_results: int,
                       mode: str = "standard") -> str:
    """Clé de cache : même requête, mêmes filtres, même plafond par source, même mode.

    Le mode n'entre dans la clé que s'il n'est pas `standard`, celui des recherches d'avant
    le mode exhaustif : une recherche, désormais toujours exhaustive, ne rejoue jamais la
    réponse plafonnée d'une recherche standard."""
    import hashlib
    _p = {"q": (query or "").strip(), "f": filters or {}, "m": int(max_results)}
    if mode and mode != "standard":
        _p["x"] = str(mode)
    payload = json.dumps(_p, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:40]


def _load_source_cache(query_hash: str, ttl_s: int | None = None) -> dict[str, dict]:
    """{fetcher: {links: [[doc_id, source, boolean_native], ...], n_records, totals, meta,
    age_s}} pour les entrées plus fraîches que le TTL. Vide (jamais d'erreur) si la table
    manque. `totals` : {source: total annoncé par l'API}, {} pour une entrée antérieure à la
    colonne ; `meta` : ce que la carte doit encore en dire (les tranches d'années de
    PubMed), {} de même."""
    ttl = SOURCE_CACHE_TTL_S if ttl_s is None else ttl_s
    if ttl <= 0:
        return {}
    out: dict[str, dict] = {}
    try:
        rows = []
        with engine.connect() as _c:
            # Du plus récent au plus ancien schéma : une colonne pas encore là n'empêche pas
            # de rejouer les liens.
            for _cols in ("totals, meta", "totals, NULL AS meta", "NULL AS totals, NULL AS meta"):
                try:
                    rows = _c.execute(text(
                        f"SELECT fetcher, links, n_records, {_cols}, "
                        "EXTRACT(EPOCH FROM (NOW() - fetched_at)) AS age_s "
                        "FROM source_query_cache WHERE query_hash = :h"
                    ), {"h": query_hash}).mappings().all()
                    break
                except Exception:                        # noqa: BLE001 - colonne pas encore là
                    _c.rollback()
        for r in rows:
            age = float(r["age_s"] or 0)
            if age > ttl:
                continue
            links = r["links"] if isinstance(r["links"], list) else json.loads(r["links"] or "[]")
            totals = r["totals"] if isinstance(r["totals"], dict) else json.loads(r["totals"] or "{}")
            meta = r["meta"] if isinstance(r["meta"], dict) else json.loads(r["meta"] or "{}")
            out[r["fetcher"]] = {"links": [tuple(x) for x in links if isinstance(x, list) and len(x) == 3],
                                 "n_records": int(r["n_records"] or 0), "age_s": age,
                                 "totals": {str(k): int(v) for k, v in (totals or {}).items()
                                            if isinstance(v, (int, float)) and not isinstance(v, bool)},
                                 "meta": meta if isinstance(meta, dict) else {}}
    except Exception as _e:                                  # noqa: BLE001
        logger.warning(f"_load_source_cache: {_e}")
    return out


def _save_source_cache(fetcher: str, query_hash: str, query: str, links: list,
                       totals: dict | None = None, meta: dict | None = None) -> None:
    try:
        with engine.begin() as _c:
            _c.execute(text("""
                INSERT INTO source_query_cache (fetcher, query_hash, query, links, n_records, totals,
                                                meta, fetched_at)
                VALUES (:f, :h, :q, CAST(:l AS jsonb), :n, CAST(:t AS jsonb), CAST(:m AS jsonb), NOW())
                ON CONFLICT (fetcher, query_hash) DO UPDATE
                SET links = CAST(:l AS jsonb), n_records = :n, query = :q,
                    totals = CAST(:t AS jsonb), meta = CAST(:m AS jsonb), fetched_at = NOW()
            """), {"f": fetcher, "h": query_hash, "q": (query or "")[:4000],
                   "l": json.dumps([list(x) for x in links]), "n": len(links),
                   "t": json.dumps(dict(totals or {})), "m": json.dumps(dict(meta or {}))})
    except Exception as _e:                                  # noqa: BLE001
        logger.warning(f"_save_source_cache {fetcher}: {_e}")
from .scenarios import _scenario_counts, _user_scenario_pipeline_jobs, _user_scenario_populate_jobs

def _run_user_scenario_populate(
    scenario_id: str,
    query: str,
    filters: dict,
    # LE plafond par source de l'application, et non 500 : chaque appelant passait
    # LIVE_MAX_PER_SOURCE, et un appelant nouveau qui l'aurait omis aurait reçu, en
    # silence, un corpus quatre fois plus petit par source.
    max_results: int = LIVE_MAX_PER_SOURCE,
    _pipeline_callback=None,
    include_live: bool = True,
    lang: str | None = None,
    force_live: bool = False,
    auto_pipeline: bool = True,
) -> int:
    """
    Construit le corpus d'un scénario = résultat de la REQUÊTE BOOLÉENNE sur
    (base locale ∪ articles récupérés en direct). Les sources live ne servent qu'à
    ENRICHIR la base ; l'appartenance au corpus est ensuite décidée UNIQUEMENT par
    la correspondance booléenne (_boolean_corpus_ids) - la même que la recherche.
    Plafond : LIVE_MAX_PER_SOURCE articles par source. include_live=False = base
    locale seulement. Retourne le nombre total d'articles ingérés.

    Toute recherche est EXHAUSTIVE (la recherche « standard », qui gardait les notices les
    plus pertinentes de chaque base, a été retirée : avec le plafond de production, elle ne
    différait plus que par ce qu'elle laissait de côté). Chaque base qui applique le booléen
    renvoie TOUT ce qu'elle apparie (jusqu'à EXHAUSTIVE_MAX_PER_SOURCE, ou
    LIVE_MAX_PER_SOURCE s'il est plus haut : un garde-fou, non un échantillon), avec le
    budget EXHAUSTIVE_FEDERATION_BUDGET (POPULATE_FEDERATION_BUDGET s'il est plus long) ;
    Crossref, qui classe des mots-clés au lieu d'appliquer le booléen et dont la liste
    complète est l'index entier, garde ses RANKED_PASS_MAX plus pertinentes ; OpenAlex rend
    toutes ses correspondances sur titre et résumé (sa recherche par défaut lit le texte
    intégral : 43 181 notices sur la question HPAI, contre 1 648 pour PubMed), plus ses
    RANKED_PASS_MAX meilleures correspondances en texte intégral ; DOAJ reçoit le booléen
    entier découpé en requêtes courtes, qu'elle sait paginer, et chaque base que la longueur
    d'URL ou le nombre d'opérateurs limite (OpenAlex, ClinicalTrials.gov, arXiv, OpenAIRE)
    le reçoit découpé de même au lieu de sa forme réduite, quand il tient en
    SPLIT_MAX_PARTS requêtes.
    `auto_pipeline=False` : ne pas enchaîner le pipeline complet après la recherche.
    """
    from .relevance import _backfill_title_abstract_chunks, _run_cross_encoder_rerank, _run_semantic_rerank_inline  # lazy: relevance is loaded after this module
    import time as _time
    import xml.etree.ElementTree as ET
    import requests as _requests
    import math
    import threading
    # NB : sur Python <3.11, as_completed() lève concurrent.futures.TimeoutError,
    # qui n'EST PAS le TimeoutError natif. On l'importe explicitement pour pouvoir
    # l'attraper (cf. bloc fédération plus bas).
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout

    # Plafond par source identique pour recherche et corpus (déterminisme).
    max_results = min(max_results, LIVE_MAX_PER_SOURCE)
    # Une seule recherche, exhaustive : le plafond des bases qui appliquent le booléen est
    # un garde-fou. Le mode reste écrit sur la carte PRISMA et dans la clé du cache des
    # sources, où les recherches standard d'avant gardent le leur.
    _search_mode = "exhaustive"
    _cap_strategy = max(int(max_results), EXHAUSTIVE_MAX_PER_SOURCE)
    # Les passages classés par pertinence (Crossref, le texte intégral d'OpenAlex) gardent
    # leur plafond propre, quel que soit `max_results` : cf. RANKED_PASS_MAX.
    _cap_ranked = max(1, min(int(max_results), RANKED_PASS_MAX))
    _fed_budget = max(POPULATE_FEDERATION_BUDGET, EXHAUSTIVE_FEDERATION_BUDGET)
    # Ce que la carte PRISMA doit dire de ce run, source par source : interrogée sur titre
    # et résumé seulement, gardée dans un ordre qui n'est pas la pertinence, ou interrogée
    # avec le booléen découpé.
    _title_abstract_sources: set[str] = set()
    _unranked_sources: set[str] = set()
    _split_queries: dict[str, list[str]] = {}
    # Recherche SANS préprints (filtre `exclude_preprints` du scénario) : les serveurs de
    # préprints ne sont pas interrogés, PubMed et Europe PMC les excluent dans la requête,
    # et ceux que rendent les autres bases sont retirés du corpus avant le screening.
    _no_preprints = excludes_preprints(filters)
    _preprints_at_source: list[str] = []

    # Garde-temps partagé entre les boucles de pagination des sources lentes
    # (OpenAlex/Crossref/EuropePMC). Quand le budget fédération est dépassé, elles
    # s'arrêtent d'elles-mêmes au lieu de continuer à paginer jusqu'à 2000 résultats
    # en arrière-plan (résultats qui seraient de toute façon écartés par le filtre
    # booléen final). Réglé juste avant le lancement de la fédération.
    _fed_deadline = [float("inf")]

    if _pipeline_callback is None:
        _user_scenario_populate_jobs[scenario_id] = {
            "status": "running", "ingested": 0, "errors": 0, "total_found": 0,
            # `phase` reflète l'ÉTAPE RÉELLE du backend (et non un minuteur côté
            # client) : local → federation → scoring → rerank → done. `rerank_status`
            # suit le cross-encoder qui tourne en arrière-plan après l'affichage.
            "phase": "local", "rerank_status": "idle",
            "search_mode": _search_mode,
            "exclude_preprints": _no_preprints,
            "sources": {
                "db_cache": 0, "pubmed": 0, "openalex": 0, "crossref": 0,
                "europepmc": 0, "preprint": 0, "semantic_scholar": 0, "doaj": 0,
                "clinicaltrials": 0, "core": 0, "arxiv": 0, "openaire": 0,
            }
        }

    def _set_phase(_phase: str, **extra):
        """Met à jour la phase réelle du job (no-op pour le pipeline complet)."""
        if _pipeline_callback is None:
            job = _user_scenario_populate_jobs.get(scenario_id)
            if job is not None:
                job["phase"] = _phase
                job.update(extra)

    ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    EMAIL = os.getenv("PUBMED_EMAIL", "literev@example.com")
    BATCH_SIZE = 200

    # Compteurs partagés thread-safe
    _counter_lock = threading.Lock()
    _ingested_total = [0]
    _errors_total = [0]
    _bool_native_ids: set = set()   # SOURCE-UNION - voir _link_to_scenario
    # Comptabilité PRISMA « identification » : enregistrements ramenés PAR SOURCE (un
    # article renvoyé par trois sources = trois enregistrements) et documents DISTINCTS
    # derrière eux. Leur différence est le nombre de doublons - que la dédup à
    # l'ingestion (index uniques) absorbait jusqu'ici sans laisser de trace.
    _ident_records: dict[str, int] = {}
    _ident_docs: set = set()
    # ③ Santé de la fédération, pour décider si un corpus PEUT rétrécir / se vider.
    #  • _source_errors  : nb de sources ayant échoué (except top-level d'un fetcher) ;
    #  • _fed_incomplete : True si le budget fédération a été dépassé (sources coupées).
    # Un résultat, vide ou plus petit, n'est fiable que si la fédération a réussi (aucune
    # source en erreur ET pas de coupure). Sinon les liens d'avant la recherche restent
    # (cf. l'assemblage final), pour ne pas les effacer sur une panne passagère d'une source.
    _source_errors = [0]
    _fed_incomplete = [False]
    # GEL du corpus : dès que l'assemblage final commence, un enregistrement qui arrive
    # encore d'une source lente (pages au-delà du budget fédération, l'executor n'attend
    # pas) est ingéré dans la base mais n'est NI compté NI lié à CETTE recherche. Avant,
    # les pages tardives des sources booléennes-natives continuaient d'insérer des liens
    # après le nettoyage du corpus et le calcul des chiffres PRISMA : 3 602 articles au
    # corpus pour 2 491 « passés au screening », des documents tardifs exempts de la
    # règle « sans résumé » et de la dédup. Le verrou couvre le test ET l'insertion :
    # aucun lien ne peut se glisser entre le gel et l'assemblage.
    _corpus_lock = threading.Lock()
    _corpus_frozen = [False]
    # Cache des sources : chaque fetcher tourne dans son thread ; on retient, par fetcher,
    # les liens qu'il produit (document, source, booléen-natif) pour les rejouer à la
    # prochaine relance de la même requête, et les fetchers tombés en erreur.
    _tls = threading.local()
    _run_links: dict[str, list] = {}
    # Sources en ÉCHEC, et la raison de chacune : l'issue `error` était servie nue, donc
    # la carte PRISMA disait « échec » sans dire de quoi.
    _fetcher_errors: dict[str, str] = {}
    # Sources écartées SANS appel réseau, et pourquoi. Une source sans clé d'API n'a pas
    # répondu « rien » : elle n'a pas été interrogée. Les deux se lisaient pareil dans la
    # ligne de couverture, et le cache des sources mémorisait même la non-interrogation
    # comme une réponse vide valable douze heures.
    _fetcher_skipped: dict[str, str] = {}
    # Sources ARRÊTÉES EN COURS de pagination par le budget. Elles ont répondu, mais pas
    # jusqu'au bout : leur compteur est un plancher, pas un total, et surtout un
    # « 0 article » de leur part ne dit RIEN sur la littérature. Sans ce marquage, une
    # source coupée revenait avec 0 et sans erreur, donc d'issue `empty` : « rien dans la
    # littérature » pour une source qu'on avait cessé de lire.
    _fetcher_cut: set[str] = set()
    # Où chaque source en est de sa lecture, {fetcher: (position, notices lues en tout)},
    # et d'où elle repart, lu en base au début de la fédération. Une source coupée garde
    # sa position ; une source qui va au bout l'efface.
    _fetch_pos: dict[str, tuple] = {}
    _resume_state: dict[str, dict] = {}
    # Fetchers dont la réponse ne doit pas être mise en cache même complète : OpenAlex
    # quand son filtre titre+résumé a été refusé, pour que la relance le retente.
    _no_cache: set[str] = set()
    # Le TOTAL VRAI que chaque API annonce pour la requête (le compteur du site), quand
    # elle en annonce un. Au plafond par source, une ligne « pubmed 2 000 » se lisait comme
    # un total ; c'est un plancher, et le vrai nombre est ici. Cinq sources l'atteignaient
    # sur le scénario de contrôle de production sans que rien ne le dise.
    _source_totals: dict[str, int] = {}
    # Les mêmes totaux, rangés par fetcher, pour être mémorisés avec ses liens dans le cache
    # des sources et rejoués avec eux.
    _fetcher_totals: dict[str, dict[str, int]] = {}

    def _note_total(source: str, total) -> None:
        try:
            _t = int(total)
        except (TypeError, ValueError):
            return
        if _t >= 0:
            with _counter_lock:
                _source_totals[source] = max(_t, _source_totals.get(source, 0))
                _f = getattr(_tls, "fetcher", None)
                if _f:
                    _ft = _fetcher_totals.setdefault(_f, {})
                    _ft[source] = max(_t, _ft.get(source, 0))

    def _mark_source_error(reason: object = None):
        """Cette source a ÉCHOUÉ, et pourquoi.

        L'issue `error` était servie nue : la carte PRISMA affichait « openalex : échec »
        sans rien de plus, et diagnostiquer demandait de retrouver la ligne de journal du
        serveur. La raison est désormais enregistrée avec l'issue et servie avec elle."""
        _source_errors[0] += 1
        _f = getattr(_tls, "fetcher", None)
        if _f:
            with _counter_lock:
                _fetcher_errors[_f] = (redact_keys(reason)[:200] if reason is not None
                                       else _fetcher_errors.get(_f) or "")

    def _mark_source_skipped(reason: str):
        """Cette source n'a pas été interrogée. À dire, pas à compter comme vide."""
        _f = getattr(_tls, "fetcher", None)
        if _f:
            with _counter_lock:
                _fetcher_skipped[_f] = reason

    def _budget_exhausted() -> bool:
        """Le budget de fédération est-il écoulé ? Si oui, la source est MARQUÉE coupée.

        Les treize boucles de pagination testaient le budget et sortaient en silence.
        Deux conséquences : l'issue annoncée de la source devenait `ok`/`empty` comme si
        elle avait fini de répondre, et `_fed_incomplete` restait faux tant que
        `as_completed` ne débordait pas lui-même, donc le corpus était autorisé à se
        VIDER sur une fédération pourtant partielle. Une fédération incomplète ne retire
        plus aucun lien d'avant la recherche, vide ou non (cf. l'assemblage final).
        """
        if _time.time() < _fed_deadline[0]:
            return False
        _fed_incomplete[0] = True       # fetch partiel : le corpus ne peut pas se VIDER
        _f = getattr(_tls, "fetcher", None)
        if _f:
            with _counter_lock:
                _fetcher_cut.add(_f)
        return True

    # ── La position de lecture, pour que la reprise continue au lieu de recommencer ──
    # Chaque boucle de pagination dit où elle en est ; une source coupée garde sa position
    # (voir `_save_fetch_cursor`), une source qui va au bout l'efface. Les boucles lisent
    # `_tls.fetcher` comme `_budget_exhausted`, donc elles n'ont pas à se nommer.
    def _note_pos(position: Any, fetched: int) -> None:
        _f = getattr(_tls, "fetcher", None)
        if not _f:
            return
        with _counter_lock:
            _fetch_pos[_f] = (position, int(fetched or 0))

    def _resume(default: Any = None) -> Any:
        """La position d'où repartir, ou `default` pour une première lecture."""
        _f = getattr(_tls, "fetcher", None)
        got = (_resume_state.get(_f) or {}).get("position") if _f else None
        return default if got in (None, "") else got

    def _resume_fetched() -> int:
        """Ce que cette source a DÉJÀ rapporté lors des tours précédents, pour que le
        plafond par source compte la lecture entière et non le seul tour en cours."""
        _f = getattr(_tls, "fetcher", None)
        return int((_resume_state.get(_f) or {}).get("fetched") or 0) if _f else 0

    # Un 429 se réessaie un nombre BORNÉ de fois. Trois fetchers bouclaient sans limite
    # (`if 429: sleep(2); continue`) : Semantic Scholar répond 429 à la moindre rafale
    # sans clé d'API, la boucle tenait donc son fil jusqu'au budget (180 s par défaut),
    # frappait l'API une quarantaine de fois de plus - ce qui durcit la limite - et en
    # sortait avec zéro article SANS marquer d'erreur. Toute recherche neuve y perdait
    # trois minutes, et le panneau PRISMA lisait « Semantic Scholar : aucun résultat ».
    # Cinq tentatives par page, 2 s puis 4, 6, 8 s (20 s au plus), et non trois pour 6 s :
    # mesuré depuis le serveur de production, le quota anonyme de Semantic Scholar répond
    # 200 puis 429 deux secondes plus tard. Sur HPAI_last, la première recherche a perdu
    # la seconde page (977 notices puis erreur) et la seconde l'a obtenue (1 961) : la
    # même requête, à six minutes de patience près. Une clé d'API reste le vrai remède ;
    # chaque fil de source attend seul, les onze autres ne l'attendent pas.
    _RATE_LIMIT_ATTEMPTS = 5
    _RATE_LIMIT_MAX_SLEEP_S = 12.0

    def _wait_out_rate_limit(resp, attempt: int, source: str, keyed: bool = False) -> None:
        """Attendre un 429 une fois de plus, ou LEVER quand il n'y a plus lieu d'attendre.

        `attempt` compte les 429 déjà reçus pour cette page. Lève au-delà de
        `_RATE_LIMIT_ATTEMPTS`, et lève aussi quand le budget ne laisse plus le temps
        d'une attente : dans les deux cas la source finit en `error`, nommée dans la
        ligne de couverture, au lieu de passer pour vide. `keyed` : l'appel porte la clé
        du serveur, et les attentes sont celles de `KEYED_429_WAITS_S`.
        """
        _waits = KEYED_429_WAITS_S if keyed else ()
        if attempt + 1 >= (len(_waits) + 1 if keyed else _RATE_LIMIT_ATTEMPTS):
            raise RuntimeError(f"{source} : 429 après {len(_waits) + 1 if keyed else _RATE_LIMIT_ATTEMPTS} "
                               "tentatives " + ("malgré la clé d'API (limite de débit de la source)" if keyed
                                                else "(limite de débit ; une clé d'API la relèverait)"))
        try:
            _after = float((getattr(resp, "headers", None) or {}).get("Retry-After") or 0)
        except (TypeError, ValueError):
            _after = 0.0
        _delay = (max(_after, _waits[attempt]) if keyed
                  else min(max(_after, 2.0 * (attempt + 1)), _RATE_LIMIT_MAX_SLEEP_S))
        _delay = min(_delay, max(0.0, _fed_deadline[0] - _time.time()))
        if _delay <= 0:
            raise RuntimeError(f"{source} : 429 et budget de fédération épuisé")
        _time.sleep(_delay)

    def _with_retries(send, source: str, retry_429: bool = True, min_wait: float = 0.0):
        """Une requête de page, réessayée sur délai dépassé, erreur de connexion, 5xx (et
        429 si `retry_429`), FETCH_RETRY_ATTEMPTS fois au plus (cf. sa note en tête du
        module). Rend la dernière réponse, que l'appelant juge comme avant (son
        `raise_for_status` nomme le statut), ou lève la dernière erreur réseau. N'attend
        jamais au-delà du budget de fédération : quand l'attente n'y tient plus, la
        tentative suivante n'a pas lieu. `min_wait` : l'espacement qu'une API exige
        (arXiv, 3 s). Semantic Scholar et CORE gardent leur propre attente des 429
        (`_wait_out_rate_limit`) : `retry_429=False`."""
        resp, last_exc = None, None
        for attempt in range(max(1, int(FETCH_RETRY_ATTEMPTS))):
            resp = None
            try:
                resp = send()
            except _requests.RequestException as _re_exc:
                last_exc = _re_exc
            if resp is not None:
                _code = int(getattr(resp, "status_code", 200) or 200)
                if not (_code >= 500 or (retry_429 and _code == 429)):
                    return resp
            if attempt + 1 >= max(1, int(FETCH_RETRY_ATTEMPTS)):
                break
            _backoff = FETCH_RETRY_BACKOFF_S
            _wait = max(float(min_wait), float(_backoff[min(attempt, len(_backoff) - 1)]))
            if resp is not None:
                try:
                    _after = float((getattr(resp, "headers", None) or {}).get("Retry-After") or 0)
                except (TypeError, ValueError):
                    _after = 0.0
                _wait = max(_wait, min(_after, FETCH_RETRY_AFTER_CAP_S))
            # Le temps qui reste au budget, comme `_wait_out_rate_limit` le mesure : une
            # attente qui le dépasserait n'a pas lieu, et la source finit sur la dernière
            # réponse (son statut nommé sur la carte), pas sur une attente coupée.
            _left = max(0.0, _fed_deadline[0] - _time.time())
            if _wait > _left:
                break
            logger.info(f"Populate {scenario_id} [{source}]: tentative {attempt + 2} dans {_wait:.0f} s "
                        f"({'HTTP ' + str(resp.status_code) if resp is not None else last_exc})")
            _time.sleep(_wait)
        if resp is not None:
            return resp
        raise last_exc if last_exc is not None else RuntimeError(f"{source} : aucune réponse")

    def _part_resume() -> dict:
        """La position retenue pour une source qui lit le booléen morceau par morceau."""
        try:
            return json.loads(_resume("") or "") or {}
        except Exception:                                    # noqa: BLE001 - position illisible
            return {}

    def _run_fetcher(fn):
        _tls.fetcher = fn.__name__
        try:
            return fn()
        finally:
            _tls.fetcher = None

    def _link_to_scenario(doc_id, boolean_native=False, source=None):
        with _corpus_lock:
            if _corpus_frozen[0]:
                return None            # arrivé après l'assemblage : pas dans cette recherche
            return _link_to_scenario_unlocked(doc_id, boolean_native, source)

    def _link_to_scenario_unlocked(doc_id, boolean_native=False, source=None):
        # Comptabilité PRISMA : un enregistrement par source qui a renvoyé l'article,
        # que la ligne soit nouvelle ou déjà connue : c'est justement le recoupement
        # entre sources (et avec la base locale) qui fait le doublon.
        if doc_id is not None and source:
            with _counter_lock:
                _ident_records[source] = _ident_records.get(source, 0) + 1
                _ident_docs.add(doc_id)
                _f = getattr(_tls, "fetcher", None)
                if _f:
                    _run_links.setdefault(_f, []).append((int(doc_id), source, bool(boolean_native)))
        # NE LIE PLUS pendant la fédération : ingérer un article live ne l'ajoute PAS
        # d'office au corpus. L'appartenance est recalculée après ingestion via la
        # correspondance booléenne - sinon le corpus gonflait avec des résultats live
        # (mots-clés) ne correspondant pas à la requête.
        # EXCEPTION - SOURCE-UNION : les sources BOOLÉENNES-NATIVES (PubMed, Europe PMC,
        # préprints EPMC) ont appliqué la VRAIE requête booléenne (MeSH / texte intégral).
        # On mémorise leurs docs pour les INCLURE directement dans le corpus, sans les
        # re-filtrer localement (le re-filtrage titre+résumé perdait leurs correspondances
        # MeSH sans phrase littérale dans le résumé - d'où « 109 PubMed → 6 »). Les
        # sources par MOTS-CLÉS, elles, restent re-filtrées (leur tri est lâche).
        if boolean_native and doc_id is not None:
            _bool_native_ids.add(doc_id)
            # Lien INCRÉMENTAL : un doc booléen-natif appartient au corpus par source-union
            # (confirmé tel quel à l'assemblage final). On le lie DÈS l'ingestion pour que
            # le compteur du corpus grandisse EN DIRECT pendant la recherche (172 → … →
            # total), au lieu de sauter d'un coup à la fin. Score NULL ici → renseigné au
            # scoring. best-effort : l'assemblage final refait l'union de toute façon.
            # Sans préprints, un préprint n'est pas lié : le compteur en direct ne doit pas
            # monter de notices que l'assemblage retirera.
            try:
                with engine.begin() as _lc:
                    _lc.execute(text(
                        "INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                        "SELECT d.id, :s, NULL FROM literature_document d WHERE d.id = :d"
                        + (f" AND NOT {preprint_sql('d')}" if _no_preprints else "")
                        + " ON CONFLICT (document_id, scenario_id) DO NOTHING"),
                        {"d": doc_id, "s": scenario_id})
            except Exception:
                pass
        return None

    def _inc(source_name, count=1, err=0):
        with _counter_lock:
            _ingested_total[0] += count
            _errors_total[0] += err
            # Après le gel, une page tardive d'une source coupée insère encore ses lignes
            # en base, mais elle ne fait plus partie de cette recherche : le corpus et les
            # chiffres PRISMA l'ignorent déjà (cf. _corpus_frozen). Le job servi à
            # l'interface l'ignorait pas : `ingested` et `sources` continuaient de monter
            # après la publication, jusqu'à afficher une ventilation par source plus
            # grande que le total qu'elle détaille (relecture adversariale, reproduit).
            if _pipeline_callback is None and not _corpus_frozen[0]:
                job = _user_scenario_populate_jobs[scenario_id]
                job["ingested"] = _ingested_total[0]
                # Remonter les erreurs dans l'état du job : sans cela, errors=0
                # masquait toute perte de données par source (échec silencieux).
                job["errors"] = _errors_total[0]
                job.setdefault("errors_by_source", {})
                if err:
                    job["errors_by_source"][source_name] = \
                        job["errors_by_source"].get(source_name, 0) + err
                job["sources"][source_name] = job["sources"].get(source_name, 0) + count

    # Le corpus tel qu'il est AVANT cette recherche, relevé avant qu'elle ne lie quoi que
    # ce soit. Une fédération qui n'aboutit pas (une source en échec, ou coupée par le
    # budget) ne dit rien des liens qu'elle n'a pas ramenés : l'assemblage final garde
    # alors ceux-ci, et seulement ceux-ci (les liens que cette recherche a posés en route
    # restent soumis à sa propre requête).
    _previous_links: list[int] | None = None
    try:
        with engine.connect() as _plc:
            _previous_links = [int(x) for x in _plc.execute(text(
                "SELECT document_id FROM article_scenarios WHERE scenario_id = :sid"),
                {"sid": scenario_id}).scalars().all()]
    except Exception as _e_prev:
        logger.warning(f"Populate {scenario_id}: corpus précédent non relevé ({_e_prev})")

    # ── Étape 0 : Linking depuis la base locale (séquentiel, rapide) ─────────
    # Le CORPUS est défini par une correspondance LEXICALE (requête booléenne),
    # indépendante du seuil sémantique : base locale ∪ nouvelles références live.
    # Le seuil sémantique n'intervient QUE dans la page scénario pour sélectionner
    # le sous-ensemble pertinent (_get_above_threshold_articles).
    local_linked = 0
    # Lié ICI et non dans le try : les chiffres PRISMA, écrits bien plus bas dans un AUTRE
    # bloc, les lisent. Assignés seulement dans le try, une exception avant leur ligne les
    # laissait non liés, et le NameError, avalé par le try du bloc PRISMA, faisait sauter
    # l'enregistrement des chiffres sans trace.
    _kw_fallback: list[str] = []
    # Les compteurs par source au gel du corpus (cf. le bloc du gel) ; None tant qu'il n'a
    # pas eu lieu, et alors le statut final relit le dict vivant, comme avant.
    _sources_snapshot: dict | None = None
    _plain_q = query
    _fallback_q = query
    _openaire_q = query
    _fallback_queries: dict[str, str] = {}
    # Liés ICI, hors du try, pour la même raison que `_kw_fallback` : les fetchers les lisent.
    _doaj_parts: list[str] = []                 # DOAJ : le booléen découpé
    # Les autres bases que l'URL limite : le booléen découpé.
    _openalex_parts: list[str] = []
    _ct_parts: list[str] = []
    _arxiv_parts: list[str] = []
    _openaire_parts: list[str] = []             # chaque morceau ≤ OPENAIRE_MAX_OPERATORS opérateurs
    _s2_bool_q: list = [None]                   # la forme bulk de Semantic Scholar, ou None
    _s2_parts: list[str] = []                   # son booléen découpé, union = le booléen
    _s2_native: list = [True]                   # faux dès que sa requête est réduite
    _s2_reduced: list = [False]                 # ce que la carte doit dire de cette source
    # Les facettes OU qu'aucune base n'a reçues (requête trop longue), et la correspondance
    # locale qui aurait atteint sa borne : dits dans les chiffres de la recherche.
    _facets_skipped: list[str] = []
    _local_capped = [False]
    # Le plafond RÉELLEMENT appliqué à une source quand il diffère du plafond commun :
    # l'endpoint classique de Semantic Scholar (offset + limit ≤ 1 000).
    _source_caps_run: dict[str, int] = {}
    # PubMed : sa requête AVANT les bornes d'années et l'exclusion des préprints, et ces
    # bornes. Le découpage par années (au-delà de 10 000 notices) réécrit la requête tranche
    # par tranche à partir d'elles. Liés ici, hors du try, parce que `_fetch_pubmed` les lit.
    _pubmed_base = query
    _pubmed_years = pubmed_year_bounds(filters)
    # Les tranches d'années de PubMed, quand il a fallu découper : dites sur la carte PRISMA.
    _year_slices: dict[str, list[str]] = {}
    _year_slices_capped: dict[str, dict[str, int]] = {}

    def _pubmed_term(lo: int | None, hi: int | None) -> str:
        """La requête PubMed d'une tranche d'années (None, None : toutes les années du
        scénario), l'exclusion des préprints en dernier sur CHAQUE tranche."""
        _t = pubmed_years_term(_pubmed_base, lo, hi)
        return f"({_t}) NOT preprint[pt]" if _no_preprints else _t

    try:
        # Le corpus = résultat de la REQUÊTE BOOLÉENNE (générée par LLM). On
        # récupère search_strategy.general ; à défaut on la génère depuis la requête.
        _boolean = query
        _pubmed_q = query
        # Recherche multi-sous-requêtes : l'appartenance au corpus est l'union /
        # l'intersection des ensembles locaux de chaque sous-requête. La fédération
        # live reste pilotée par la stratégie booléenne de la requête synthétisée
        # (elle ne fait qu'ENRICHIR la base ; l'appartenance est recalculée après).
        _sub_queries: list[dict] = []
        _combinator = "union"
        try:
            with engine.connect() as _sc:
                _srow = _sc.execute(text(
                    "SELECT search_strategy, sub_queries, combinator FROM user_scenarios WHERE id = :sid"),
                    {"sid": scenario_id}).mappings().first()
            _strat = _srow["search_strategy"] if _srow else None
            _sub_queries = _normalize_sub_queries(_srow["sub_queries"]) if _srow else []
            if _srow and _srow["combinator"] in ("union", "intersection"):
                _combinator = _srow["combinator"]
            if isinstance(_strat, dict) and not _strategy_is_degraded(_strat, query):
                _boolean = _strat["general"]
                _pubmed_q = _strat.get("pubmed") or _strat["general"]
            else:
                # Absent ou dégradé (cache empoisonné pendant une panne quota) → régénérer.
                # Une stratégie saisie reste la requête saisie : une correction proposée
                # n'est jamais appliquée ici, ni enregistrée (l'écran de recherche la propose).
                _gen = _generate_search_strategy(query)
                _boolean = _gen.get("general") or query
                _pubmed_q = _gen.get("pubmed") or _boolean
                if not _strategy_is_degraded(_gen, query):
                    with engine.begin() as _sc2:
                        _sc2.execute(text("UPDATE user_scenarios SET search_strategy = CAST(:s AS jsonb) WHERE id = :id"),
                                     {"s": json.dumps(_strategy_to_store(_gen)), "id": scenario_id})
        except Exception as _be:
            logger.warning(f"Populate {scenario_id} boolean strategy: {_be}")
        # Facettes UNIES (OU) d'une recherche multi-facettes : fédérées AUSSI, en
        # élargissant les requêtes live (« principal OR facette »). Les facettes ET
        # restent re-matchées localement (sous-ensemble du principal).
        if _sub_queries:
            try:
                _boolean, _pubmed_q, _n_or = _widen_boolean_for_or_facets(_boolean, _pubmed_q, _sub_queries, _combinator,
                                                                          skipped=_facets_skipped)
                if _n_or:
                    logger.info(f"Populate {scenario_id}: fédération élargie à {_n_or} facette(s) OU - {_boolean[:160]!r}")
            except Exception as _we:                      # noqa: BLE001
                logger.warning(f"Populate {scenario_id} facettes OU: {_we}")
        # Variante par type de source (comme le font déjà les _live_fetch_*) :
        #  - _pubmed_q : booléen MeSH → sources proxyfiées PubMed (eutils)
        #  - _boolean  : booléen général → API qui acceptent les opérateurs (EuropePMC)
        #  - _plain_q  : mots-clés simples (sans opérateurs ni '?') → OpenAlex / Crossref / preprints
        # CORRECTIF : auparavant TOUTES les sources recevaient la requête BRUTE en
        # langage naturel (avec le '?'), d'où OpenAlex 400 et booléens Cochrane/
        # PROSPERO malformés → 0 article récupéré en direct.
        _plain_q = _plain_keywords(_boolean) or _plain_keywords(query) or query
        # ── Routage BOOLÉEN par source (docs vérifiées 2026) ────────────────────────
        # Beaucoup d'API acceptent un vrai booléen (AND/OR/NOT + parenthèses + guillemets),
        # pas seulement PubMed/EuropePMC : OpenAlex, DOAJ, CORE, ClinicalTrials, arXiv. On
        # leur envoie donc le booléen PORTABLE (tags de champ PubMed retirés) au lieu de
        # mots-clés aplatis, et on les traite en SOURCE-UNION. Repli mots-clés si pas de vrai
        # booléen (mode dégradé) ou si l'URL dépasse ~4 Ko (limite OpenAlex).
        # Europe PMC et ses preprints : SANS les tags de champ de PubMed, qu'elle apparie
        # comme du texte. 2 398 notices sans tags contre 3 avec, mesure faite sur la sonde
        # de production : les tags coutaient 99,9 % du rappel de cette source.
        _epmc_q = epmc_query(_boolean)
        _portable_bool = " ".join(_strip_field_tags(_boolean).split())
        _bool_is_real = bool(_portable_bool) and _looks_boolean(_portable_bool)
        # La forme PROPRE du booléen entier pour les API qui le reçoivent dans une URL
        # (OpenAlex, DOAJ, CORE, ClinicalTrials.gov) : rendue depuis l'arbre, guillemets
        # normalisés, sans étoile de troncature, exclusions comprises. Sur HPAI_last,
        # OpenAlex et DOAJ répondaient 400 au booléen portable recopié tel quel
        # (« "occupational*" », « "farm worker*" »), CORE et ClinicalTrials.gov l'acceptaient.
        _clean_bool = _clean_boolean(_portable_bool) if _bool_is_real else ""
        # La limite d'URL se mesure sur la requête ENCODÉE (URL_QUERY_MAX_ENCODED) : 1 200
        # caractères en clair n'en disaient rien, les guillemets et les parenthèses y
        # prenant trois octets chacun.
        _send_bool = bool(_clean_bool) and _url_encoded_len(_clean_bool) <= URL_QUERY_MAX_ENCODED
        # Au-delà de la limite (ou devant un NOT, pour arXiv) : la requête RÉDUITE, pas
        # huit mots. Même arbre, chaque bloc OU tronqué à ses premiers termes, les
        # exclusions retirées, jusqu'à tenir sous la limite. Le sac de mots perdait la
        # structure et chaque moteur en faisait autre chose, mesuré sur le premier run de
        # production après #326 : arXiv et CORE le lisaient en OU (2 000 notices chacun,
        # sur « exposure » ou « virus », toutes hors requête), OpenAlex, DOAJ et
        # ClinicalTrials.gov en ET de huit mots dont des variantes qui s'excluent
        # (« h5n1 » ET « h7n9 »). Les notices de ces sources restent ré-appariées en local
        # contre le booléen entier (boolean_native=False) ; le sac de mots ne sert plus
        # qu'à Crossref (texte libre) et quand il n'y a pas de booléen du tout.
        _short_bool = (_shorten_boolean(_portable_bool, URL_QUERY_MAX_ENCODED, measure=_url_encoded_len)
                       if _bool_is_real else "")
        _fallback_q = _short_bool or _plain_q
        _bool_query = _clean_bool if _send_bool else _fallback_q          # OpenAlex/DOAJ/CT
        # CORE reçoit sa requête dans le CORPS d'un POST JSON : aucune limite d'URL ne la
        # concerne, elle prend le booléen entier quelle que soit sa longueur, dans les deux
        # modes, et ses notices sont natives.
        _core_q, _core_native = (_clean_bool, True) if _clean_bool else (_bool_query, _send_bool)
        _arxiv_q, _arxiv_native = f"all:{_plain_q}", False                # arXiv : syntaxe dédiée
        if _bool_is_real:
            try:
                _ax = _boolean_to_arxiv(_parse_boolean_ast(_tokenize_boolean(_portable_bool)))
                if _ax and _url_encoded_len(_ax) <= URL_QUERY_MAX_ENCODED:
                    _arxiv_q, _arxiv_native = _ax, True
                else:
                    _ax_short = _shorten_boolean(_portable_bool, URL_QUERY_MAX_ENCODED,
                                                 render=_boolean_to_arxiv, measure=_url_encoded_len)
                    if _ax_short:
                        _arxiv_q = _ax_short
                    # Le booléen entier en morceaux qui tiennent dans l'URL, au lieu de
                    # synonymes en moins (impossible devant un NOT, qu'arXiv ne sait pas
                    # écrire seul).
                    if _ax:
                        _ax_parts = _split_boolean(_portable_bool, URL_QUERY_MAX_ENCODED,
                                                   render=_boolean_to_arxiv, measure=_url_encoded_len)
                        if _ax_parts:
                            _arxiv_parts[:] = _ax_parts
                            _arxiv_native = True
                            _split_queries["arxiv"] = list(_ax_parts)
            except Exception:
                pass
        # Booléen trop long pour l'URL : OpenAlex et ClinicalTrials.gov le reçoivent découpé
        # en requêtes dont l'union est le booléen (exclusions attachées à chaque morceau),
        # comme DOAJ, au lieu de sa forme réduite. Leurs notices sont alors natives. Au-delà
        # de SPLIT_MAX_PARTS morceaux, la forme réduite reste.
        if _bool_is_real and _clean_bool and not _send_bool:
            _url_parts = _split_boolean(_portable_bool, URL_QUERY_MAX_ENCODED, measure=_url_encoded_len)
            if _url_parts:
                _ct_parts[:] = _url_parts
                _split_queries["clinicaltrials"] = list(_url_parts)
        # OpenAlex refuse aussi une requête de TROP D'OPÉRATEURS, même courte : le morceau de
        # 60 opérateurs du scénario T2.4 exposition/transmission (400), quand ceux de 23 à 47
        # passaient. Ses morceaux sont donc bornés en longueur ET en opérateurs, et un booléen
        # qui tient dans l'URL mais dépasse OPENALEX_MAX_OPERATORS est découpé lui aussi.
        if _bool_is_real and _clean_bool and (
                not _send_bool or _count_operators(_clean_bool) > OPENALEX_MAX_OPERATORS):
            _oa_parts = _split_boolean(_portable_bool, URL_QUERY_MAX_ENCODED, measure=_url_encoded_len,
                                       max_operators=OPENALEX_MAX_OPERATORS, max_parts=OPENALEX_SPLIT_MAX_PARTS)
            if _oa_parts:
                _openalex_parts[:] = _oa_parts
                _split_queries["openalex"] = list(_oa_parts)
        # Les sources qui n'ont PAS reçu le booléen entier, à DIRE : PRISMA-S demande la
        # stratégie réellement soumise à chaque base, et le tableau d'identification
        # présentait leurs notices comme le produit d'une recherche booléenne. Un booléen
        # portable de plus de 1 200 caractères y basculait les six sources ci-dessous en
        # silence.
        # OpenAIRE n'accepte que QUATRE opérateurs logiques (« Too many logical operators
        # found. Max allowed is 4 », mesuré en production : 48 sur la requête recommandée,
        # 95 sur la requête HPAI entière). Toute stratégie booléenne lui valait donc un 400
        # depuis la migration vers l'API Graph v2 : elle reçoit sa propre réduction, bornée
        # en opérateurs, et ses notices sont ré-appariées en local.
        _openaire_q = (_shorten_boolean(_portable_bool, URL_QUERY_MAX_ENCODED,
                                        max_operators=OPENAIRE_MAX_OPERATORS, measure=_url_encoded_len)
                       if _bool_is_real else "") or _plain_q
        # Le booléen entier en morceaux de quatre opérateurs au plus chacun, quand il en faut
        # SPLIT_MAX_PARTS au plus (une stratégie de revue en demande d'ordinaire bien
        # davantage, et garde alors la réduction ci-dessus).
        if _clean_bool:
            _oaire_parts = _split_boolean(_portable_bool, URL_QUERY_MAX_ENCODED, measure=_url_encoded_len,
                                          max_operators=OPENAIRE_MAX_OPERATORS)
            if _oaire_parts:
                _openaire_parts[:] = _oaire_parts
                _split_queries["openaire"] = list(_oaire_parts)
        # DOAJ ne pagine pas au-delà de DOAJ_MAX_QUERY_CHARS (502 dès la page 2, mesuré en
        # production) : au-delà, sa propre réduction, et ses notices ré-appariées en local.
        _doaj_q, _doaj_native = _bool_query, _send_bool
        if _bool_is_real and len(_doaj_q) > DOAJ_MAX_QUERY_CHARS:
            _doaj_q = _shorten_boolean(_portable_bool, DOAJ_MAX_QUERY_CHARS) or _plain_q
            _doaj_native = False
        # Au lieu de synonymes en moins, le booléen ENTIER découpé en requêtes que DOAJ
        # pagine. Leur union est le booléen, exclusions comprises : les notices sont donc
        # natives, comme celles d'une requête entière.
        if _bool_is_real and not _doaj_native:
            _parts = _split_boolean(_portable_bool, DOAJ_MAX_QUERY_CHARS)
            if _parts:
                _doaj_parts[:] = _parts
                _doaj_native = True
                _split_queries["doaj"] = list(_parts)
        # Les sources qui ont reçu une forme RÉDUITE (ou des mots-clés), source par source.
        _kw_fallback = ([] if (_send_bool or _openalex_parts) else ["openalex"]) \
            + ([] if _doaj_native else ["doaj"]) \
            + ([] if _core_native else ["core"]) \
            + ([] if (_send_bool or _ct_parts) else ["clinicaltrials"]) \
            + ([] if _arxiv_native else ["arxiv"]) \
            + ([] if _openaire_parts else ["openaire"])
        # Semantic Scholar : un vrai booléen part vers l'endpoint BULK, qui ne classe pas par
        # pertinence (ordre des identifiants). Au plafond, ce qu'on y garde est un
        # sous-ensemble arbitraire : la carte doit le dire, que la réponse soit fraîche ou
        # rejouée depuis le cache.
        if _bool_is_real:
            try:
                _s2_bool_q[0] = _boolean_to_s2(_parse_boolean_ast(_tokenize_boolean(_portable_bool)))
            except Exception:                            # noqa: BLE001
                _s2_bool_q[0] = None
            # Sa requête part dans une URL, comme celle d'OpenAlex, et RIEN ne la mesurait :
            # la plus longue chaîne T2.4 (COVID exposition, 4 519 caractères) revenait en
            # 400 Bad Request et la source ne versait rien. Même traitement que les autres :
            # entière si elle tient, sinon RÉDUITE, et ses notices ne sont alors plus
            # natives puisqu'elles ne répondent plus au booléen entier (ré-appariement
            # local, qui applique la requête complète, exclusions comprises).
            if _s2_bool_q[0] and _url_encoded_len(_s2_bool_q[0]) > S2_MAX_QUERY_ENCODED:
                _enc = _url_encoded_len(_s2_bool_q[0])
                # D'ABORD le DÉCOUPAGE, comme pour OpenAlex et ClinicalTrials.gov : des
                # requêtes dont l'UNION est le booléen entier, donc rien de perdu et des
                # notices natives. La forme réduite ne vient qu'après, si le booléen ne
                # tient pas en SPLIT_MAX_PARTS morceaux.
                _sp = _split_boolean(_portable_bool, S2_MAX_QUERY_ENCODED,
                                     render=_boolean_to_s2, measure=_url_encoded_len)
                if _sp:
                    _s2_parts[:] = _sp
                    _split_queries["semantic_scholar"] = list(_sp)
                    _s2_bool_q[0] = None               # le lot des morceaux remplace la requête
                    logger.warning("Semantic Scholar : requete de %d caracteres encodes, "
                                   "au-dela de %d ; decoupee en %d requetes dont l'union est "
                                   "le booleen.", _enc, S2_MAX_QUERY_ENCODED, len(_sp))
                else:
                    _short_s2 = _shorten_boolean(_portable_bool, S2_MAX_QUERY_ENCODED,
                                                 render=_boolean_to_s2, measure=_url_encoded_len)
                    logger.warning(
                        "Semantic Scholar : requete de %d caracteres encodes, au-dela de %d ; "
                        "%s.", _enc, S2_MAX_QUERY_ENCODED,
                        "forme reduite envoyee" if _short_s2 else "repli sur l'endpoint classique")
                    _s2_bool_q[0] = _short_s2 or None
                    _s2_native[0] = False
                    if _short_s2:
                        _s2_reduced[0] = True
        if _s2_bool_q[0]:
            _unranked_sources.add("semantic_scholar")
        else:
            # L'endpoint classique (sans booléen, ou devant un NOT) ne rend pas plus de
            # 1 000 notices (offset + limit) : c'est SON plafond, et « plafonnée » se juge
            # contre lui. Comparé au plafond commun de 2 000, un total entre 1 001 et 2 000
            # passait pour complet.
            _source_caps_run["semantic_scholar"] = min(int(max_results), 1000)
        # OpenAlex : TOUTES ses correspondances sur titre et résumé (filtre
        # title_and_abstract.search, qui accepte le booléen), plus ses meilleures en texte
        # intégral. Sa recherche par défaut lit aussi le texte intégral, d'où 43 181 notices
        # sur HPAI.
        if _send_bool or _openalex_parts:
            _title_abstract_sources.add("openalex")
        # La requête RÉELLEMENT soumise à chacune : la même pour les sources à la limite
        # d'URL, la syntaxe `all:` pour arXiv, la forme à quatre opérateurs pour OpenAIRE,
        # la forme courte pour DOAJ.
        _fallback_queries = {s: _fallback_q for s in _kw_fallback if s not in ("arxiv", "openaire", "doaj")}
        if "arxiv" in _kw_fallback:
            _fallback_queries["arxiv"] = _arxiv_q
        if "doaj" in _kw_fallback:
            _fallback_queries["doaj"] = _doaj_q
        if "openaire" in _kw_fallback:
            _fallback_queries["openaire"] = _openaire_q
        logger.info(f"Populate {scenario_id}: requête réduite pour {', '.join(_kw_fallback) or 'aucune source'} "
                    f"(booléen propre de {len(_clean_bool)} caractères, {_url_encoded_len(_clean_bool)} encodés) ; "
                    f"soumis = « {_fallback_q} » ; OpenAIRE = « {_openaire_q} » ; "
                    f"découpé pour {', '.join(sorted(_split_queries)) or 'aucune source'}")
        # PubMed RECALL : la requête MeSH générée par le LLM (_pubmed_q) est parfois
        # BEAUCOUP plus étroite que le booléen général - p. ex. 35 résultats contre 306
        # pour le même booléen collé sur le site PubMed. On interroge donc PubMed sur
        # l'UNION « (MeSH) OR (booléen portable) » : on garde les correspondances MeSH
        # ET les correspondances de phrase (all-fields). Ne peut qu'AJOUTER des résultats.
        # TOUJOURS, quelle que soit la longueur : `_ncbi_get` envoie en POST ce qui ne tient
        # pas dans une URL. La borne de 1 900 caractères qui protégeait l'URL laissait à
        # PubMed la seule requête MeSH dès qu'une stratégie était longue (35 contre 306).
        if (_bool_is_real and _portable_bool and _pubmed_q
                and _portable_bool not in _pubmed_q):
            _pubmed_q = f"({_pubmed_q}) OR ({_portable_bool})"
        # Sans préprints : PubMed et Europe PMC les écartent dans la requête elle-même (le
        # type de publication « Preprint » de PubMed, la collection PPR d'Europe PMC), et le
        # total qu'elles annoncent est celui de la recherche réellement faite. Les autres
        # bases n'ont pas de filtre sûr ; leurs préprints sont retirés à l'assemblage.
        # Les années du scénario partent AUSSI dans la requête PubMed ([dp]). Elles n'y
        # étaient pas : PubMed rendait ses plus pertinentes TOUTES années confondues, et
        # l'assemblage jetait ensuite celles d'hors période, si bien qu'une recherche bornée
        # perdait des notices de sa période au-delà des 10 000 premières, et annonçait le
        # total de toutes les années. Le découpage par années reste dans ces bornes.
        # L'assemblage garde son propre filtre d'années, inchangé.
        _pubmed_base = _pubmed_q
        _pubmed_q = _pubmed_term(*_pubmed_years)
        if _no_preprints:
            _epmc_q = f"({_epmc_q}) NOT SRC:PPR"
            _preprints_at_source[:] = ["pubmed", "europepmc"]
        # Le MÊME helper que l'assemblage final (ligne ~1290) et que la recherche :
        # ce bloc appelait `_search_local_doc_ids(..., limit=100_000)` là où
        # `_boolean_corpus_ids` coupe à 500 000. Deux plafonds pour une seule question
        # (« quels documents locaux correspondent ») donnaient deux corpus différents
        # au même scénario selon l'étape, et le plus bas des deux était sous la taille
        # de la table.
        _local_ids = (_multi_query_corpus_ids(_sub_queries, _combinator, filters)
                      if _sub_queries else _boolean_corpus_ids(_boolean, filters))
        _local_capped[0] = bool(getattr(_local_ids, "capped", False))

        if _local_ids:
            with engine.begin() as _lc2:
                # Les correspondances locales sont liées tout de suite, et RIEN n'est
                # supprimé ici. Ce bloc remettait le corpus à la seule correspondance
                # locale AVANT la fédération : les liens que seules les sources
                # booléennes-natives apportent (MeSH, synonymes d'Europe PMC)
                # disparaissaient avant même de savoir si ces sources répondraient. Le
                # 12 octobre 2026, Europe PMC répondait 503 et le corpus de HPAI_last est
                # retombé à 8 494 articles au lieu de 15 000 à 18 000. Les liens obsolètes
                # sont retirés à l'assemblage final, qui sait si la fédération a abouti
                # (_set_scenario_corpus).
                # Insertion en masse (un seul aller-retour), tableau côté serveur : le
                # corpus local doit être lié quasi instantanément, sans une requête ni
                # un paramètre par document.
                _lc2.execute(text("""
                    INSERT INTO article_scenarios (document_id, scenario_id, similarity_score)
                    SELECT unnest(CAST(:ids AS bigint[])), :sid, NULL
                    ON CONFLICT (document_id, scenario_id) DO NOTHING
                """), {"ids": list(_local_ids), "sid": scenario_id})
                local_linked = len(_local_ids)
            _inc("db_cache", local_linked)
            # La base locale est une source interrogée comme les autres : ses
            # correspondances sont des enregistrements identifiés (PRISMA).
            _ident_records["db_cache"] = local_linked
            _ident_docs.update(_local_ids)
            logger.info(f"Populate {scenario_id}: corpus booléen = {local_linked} docs (base locale)")

        if local_linked > 0:
            with engine.begin() as _uc:
                _uc.execute(text("""
                    UPDATE user_scenarios SET article_count = :cnt WHERE id = :sid
                """), {"cnt": local_linked, "sid": scenario_id})

    except Exception as _e_local:
        logger.warning(f"Local DB link failed for {scenario_id}: {_e_local}")

    # ── Étape 1 : Interrogation parallèle des 11 sources externes ────────────
    # (11 fetchers ; Europe PMC sert 2 facettes - europepmc + préprints. Cf. source_funcs
    #  plus bas : sans préprints, la facette préprints et arXiv ne partent pas.)

    def _fetch_pubmed():
        count = 0
        try:
            # Throttle partagé eutils (verrou global + clé API + retry 429) : sinon
            # PubMed se faisait évincer par PROSPERO/Cochrane (mêmes serveurs eutils,
            # 3 req/s sans clé) → esearch 429 → total_found=0 → 0 article ingéré.
            # Identifiants D'ABORD : esearch renvoie la liste des PMID (du plus récent au
            # plus ancien, jusqu'au plafond par source). Les PMID déjà en base sont liés
            # sans rien télécharger ; seuls les inconnus passent par efetch. Avant, chaque
            # relance retéléchargeait les notices complètes de tout le résultat pour
            # n'insérer aucune ligne (ON CONFLICT) : la quasi-totalité du trafic PubMed.
            def _esearch(term: str, retstart: int, retmax: int) -> dict:
                """Une page d'identifiants (ou le seul compte, retmax=0), ou une exception."""
                _r = _with_retries(lambda: _ncbi_get(
                    f"{ENTREZ_BASE}/esearch.fcgi",
                    # `sort=relevance` (le « Best Match » de PubMed) : au plafond par source,
                    # on garde les plus PERTINENTS, comme OpenAlex et Europe PMC. Trié par date
                    # jusqu'ici, PubMed plafonné gardait les 2 000 plus récents et perdait la
                    # littérature H5N1 de 2004 à 2012 ; même plafond, deux politiques. Décision
                    # de méthode prise par le propriétaire du projet, pas un correctif.
                    {"db": "pubmed", "term": term, "retstart": retstart, "retmax": retmax,
                     "sort": "relevance", "retmode": "json", "email": EMAIL},
                    timeout=30,
                ), "PubMed")
                _r.raise_for_status()
                return _r.json()["esearchresult"]

            # eutils ne rend pas plus de 10 000 identifiants par recherche : la requête
            # entière d'abord, qui donne son vrai total et, sous 10 000, toute sa réponse.
            _ceiling = int(NCBI_ESEARCH_CEILING)
            _first_n = max(0, min(_cap_strategy, _ceiling))
            search_result = _esearch(_pubmed_q, 0, _first_n)
            total_found = int(search_result.get("count", 0))
            _note_total("pubmed", total_found)
            if _pipeline_callback:
                _pipeline_callback("pubmed_found", total_found)
            _pmids = [str(p).strip() for p in (search_result.get("idlist") or []) if str(p).strip()]
            _pmids = _pmids[:_first_n]
            # Au-delà, quand le plafond par source en demande davantage : la MÊME requête par
            # tranches d'années de publication, coupées en deux jusqu'à tenir sous 10 000
            # chacune (décision du propriétaire du produit). Une requête qui en annonçait
            # 23 000 n'en rapportait que 10 000. Le total de la carte reste celui de la
            # requête entière, lu ci-dessus, et non la somme des tranches, qui compterait
            # deux fois les notices à cheval sur deux années.
            if total_found > _ceiling and _cap_strategy > _ceiling:
                _split = pubmed_esearch_by_years(
                    _esearch, _pubmed_term, total=total_found, cap=_cap_strategy,
                    year_min=_pubmed_years[0], year_max=_pubmed_years[1],
                    ceiling=_ceiling, stop=_budget_exhausted)
                if _split["error"]:
                    _mark_source_error(f"PubMed, découpage par années : {_split['error']}")
                if _split["slices"]:
                    with _counter_lock:
                        _year_slices["pubmed"] = [s["years"] for s in _split["slices"]]
                        _over = {s["years"]: int(s["count"]) for s in _split["slices"] if s["capped"]}
                        if _over:
                            _year_slices_capped["pubmed"] = _over
                # Les tranches d'abord, puis ce que la requête entière a déjà rendu : un
                # découpage qui n'a pas pu aller au bout garde au moins ces 10 000-là.
                _merged: list[str] = []
                _seen_pm: set[str] = set()
                for _pm in list(_split["ids"]) + _pmids:
                    if _pm not in _seen_pm:
                        _seen_pm.add(_pm)
                        _merged.append(_pm)
                _pmids = _merged[:max(0, int(_cap_strategy))]
                logger.info(f"PubMed populate {scenario_id}: {total_found} résultats annoncés, "
                            f"requête découpée en {len(_split['slices'])} tranche(s) d'années "
                            f"({', '.join(s['years'] for s in _split['slices'])}) ; "
                            f"{len(_split['ids'])} PMID distincts"
                            + ("" if _split["complete"] else " (découpage incomplet)"))
            _pmid_set = set(_pmids)
            _known: dict[str, int] = {}
            for _ki in range(0, len(_pmids), 1000):
                _chunk = _pmids[_ki:_ki + 1000]
                try:
                    with engine.connect() as _kc:
                        # Deux recherches indexées réunies par UNION, et non un OR sur
                        # deux colonnes : Postgres ne peut pas combiner deux index pour
                        # un OR portant sur des colonnes différentes, il balayait donc
                        # toute la table literature_document (des centaines de milliers
                        # de lignes) une fois par lot de 1 000 PMID, par recherche.
                        # `pmid::text` écartait de surcroît l'index : on compare la
                        # colonne telle quelle à un tableau de texte.
                        _krows = _kc.execute(text("""
                            SELECT id, pmid::text AS pmid, external_id
                            FROM literature_document WHERE pmid = ANY(:p)
                            UNION
                            SELECT id, pmid::text AS pmid, external_id
                            FROM literature_document WHERE external_id = ANY(:e)
                        """), {"p": _chunk,
                               "e": [f"pmid:{p}" for p in _chunk]}).mappings().all()
                except Exception:                            # base sans colonne external_id
                    with engine.connect() as _kc:
                        _krows = _kc.execute(text(
                            "SELECT id, pmid::text AS pmid, NULL AS external_id FROM literature_document "
                            "WHERE pmid::text = ANY(:p)"
                        ), {"p": _chunk}).mappings().all()
                for _kr in _krows:
                    _pm = str(_kr["pmid"] or "").strip()
                    if not _pm and _kr["external_id"]:
                        _pm = str(_kr["external_id"]).strip()[5:]
                    if _pm in _pmid_set:
                        _known.setdefault(_pm, int(_kr["id"]))
            for _pm in _pmids:
                _did = _known.get(_pm)
                if _did is not None:
                    _link_to_scenario(_did, boolean_native=True, source="pubmed")   # déjà en base
            _unknown = [p for p in _pmids if p not in _known]
            logger.info(f"PubMed populate {scenario_id}: {total_found} résultats, {len(_pmids)} PMID retenus, "
                        f"{len(_known)} déjà en base, {len(_unknown)} à télécharger")
            n_batches = math.ceil(len(_unknown) / BATCH_SIZE) if _unknown else 0
            for batch_idx in range(n_batches):
                # Budget fédération dépassé → on ARRÊTE la pagination PubMed (comme
                # toutes les autres sources). Sans ce garde, PubMed continuait à
                # ingérer APRÈS la reconstruction finale du corpus (liens
                # boolean_native écrits trop tard) → membres non scorés + divergence
                # de article_count.
                if _budget_exhausted():
                    break
                _batch_ids = _unknown[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
                if not _batch_ids:
                    break
                try:
                    _ef_data = {"db": "pubmed", "id": ",".join(_batch_ids),
                                "rettype": "xml", "retmode": "xml", "email": EMAIL}
                    _ef_key = os.getenv("NCBI_API_KEY")
                    if _ef_key:
                        _ef_data["api_key"] = _ef_key

                    def _efetch_once():
                        # Espacer le DÉMARRAGE de la requête vis-à-vis des autres appels
                        # eutils (verrou partagé), sans tenir le verrou pendant le POST
                        # (lent) afin de ne pas sérialiser PROSPERO/Cochrane.
                        with _NCBI_LOCK:
                            _ef_wait = (0.11 if _ef_key else _NCBI_MIN_INTERVAL) - (_time.time() - _NCBI_LAST[0])
                            if _ef_wait > 0:
                                _time.sleep(_ef_wait)
                            _NCBI_LAST[0] = _time.time()
                        return _requests.post(f"{ENTREZ_BASE}/efetch.fcgi", data=_ef_data, timeout=90)
                    # Un lot qui échoue se RÉESSAIE avant d'être sauté.
                    r2 = _with_retries(_efetch_once, "PubMed efetch")
                    r2.raise_for_status()
                except Exception as _e_fetch:
                    # Un lot perdu = jusqu'à BATCH_SIZE notices absentes du corpus et des
                    # chiffres PRISMA. On continue (corpus partiel plutôt que rien) mais on
                    # MARQUE la source en erreur : sans cela, le fetcher passait pour
                    # complet et sa liste amputée était mémorisée dans le cache des
                    # sources, donc rejouée telle quelle pendant douze heures.
                    logger.warning(f"PubMed efetch batch {batch_idx}: {_e_fetch}")
                    _mark_source_error(_e_fetch)
                    _time.sleep(1)
                    continue
                root = ET.fromstring(r2.content)
                for article_elem in root.findall(".//PubmedArticle"):
                    pmid = article_elem.findtext(".//PMID") or ""
                    title_elem = article_elem.find(".//ArticleTitle")
                    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
                    abstract_parts = []
                    for node in article_elem.findall(".//Abstract/AbstractText"):
                        txt = "".join(node.itertext()).strip()
                        if txt:
                            abstract_parts.append(txt)
                    abstract = " ".join(abstract_parts).strip()
                    year = _pubmed_year(article_elem)
                    authors = _pubmed_authors(article_elem)       # tous, pas les six premiers
                    journal = (article_elem.findtext(".//Journal/Title")
                               or article_elem.findtext(".//ISOAbbreviation") or None)
                    doi = None
                    for id_elem in article_elem.findall(".//ArticleIdList/ArticleId"):
                        if id_elem.get("IdType") == "doi":
                            doi = id_elem.text
                            break
                    if not pmid or not title:
                        continue
                    content_text = f"{title}\n\n{abstract}".strip()
                    if len(content_text) < 30:
                        continue
                    try:
                        doc_id, _new = _ingest_doc_direct(
                            source="pubmed", title=title, abstract=abstract or None,
                            year=year, url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            # external_id « pmid:<id> » - MÊME format que la recherche
                            # live (_live_fetch_pubmed) et que la convention des autres
                            # sources (« s2: », « nct: », …). Auparavant brut (« <id> »),
                            # d'où deux lignes pour le même article sans DOI selon le
                            # chemin d'ingestion : le pré-SELECT par external_id ne les
                            # rapprochait pas. Uniformisé, le doublon n'est plus créé.
                            external_id=f"pmid:{pmid}", doi=doi, authors=authors, journal=journal,
                        )
                        _link_to_scenario(doc_id, boolean_native=True, source="pubmed")   # source-union
                        if _new:
                            count += 1
                            _inc("pubmed")
                    except Exception as e:
                        logger.warning(f"PubMed PMID {pmid}: {e}")
                        _inc("pubmed", 0, 1)
        except Exception as _e:
            logger.warning(f"PubMed populate {scenario_id}: {_e}")
            _mark_source_error(_e)
        return ("pubmed", count)

    def _fetch_openalex():
        count = 0
        # Deux passages : TOUTES les correspondances sur titre et résumé (filtre
        # title_and_abstract.search, qui accepte le booléen ; pagination par curseur, celle
        # par numéro de page s'arrêtant à 10 000), PUIS la recherche par défaut (titre,
        # résumé ET texte intégral), par pertinence, jusqu'au plafond classé : sans elle,
        # 30 articles pertinents de HPAI_last, trouvés par le texte intégral, sortaient du
        # corpus. Une notice rendue par les deux passages n'est comptée qu'une fois.
        # Un plan de passages (passage, requête, notices natives, plafond). Le booléen
        # découpé (trop long pour l'URL) donne un passage par morceau : leur union est le
        # booléen, et le passage en texte intégral se partage le plafond classé entre les
        # morceaux.
        _oa_split = list(_openalex_parts)
        if "openalex" in _title_abstract_sources and _oa_split:
            _top_each = max(1, math.ceil(_cap_ranked / len(_oa_split)))
            _passes = ([("scoped", p, True, _cap_strategy) for p in _oa_split]
                       + [("fulltext_top", p, True, _top_each) for p in _oa_split])
        elif "openalex" in _title_abstract_sources:
            _passes = [("scoped", _bool_query, True, _cap_strategy),
                       ("fulltext_top", _bool_query, True, _cap_ranked)]
        else:
            _passes = [("fulltext_all", _bool_query, _send_bool, _cap_strategy)]
        _seen: set = set()
        # Reprise : OpenAlex lit passage par passage (un par morceau du booléen découpé),
        # et sa liste de passages se réécrit en cours de route (un morceau refusé est coupé
        # en deux). Un NUMÉRO de passage ne serait donc pas stable d'un tour à l'autre : la
        # position retenue nomme le passage par son type ET sa requête, et le tour suivant
        # saute les passages déjà lus jusqu'à retrouver celui-là. Si la liste a changé au
        # point de ne plus le contenir, on repart du début, ce qui est sûr.
        _oa_skip = None
        try:
            _oa_skip = json.loads(_resume("") or "") or None
        except Exception:                                    # noqa: BLE001 - position illisible
            _oa_skip = None
        _oa_total = [_resume_fetched()]
        # Un morceau qu'OpenAlex refuse (4xx à sa première page) est coupé en deux et relancé,
        # jusqu'à OPENALEX_BISECT_DEPTH fois : un seul morceau refusé faisait échouer toute la
        # source (le 400 levait), et les morceaux suivants n'étaient jamais demandés.
        _oa_depth: dict[str, int] = {}
        _oa_bisected = [False]
        try:
            while _passes:
                _pass, _ox_q, _ox_native, _oa_limit = _passes.pop(0)
                _oa_page, _oa_fetched = 1, 0
                _oa_cursor = "*"
                if _oa_skip is not None:
                    if _oa_skip.get("kind") != _pass or _oa_skip.get("q") != _ox_q:
                        continue                 # passage déjà lu lors d'un tour précédent
                    _oa_cursor = _oa_skip.get("cursor") or "*"
                    _oa_page = int(_oa_skip.get("page") or 1)
                    _oa_fetched = int(_oa_skip.get("fetched_pass") or 0)
                    _oa_skip = None
                while _oa_fetched < _oa_limit:
                    if _budget_exhausted():
                        break  # budget fédération dépassé - on arrête de paginer
                    _oa_batch = min(200, _oa_limit - _oa_fetched)
                    if _pass == "fulltext_top":
                        # sort=relevance_score:desc → quand on plafonne à max_results, on
                        # garde les 2000 LES PLUS PERTINENTS (BM25 OpenAlex) et non les plus
                        # récents. `per-page`, avec un TRAIT D'UNION : c'est le nom du
                        # paramètre chez OpenAlex ; `per_page` était refusé.
                        _oa_params = openalex_params({"search": _ox_q, "per-page": _oa_batch, "page": _oa_page,
                                                      "sort": "relevance_score:desc"})
                    else:
                        # Pas de virgule dans un filtre OpenAlex : elle y sépare deux filtres.
                        _oa_params = ({"filter": "title_and_abstract.search:" + _ox_q.replace(",", " ")}
                                      if _pass == "scoped" else {"search": _ox_q})
                        # `mailto`, et la clé OpenAlex quand le serveur en a une (`openalex_params`).
                        _oa_params = openalex_params({**_oa_params, "per-page": _oa_batch,
                                                      "cursor": _oa_cursor})
                    oa_resp = _with_retries(lambda: _requests.get("https://api.openalex.org/works",
                                                                  params=_oa_params, timeout=20), "OpenAlex")
                    if (_oa_fetched == 0 and 400 <= oa_resp.status_code < 500 and oa_resp.status_code != 429
                            and _oa_depth.get(_ox_q, 0) < OPENALEX_BISECT_DEPTH):
                        _halves = _bisect_boolean(_ox_q)
                        if len(_halves) == 2:
                            for _h in _halves:
                                _oa_depth[_h] = _oa_depth.get(_ox_q, 0) + 1
                            _share = max(1, math.ceil(_oa_limit / 2)) if _pass == "fulltext_top" else _oa_limit
                            _passes[:0] = [(_pass, _h, _ox_native, _share) for _h in _halves]
                            with _counter_lock:
                                _sq = _split_queries.setdefault("openalex", [])
                                if _ox_q in _sq:
                                    _sq[_sq.index(_ox_q):_sq.index(_ox_q) + 1] = _halves
                                else:
                                    _sq.extend(h for h in _halves if h not in _sq)
                                _no_cache.add("_fetch_openalex")
                            _oa_bisected[0] = True
                            logger.warning(f"OpenAlex populate {scenario_id}: requête refusée ({oa_resp.status_code}, "
                                           f"{_count_operators(_ox_q)} opérateurs) ; coupée en deux et relancée")
                            break
                    if _pass == "scoped" and _oa_fetched == 0 and 400 <= oa_resp.status_code < 500:
                        # Le filtre titre+résumé refusé : toute la recherche par défaut, et
                        # la carte ne dit plus « titre et résumé ». Pas de mise en cache :
                        # une relance doit retenter le filtre.
                        logger.warning(f"OpenAlex populate {scenario_id}: filtre title_and_abstract.search "
                                       f"refusé ({oa_resp.status_code}) ; recherche par défaut")
                        with _counter_lock:
                            _title_abstract_sources.discard("openalex")
                            _no_cache.add("_fetch_openalex")
                        _passes = [("fulltext_all", q, n, _cap_strategy)
                                   for (k, q, n, _l) in [(_pass, _ox_q, _ox_native, _oa_limit)] + _passes
                                   if k == "scoped"]
                        break
                    oa_resp.raise_for_status()
                    _oa_json = oa_resp.json()
                    # Le total de la recherche faite : celui du passage qui prend tout ; le
                    # passage en texte intégral n'en est qu'un ajout. Celui d'un booléen
                    # découpé n'est pas la somme des morceaux (ils se recoupent) : il n'est
                    # pas annoncé, comme pour DOAJ.
                    if _pass != "fulltext_top" and not _oa_split and not _oa_bisected[0]:
                        _note_total("openalex", (_oa_json.get("meta") or {}).get("count"))
                    _oa_results = _oa_json.get("results", [])
                    if not _oa_results:
                        break
                    for work in _oa_results:
                        ext_id = work.get("id", "").split("/")[-1]
                        title = work.get("title") or ""
                        if not ext_id or not title or ext_id in _seen:
                            continue
                        _seen.add(ext_id)
                        abstract = None
                        inv = work.get("abstract_inverted_index")
                        if inv:
                            try:
                                words = {}
                                for w, positions in inv.items():
                                    for pos in positions:
                                        words[pos] = w
                                abstract = " ".join([words[i] for i in sorted(words.keys())])
                            except Exception:
                                pass
                        year = work.get("publication_year")
                        doi = _normalize_doi(work.get("doi"))
                        url = doi or f"https://openalex.org/{ext_id}"
                        content_text = f"{title}\n\n{abstract or ''}".strip()
                        if len(content_text) < 30:
                            continue
                        try:
                            doc_id, _new = _ingest_doc_direct(
                                source="openalex", title=title, abstract=abstract or None,
                                year=year, url=url, external_id=ext_id, doi=doi,
                            )
                            _link_to_scenario(doc_id, boolean_native=_ox_native, source="openalex")   # source-union si booléen
                            if _new:
                                count += 1
                                _inc("openalex")
                        except Exception:
                            _inc("openalex", 0, 1)
                    _oa_fetched += len(_oa_results)
                    _oa_total[0] += len(_oa_results)
                    if _pass == "fulltext_top":
                        if len(_oa_results) < _oa_batch or _oa_fetched >= _oa_limit:
                            break
                        _oa_page += 1
                    else:
                        _oa_cursor = (_oa_json.get("meta") or {}).get("next_cursor")
                        if not _oa_cursor or _oa_fetched >= _oa_limit:
                            break
                    _note_pos(json.dumps({"kind": _pass, "q": _ox_q, "cursor": _oa_cursor,
                                          "page": _oa_page, "fetched_pass": _oa_fetched}),
                              _oa_total[0])
                    _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"OpenAlex populate {scenario_id}: {_e}")
            _mark_source_error(_e)
        return ("openalex", count)

    def _fetch_crossref():
        count = 0
        try:
            # Reprise : le décalage est la position, et il vaut pour la même requête.
            _cr_offset = int(_resume(0) or 0)
            _cr_fetched = _resume_fetched()
            _cr_limit = _cap_ranked
            _cr_rows = min(1000, _cr_limit)   # max Crossref : 10× moins d'allers-retours
            while _cr_fetched < _cr_limit:
                if _budget_exhausted():
                    break  # budget fédération dépassé - on arrête de paginer
                cr_resp = _with_retries(lambda: _requests.get(
                    "https://api.crossref.org/works",
                    # NB : PAS de sort=published desc ici (contrairement aux autres
                    # sources). Les dates Crossref sont peu fiables : un tri par date
                    # remonte des enregistrements à dates erronées (ex. « 2121 ») en
                    # tête. On garde donc le tri par pertinence (défaut), qui place les
                    # articles les plus pertinents - pas les plus faussement récents.
                    params={"query": _plain_q, "rows": _cr_rows, "offset": _cr_offset,
                            "mailto": "literev@gesica.ch"},
                    timeout=20,
                ), "Crossref")
                cr_resp.raise_for_status()
                _cr_msg = cr_resp.json().get("message", {})
                _note_total("crossref", _cr_msg.get("total-results"))
                _cr_items = _cr_msg.get("items", [])
                if not _cr_items:
                    break
                for item in _cr_items:
                    doi = _normalize_doi(item.get("DOI"))
                    titles = item.get("title", [])
                    title = titles[0] if titles else ""
                    if not doi or not title:
                        continue
                    abstract = item.get("abstract")
                    if abstract and abstract.startswith("<"):
                        try:
                            import xml.etree.ElementTree as _ET
                            abstract = "".join(_ET.fromstring(abstract).itertext()).strip()
                        except Exception:
                            pass
                    year = _crossref_year(item)
                    content_text = f"{title}\n\n{abstract or ''}".strip()
                    if len(content_text) < 30:
                        continue
                    try:
                        doc_id, _new = _ingest_doc_direct(
                            source="crossref", title=title, abstract=abstract or None,
                            year=year, url=f"https://doi.org/{doi}", external_id=doi, doi=doi,
                        )
                        _link_to_scenario(doc_id, source="crossref")
                        if _new:
                            count += 1
                            _inc("crossref")
                    except Exception:
                        _inc("crossref", 0, 1)
                _cr_fetched += len(_cr_items)
                if len(_cr_items) < _cr_rows or _cr_fetched >= _cr_limit:
                    break
                _cr_offset += _cr_rows
                _note_pos(_cr_offset, _cr_fetched)
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"Crossref populate {scenario_id}: {_e}")
            _mark_source_error(_e)
        return ("crossref", count)

    def _fetch_europepmc():
        count = 0
        try:
            # Reprise : le curseur d'Europe PMC est opaque et stable pour une requête
            # donnée, donc il repart exactement où le tour précédent s'est arrêté.
            _ep_cursor_mark = _resume("*")
            _ep_fetched = _resume_fetched()
            _ep_limit = _cap_strategy
            _ep_page_size = 1000   # max Europe PMC : moins d'allers-retours → fédération plus rapide
            while _ep_fetched < _ep_limit:
                if _budget_exhausted():
                    break  # budget fédération dépassé - on arrête de paginer
                ep_resp = _with_retries(lambda: epmc_send(
                    _requests,
                    # Pas de tri par date : on laisse le tri par PERTINENCE (défaut Europe PMC)
                    # → au plafond de 2000, on garde les plus pertinents et non les plus récents.
                    # Une stratégie longue part en POST : dans une URL, leur serveur répond 503
                    # (cf. EPMC_POST_ABOVE_ENCODED), et aucun nouvel essai n'y changeait rien.
                    {"query": _epmc_q, "format": "json", "pageSize": _ep_page_size,
                     "resultType": "core",
                     "cursorMark": _ep_cursor_mark},
                    timeout=20,
                ), "Europe PMC")
                ep_resp.raise_for_status()
                _ep_data = ep_resp.json()
                _note_total("europepmc", _ep_data.get("hitCount"))
                _ep_results = _ep_data.get("resultList", {}).get("result", [])
                if not _ep_results:
                    break
                for res in _ep_results:
                    pmid = res.get("pmid")
                    pmcid = res.get("pmcid")
                    doi = _normalize_doi(res.get("doi"))
                    ext_id = pmcid or pmid or doi
                    title = res.get("title") or ""
                    if not ext_id or not title:
                        continue
                    abstract = res.get("abstractText")
                    if abstract and abstract.startswith("<"):
                        try:
                            import xml.etree.ElementTree as _ET2
                            abstract = "".join(_ET2.fromstring(f"<root>{abstract}</root>").itertext()).strip()
                        except Exception:
                            pass
                    year = None
                    yt = res.get("pubYear")
                    if yt and str(yt).isdigit():
                        year = int(yt)
                    url = (f"https://europepmc.org/article/{pmcid or pmid}" if (pmcid or pmid)
                           else (f"https://doi.org/{doi}" if doi else None))
                    content_text = f"{title}\n\n{abstract or ''}".strip()
                    if len(content_text) < 30:
                        continue
                    try:
                        doc_id, _new = _ingest_doc_direct(
                            source="europepmc", title=title, abstract=abstract or None,
                            year=year, url=url, external_id=ext_id, doi=doi,
                        )
                        _link_to_scenario(doc_id, boolean_native=True, source="europepmc")   # source-union
                        if _new:
                            count += 1
                            _inc("europepmc")
                    except Exception:
                        _inc("europepmc", 0, 1)
                _ep_fetched += len(_ep_results)
                _ep_next_cursor = _ep_data.get("nextCursorMark")
                if not _ep_next_cursor or _ep_next_cursor == _ep_cursor_mark or len(_ep_results) < _ep_page_size or _ep_fetched >= _ep_limit:
                    break
                _ep_cursor_mark = _ep_next_cursor
                _note_pos(_ep_cursor_mark, _ep_fetched)
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"EuropePMC populate {scenario_id}: {_e}")
            _mark_source_error(_e)
        return ("europepmc", count)

    def _fetch_preprints():
        # Préprints (bioRxiv, medRxiv, Research Square, …) via Europe PMC (SRC:PPR) :
        # recherche par mots-clés RÉELLE. L'API bioRxiv ne fait que dates et DOI : le
        # fetcher natif qui scannait ses 45 derniers jours et filtrait côté client ne
        # trouvait rien, et a quitté la fédération ; Europe PMC indexe les deux serveurs.
        count = 0
        try:
            _pp_cursor = _resume("*")
            _pp_fetched = 0
            _pp_query = f"({_epmc_q}) AND (SRC:PPR)"
            while _pp_fetched < _cap_strategy:
                if _budget_exhausted():
                    break  # budget fédération dépassé
                _pp_resp = _with_retries(lambda: epmc_send(
                    _requests,
                    # Par PERTINENCE (le tri par défaut d'Europe PMC), comme toutes les bases
                    # qui le permettent : au plafond, on garde les préprints les plus
                    # pertinents. Triés par date, on gardait les plus récents.
                    {"query": _pp_query, "format": "json", "pageSize": 100,
                     "resultType": "core", "cursorMark": _pp_cursor},
                    timeout=20,
                ), "Europe PMC preprints")
                _pp_resp.raise_for_status()
                _pp_data = _pp_resp.json()
                # Le total annoncé : au plafond, la carte dit « n / total » et « plafonnée ».
                _note_total("preprints", _pp_data.get("hitCount"))
                _pp_results = _pp_data.get("resultList", {}).get("result", [])
                if not _pp_results:
                    break
                for res in _pp_results:
                    doi = _normalize_doi(res.get("doi"))
                    ext_id = res.get("id") or doi
                    title = res.get("title") or ""
                    if not ext_id or not title:
                        continue
                    abstract = res.get("abstractText")
                    if abstract and abstract.startswith("<"):
                        try:
                            import xml.etree.ElementTree as _ET3
                            abstract = "".join(_ET3.fromstring(f"<root>{abstract}</root>").itertext()).strip()
                        except Exception:
                            pass
                    year = None
                    yt = res.get("pubYear")
                    if yt and str(yt).isdigit():
                        year = int(yt)
                    url = (f"https://europepmc.org/article/{res.get('source','PPR')}/{res.get('id')}"
                           if res.get("id") else (f"https://doi.org/{doi}" if doi else None))
                    content_text = f"{title}\n\n{abstract or ''}".strip()
                    if len(content_text) < 30:
                        continue
                    try:
                        doc_id, _new = _ingest_doc_direct(
                            source="preprint", title=title, abstract=abstract or None,
                            year=year, url=url, external_id=ext_id, doi=doi,
                            source_type="preprint",
                        )
                        _link_to_scenario(doc_id, boolean_native=True, source="preprint")   # source-union
                        if _new:
                            count += 1
                            _inc("preprint")
                    except Exception:
                        _inc("preprint", 0, 1)
                _pp_fetched += len(_pp_results)
                _pp_next = _pp_data.get("nextCursorMark")
                if not _pp_next or _pp_next == _pp_cursor or len(_pp_results) < 100:
                    break
                _pp_cursor = _pp_next
                _note_pos(_pp_cursor, _pp_fetched)
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"Préprints (Europe PMC) populate {scenario_id}: {_e}")
            _mark_source_error(_e)
        return ("preprints", count)

    def _ingest_parsed(source, docs, boolean_native=False):
        """Ingère une liste de docs parsés (helper commun aux nouvelles sources REST).
        Renvoie le nombre ingéré. Chaque doc : {title, abstract, year, url,
        external_id, doi, source_type}. Les docs <30 caractères sont ignorés.
        boolean_native=True : la source a appliqué un VRAI booléen → source-union."""
        c = 0
        for _d in docs:
            if len(f"{_d.get('title','')}\n\n{_d.get('abstract') or ''}".strip()) < 30:
                continue
            try:
                _doc_id, _new = _ingest_doc_direct(
                    source=source, title=_d["title"], abstract=_d.get("abstract"),
                    year=_d.get("year"), url=_d.get("url"), external_id=_d["external_id"],
                    doi=_d.get("doi"), source_type=_d.get("source_type", "article"),
                )
                _link_to_scenario(_doc_id, boolean_native=boolean_native, source=source)
                if _new:
                    c += 1
                    _inc(source)
            except Exception:
                _inc(source, 0, 1)
        return c

    def _fetch_semantic_scholar():
        count = 0
        # Les en-têtes (la clé) se demandent à chaque tentative : avec une clé, chaque requête
        # attend son tour, une par seconde pour tout le serveur (`s2_headers_paced`).
        _fields = "title,abstract,year,externalIds,url"
        # Vrai booléen → endpoint BULK (opérateurs AND/OR + jusqu'à 10M via token) +
        # source-union. Sinon → endpoint classique (relevance, offset≤1000) + mots-clés.
        # L'endpoint bulk rend les notices dans l'ordre des identifiants, PAS par pertinence
        # (calculé avec les autres requêtes, plus haut, pour que la carte le dise).
        _s2_bool = _s2_bool_q[0]
        # Un seul booléen, ou les morceaux dont l'union est le booléen entier. Liste vide :
        # pas de booléen pour cette source, et c'est l'endpoint classique qui répond, dans
        # le `else` plus bas. `for ... else` ne dirait PAS cela : son else s'exécute quand
        # la boucle finit sans break, donc après un passage réussi.
        _s2_queries = list(_s2_parts) or ([_s2_bool] if _s2_bool else [])
        try:
            _skip, _total = _part_resume(), [_resume_fetched()]
            for _s2_bool in _s2_queries:
                _tok, _fetched, _429 = None, 0, 0
                if _skip:
                    if _skip.get("q") != _s2_bool:
                        continue
                    _tok, _fetched = _skip.get("pos"), int(_skip.get("fetched_part") or 0)
                    _skip = {}
                while _fetched < _cap_strategy:
                    if _budget_exhausted():
                        break
                    _p = {"query": _s2_bool, "fields": _fields}
                    if _tok:
                        _p["token"] = _tok
                    _r = _with_retries(lambda: _requests.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
                        params=_p, headers=s2_headers_paced(), timeout=25), "Semantic Scholar", retry_429=False)
                    if _r.status_code == 429:
                        _wait_out_rate_limit(_r, _429, "Semantic Scholar", keyed=bool(s2_key()))
                        _429 += 1
                        continue
                    _429 = 0                      # cette page est passée
                    _r.raise_for_status()
                    _payload = _r.json()
                    _note_total("semantic_scholar", _payload.get("total"))
                    _n = len(_payload.get("data") or [])
                    count += _ingest_parsed("semantic_scholar", _parse_semantic_scholar(_payload),
                                            boolean_native=_s2_native[0])
                    _fetched += _n
                    _tok = _payload.get("token")
                    if not _tok or _n == 0:
                        break
                    _total[0] += _n
                    _note_pos(json.dumps({"q": _s2_bool, "pos": _tok, "fetched_part": _fetched}),
                              _total[0])
                    _time.sleep(0.3)
            if not _s2_queries:
                _off, _cap, _429 = 0, min(max_results, 1000), 0   # search classique : offset+limit ≤ 1000
                while _off < _cap:
                    if _budget_exhausted():
                        break
                    _bulk = min(100, _cap - _off)
                    _r = _with_retries(lambda: _requests.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        params={"query": _plain_q, "offset": _off, "limit": _bulk, "fields": _fields},
                        headers=s2_headers_paced(), timeout=20,
                    ), "Semantic Scholar", retry_429=False)
                    if _r.status_code == 429:
                        _wait_out_rate_limit(_r, _429, "Semantic Scholar", keyed=bool(s2_key()))
                        _429 += 1
                        continue
                    _429 = 0                      # cette page est passée
                    _r.raise_for_status()
                    _payload = _r.json()
                    _note_total("semantic_scholar", _payload.get("total"))
                    _got = len(_payload.get("data") or [])
                    count += _ingest_parsed("semantic_scholar", _parse_semantic_scholar(_payload))
                    if _got < _bulk:
                        break
                    _off += _bulk
                    _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"Semantic Scholar populate {scenario_id}: {_e}")
            _mark_source_error(_e)
        return ("semantic_scholar", count)

    def _fetch_doaj():
        count = 0
        try:
            import urllib.parse as _ulib
            # Une requête, ou le booléen découpé : une notice rendue
            # par deux morceaux n'est comptée qu'une fois, sinon elle gonflerait le compte de
            # DOAJ et les doublons du PRISMA d'une notice que la base n'a rendue qu'une fois.
            _seen: set = set()
            _skip, _total = _part_resume(), [_resume_fetched()]
            for _dq in (_doaj_parts or [_doaj_q]):
                _page, _fetched = 1, 0
                if _skip:
                    if _skip.get("q") != _dq:
                        continue              # morceau déjà lu lors d'un tour précédent
                    _page, _fetched = int(_skip.get("pos") or 1), int(_skip.get("fetched_part") or 0)
                    _skip = {}
                while _fetched < _cap_strategy:
                    if _budget_exhausted():
                        break
                    _r = _with_retries(lambda: _requests.get(
                        f"https://doaj.org/api/search/articles/{_ulib.quote(_dq, safe='')}",
                        params={"pageSize": 100, "page": _page}, timeout=20,
                    ), "DOAJ")
                    # DOAJ sert la première page d'un booléen long et répond 502 à toutes les
                    # suivantes, quelle que soit la taille de page (reproduit depuis le serveur
                    # de production sur HPAI_last ; une requête courte pagine normalement). La
                    # raison servie à la carte dit la page et ce qui a été gardé, au lieu d'une
                    # URL de 1 300 caractères coupée avant le numéro de page.
                    if _r.status_code >= 500 and _page > 1:
                        raise RuntimeError(
                            f"DOAJ : {_r.status_code} à la page {_page} ({_fetched} notices de la "
                            f"première page gardées, {FETCH_RETRY_ATTEMPTS} tentatives) ; l'API ne "
                            f"pagine pas plus loin sur une requête de cette longueur")
                    _r.raise_for_status()
                    _payload = _r.json()
                    _n = len(_payload.get("results") or [])
                    if _n == 0:
                        break
                    # Le total d'UNE requête ; celui d'un booléen découpé n'est pas la somme
                    # des morceaux (ils se recoupent), il n'est donc pas annoncé.
                    if not _doaj_parts:
                        _note_total("doaj", _payload.get("total"))
                    _docs = [d for d in _parse_doaj(_payload) if d.get("external_id") not in _seen]
                    _seen.update(d.get("external_id") for d in _docs)
                    count += _ingest_parsed("doaj", _docs, boolean_native=_doaj_native)
                    _fetched += _n
                    if _n < 100:
                        break
                    _page += 1
                    _total[0] += _n
                    _note_pos(json.dumps({"q": _dq, "pos": _page, "fetched_part": _fetched}),
                              _total[0])
                    _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"DOAJ populate {scenario_id}: {_e}")
            _mark_source_error(_e)
        return ("doaj", count)

    def _fetch_clinicaltrials():
        count = 0
        try:
            # Une requête, ou le booléen découpé (trop long pour l'URL) : une étude rendue
            # par deux morceaux n'est comptée qu'une fois.
            _ct_seen: set = set()
            _ct_native = _send_bool or bool(_ct_parts)
            for _ctq in (_ct_parts or [_bool_query]):
                _ct_token, _ct_fetched = _resume(None), _resume_fetched()
                while _ct_fetched < _cap_strategy:
                    if _budget_exhausted():
                        break
                    # `countTotal` : le total annoncé, pour que « clinicaltrials 2 000 » au
                    # plafond ne se lise pas comme ce que le registre contient.
                    _params = {"query.term": _ctq, "pageSize": 100, "format": "json",
                               "countTotal": "true"}
                    if _ct_token:
                        _params["pageToken"] = _ct_token
                    _r = _with_retries(lambda: _requests.get("https://clinicaltrials.gov/api/v2/studies",
                                                             params=_params, timeout=20), "ClinicalTrials.gov")
                    _r.raise_for_status()
                    _payload = _r.json()
                    if not _ct_parts:
                        _note_total("clinicaltrials", _payload.get("totalCount"))
                    _n = len(_payload.get("studies") or [])
                    if _n == 0:
                        break
                    _docs = [d for d in _parse_clinicaltrials(_payload) if d.get("external_id") not in _ct_seen]
                    _ct_seen.update(d.get("external_id") for d in _docs)
                    count += _ingest_parsed("clinicaltrials", _docs, boolean_native=_ct_native)
                    _ct_fetched += _n
                    _ct_token = _payload.get("nextPageToken")
                    if not _ct_token:
                        break
                    _note_pos(_ct_token, _ct_fetched)
                    _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"ClinicalTrials.gov populate {scenario_id}: {_e}")
            _mark_source_error(_e)
        return ("clinicaltrials", count)

    def _fetch_core():
        # CORE exige une clé API (gratuite). Sans clé → source ignorée proprement
        # (comme NCBI_API_KEY : optionnelle, le déploiement reste vert).
        _core_key = os.getenv("CORE_API_KEY")
        if not _core_key:
            logger.info(f"CORE populate {scenario_id}: CORE_API_KEY absent - source ignorée.")
            _mark_source_skipped("no_api_key")
            return ("core", 0)
        count = 0
        try:
            _core_offset, _core_fetched, _429 = int(_resume(0) or 0), _resume_fetched(), 0
            while _core_fetched < _cap_strategy:
                if _budget_exhausted():
                    break
                _bulk = min(100, _cap_strategy - _core_fetched)
                # Le booléen ENTIER, quelle que soit sa longueur : il voyage dans le corps
                # JSON d'un POST, qu'aucune limite d'URL ne concerne (cf. `_core_q`).
                _r = _with_retries(lambda: _requests.post(
                    "https://api.core.ac.uk/v3/search/works",
                    headers={"Authorization": f"Bearer {_core_key}"},
                    json={"q": _core_q, "limit": _bulk, "offset": _core_offset},
                    timeout=25,
                ), "CORE", retry_429=False)
                if _r.status_code == 429:
                    _wait_out_rate_limit(_r, _429, "CORE")
                    _429 += 1
                    continue
                _429 = 0                          # cette page est passée
                _r.raise_for_status()
                _payload = _r.json()
                _note_total("core", _payload.get("totalHits"))
                _n = len(_payload.get("results") or [])
                if _n == 0:
                    break
                count += _ingest_parsed("core", _parse_core(_payload), boolean_native=_core_native)
                _core_fetched += _n
                _core_offset += _bulk
                _note_pos(_core_offset, _core_fetched)
                if _n < _bulk:
                    break
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"CORE populate {scenario_id}: {_e}")
            _mark_source_error(_e)
        return ("core", count)

    def _fetch_arxiv():
        count = 0
        try:
            # Une requête, ou le booléen découpé : une notice rendue par deux morceaux n'est
            # comptée qu'une fois.
            _ax_seen: set = set()
            _skip, _total = _part_resume(), [_resume_fetched()]
            for _axq in (_arxiv_parts or [_arxiv_q]):
                _ax_start = 0
                if _skip:
                    if _skip.get("q") != _axq:
                        continue
                    _ax_start = int(_skip.get("pos") or 0)
                    _skip = {}
                while _ax_start < _cap_strategy:
                    if _budget_exhausted():
                        break
                    _bulk = min(100, _cap_strategy - _ax_start)
                    _r = _with_retries(lambda: _requests.get(
                        "http://export.arxiv.org/api/query",
                        params={"search_query": _axq, "start": _ax_start, "max_results": _bulk},
                        timeout=25,
                    ), "arXiv", min_wait=3.0)   # arXiv demande ≥3 s entre requêtes, retries compris
                    _r.raise_for_status()
                    if not _arxiv_parts:
                        _note_total("arxiv", _parse_arxiv_total(_r.text))
                    _page_docs = _parse_arxiv(_r.text)
                    if not _page_docs:
                        break
                    _docs = [d for d in _page_docs if d.get("external_id") not in _ax_seen]
                    _ax_seen.update(d.get("external_id") for d in _docs)
                    count += _ingest_parsed("arxiv", _docs, boolean_native=_arxiv_native)
                    if len(_page_docs) < _bulk:
                        break
                    _ax_start += _bulk
                    _total[0] += len(_page_docs)
                    _note_pos(json.dumps({"q": _axq, "pos": _ax_start}), _total[0])
                    _time.sleep(3)      # arXiv demande ≥3 s entre requêtes
                if _arxiv_parts:
                    _time.sleep(3)
        except Exception as _e:
            logger.warning(f"arXiv populate {scenario_id}: {_e}")
            _mark_source_error(_e)
        return ("arxiv", count)

    def _fetch_openaire():
        count = 0
        # Migration vers l'API Graph v2 : l'ancien /search/publications a été RETIRÉ le
        # 2026-05-31. `search=` accepte AND/OR/NOT + parenthèses + guillemets, mais QUATRE
        # opérateurs au plus (mesuré en production). Pagination par CURSEUR.
        # Sa propre réduction, bornée à OPENAIRE_MAX_OPERATORS opérateurs (calculée avec les
        # autres requêtes, plus haut) ; jamais « native » : ses notices sont ré-appariées
        # en local contre le booléen entier.
        _oa_q, _oa_native = _openaire_q, False
        # Le booléen ENTIER en morceaux de quatre opérateurs au plus
        # (`_openaire_parts`, quand il en faut SPLIT_MAX_PARTS au plus). Leur union est le
        # booléen : ces notices-là sont natives ; la réduction, elle, ne l'est jamais.
        if _openaire_parts:
            _oa_native = True
        try:
            _oa_seen: set = set()
            _skip, _total = _part_resume(), [_resume_fetched()]
            for _oaq in (_openaire_parts or [_oa_q]):
                _cursor, _fetched = "*", 0
                if _skip:
                    if _skip.get("q") != _oaq:
                        continue
                    _cursor = _skip.get("pos") or "*"
                    _fetched = int(_skip.get("fetched_part") or 0)
                    _skip = {}
                while _fetched < _cap_strategy:
                    if _budget_exhausted():
                        break
                    _r = _with_retries(lambda: _requests.get(
                        "https://api.openaire.eu/graph/v2/researchProducts",
                        params={"search": _oaq, "pageSize": 50, "cursor": _cursor},
                        headers={"Accept": "application/json"}, timeout=25,
                    ), "OpenAIRE")
                    _r.raise_for_status()
                    _payload = _r.json()
                    _page_docs = _parse_openaire_graph(_payload)
                    _docs = [d for d in _page_docs if d.get("external_id") not in _oa_seen]
                    _oa_seen.update(d.get("external_id") for d in _docs)
                    count += _ingest_parsed("openaire", _docs, boolean_native=_oa_native)
                    _fetched += len(_page_docs)
                    _next = (_payload.get("header") or {}).get("nextCursor")
                    if not _page_docs or not _next or _next == _cursor:
                        break
                    _cursor = _next
                    _total[0] += len(_page_docs)
                    _note_pos(json.dumps({"q": _oaq, "pos": _cursor, "fetched_part": _fetched}),
                              _total[0])
                    _time.sleep(0.5)
        except Exception as _e:
            logger.warning(f"OpenAIRE (Graph API v2) populate {scenario_id}: {_e}")
            _mark_source_error(_e)
        return ("openaire", count)

    # Lancer toutes les sources en parallèle. bioRxiv et medRxiv n'y sont plus en direct :
    # leur API ne cherche pas par mots-clés (fenêtre de dates seulement), le fetcher
    # scannait les 45 derniers jours et filtrait côté client, et rendait 0 notice sur
    # HPAI_last quand la facette préprints d'Europe PMC, qui indexe ces deux serveurs, en
    # rendait 457 (tous serveurs confondus). L'API bioRxiv reste employée pour le texte
    # intégral, par DOI.
    source_funcs = [
        _fetch_pubmed, _fetch_openalex, _fetch_crossref,
        _fetch_europepmc, _fetch_preprints,
        _fetch_semantic_scholar, _fetch_doaj, _fetch_clinicaltrials, _fetch_core,
        _fetch_arxiv, _fetch_openaire,
    ]
    # Sans préprints : les serveurs de préprints ne sont pas interrogés du tout. Ils ne
    # sont pas « lancés », donc ni comptés parmi les sources interrogées ni annoncés en
    # échec : la carte PRISMA les nomme sur la ligne de l'exclusion.
    _preprint_servers = (_fetch_preprints, _fetch_arxiv)
    _preprint_servers_skipped = ([_source_label(fn.__name__) for fn in _preprint_servers]
                                 if _no_preprints else [])
    if _no_preprints:
        source_funcs = [fn for fn in source_funcs if fn not in _preprint_servers]
    t_start = _time.time()
    _completed_ok: set[str] = set()
    # Fetchers qui ont RÉPONDU pendant ce run (cache rejoué inclus) : la couverture
    # annoncée à l'utilisateur en fin de recherche doit être celle-là, pas un « 13 »
    # constant. Une source coupée par le budget n'y figure pas.
    _queried: set[str] = set()
    # Ce que chaque fetcher a RENVOYÉ (nombre d'articles ingérés), et l'issue nommée
    # qu'on en tire : ok, empty, cached, skipped, error, cut_by_budget. Le PRISMA et la
    # ligne de couverture lisent celle-ci, plus un compte de sources « interrogées » qui
    # incluait les échecs.
    _returned: dict[str, int] = {}
    _fetcher_outcome: dict[str, str] = {}
    _qhash = _source_query_hash(query, filters, max_results, mode=_search_mode)
    if include_live:
        _set_phase("federation")
        # Garde-temps : passé ce délai, les boucles de pagination des sources lentes
        # s'arrêtent (cf. _fed_deadline) et on poursuit avec le corpus partiel.
        _fed_deadline[0] = t_start + _fed_budget
        # Cache des sources : les fetchers dont la réponse à CETTE requête est encore
        # fraîche (SOURCE_CACHE_TTL_S) sont rejoués depuis la base, sans appel réseau,
        # avec les mêmes compteurs ; les autres partent normalement.
        _cached = {} if force_live else _load_source_cache(_qhash)
        # D'où chaque source coupée doit repartir. `force_live` veut dire « relis tout
        # depuis le début », donc il oublie aussi les positions : sans cela, une relance
        # demandée pour repartir à neuf aurait sauté ce qui avait déjà été lu.
        if force_live:
            _clear_fetch_cursor(scenario_id, query_hash=_qhash)
        else:
            _resume_state.update(_load_fetch_cursors(scenario_id, _qhash))
            if _resume_state:
                logger.info(f"Populate {scenario_id}: reprise de "
                            + ", ".join(f"{k} (après {v.get('fetched')} notices, "
                                        f"tour {int(v.get('rounds') or 0) + 1})"
                                        for k, v in sorted(_resume_state.items())))
        for _fname, _entry in _cached.items():
            for _cd, _cs, _cn in _entry["links"]:
                _link_to_scenario(_cd, boolean_native=_cn, source=_cs)
            # Le total annoncé par la source rejoue avec ses liens : la carte peut encore
            # dire qu'elle a été plafonnée.
            for _src, _tot in (_entry.get("totals") or {}).items():
                _note_total(_src, _tot)
            # Et les tranches d'années de PubMed, quand il a fallu découper.
            _meta = _entry.get("meta") or {}
            with _counter_lock:
                for _src, _sl in (_meta.get("year_slices") or {}).items():
                    if isinstance(_sl, list) and _sl:
                        _year_slices[str(_src)] = [str(x) for x in _sl]
                for _src, _ov in (_meta.get("year_slices_capped") or {}).items():
                    if isinstance(_ov, dict) and _ov:
                        _year_slices_capped[str(_src)] = {str(k): int(v) for k, v in _ov.items()
                                                          if isinstance(v, (int, float))}
            if _pipeline_callback is None:
                with _counter_lock:
                    _job = _user_scenario_populate_jobs.get(scenario_id)
                    if _job is not None:
                        _job.setdefault("cached_sources", []).append(_fname.replace("_fetch_", ""))
            _queried.add(_fname)                     # servie, sans appel réseau
            logger.info(f"Populate {scenario_id} [{_fname}]: {len(_entry['links'])} liens rejoués depuis le "
                        f"cache des sources ({_entry['age_s'] / 60:.0f} min)")
        _to_run = [fn for fn in source_funcs if fn.__name__ not in _cached]
        # IMPORTANT : on N'UTILISE PAS `with ThreadPoolExecutor(...)` : sa sortie
        # appelle shutdown(wait=True), qui attend TOUTES les sources (jusqu'à ~5 min
        # quand OpenAlex/Crossref paginent vers 2000), annulant de fait le budget.
        # On gère l'executor manuellement et on l'arrête SANS attendre.
        executor = ThreadPoolExecutor(max_workers=12)
        try:
            futures = {executor.submit(_run_fetcher, fn): fn.__name__ for fn in _to_run}
            try:
                # Budget global : ne pas attendre indéfiniment une source lente.
                for future in as_completed(futures, timeout=_fed_budget):
                    _fname = futures[future]
                    try:
                        src_name, src_count = future.result()
                        _returned[_fname] = int(src_count or 0)
                        logger.info(f"Populate {scenario_id} [{src_name}]: {src_count} articles ingérés")
                        # Pagination allée au bout AVANT le budget, sans erreur et sans
                        # avoir été écartée faute de clé : la réponse de ce fetcher peut
                        # être rejouée à la prochaine relance. Mémoriser une source NON
                        # INTERROGÉE revenait à cacher douze heures une réponse vide
                        # qu'aucune source n'avait donnée.
                        if (_time.time() < _fed_deadline[0] and _fname not in _fetcher_errors
                                and _fname not in _fetcher_skipped):
                            _completed_ok.add(_fname)
                    except Exception as _fe:
                        logger.warning(f"Populate {scenario_id} source future error: {_fe}")
                        with _counter_lock:
                            _fetcher_errors[_fname] = redact_keys(_fe)[:200]
            # CRITIQUE - sur Python <3.11, as_completed lève
            # concurrent.futures.TimeoutError (≠ TimeoutError natif). Sans
            # _FuturesTimeout dans le except, l'exception remontait, le bloc
            # plantait, et la reconstruction du corpus + le scoring + le passage à
            # « done » étaient SAUTÉS → corpus figé sur la base locale, statut
            # bloqué sur « running », résultats live perdus.
            except (TimeoutError, _FuturesTimeout):
                _fed_incomplete[0] = True   # fetch partiel → corpus non autorisé à rétrécir
                _done = sum(1 for _f in futures if _f.done())
                logger.warning(
                    f"Populate {scenario_id}: budget fédération {_fed_budget:.0f}s dépassé - "
                    f"{_done}/{len(futures)} sources terminées ; poursuite avec le corpus partiel "
                    f"(les sources lentes continuent en arrière-plan)."
                )
        finally:
            # wait=False : ne PAS bloquer sur les sources lentes. cancel_futures=True
            # annule celles qui n'ont pas démarré ; celles en cours s'arrêteront au
            # prochain tour de pagination grâce à _fed_deadline.
            executor.shutdown(wait=False, cancel_futures=True)
        for _fname in _completed_ok:
            if _fname in _no_cache:
                continue
            with _counter_lock:
                _links_now = list(_run_links.get(_fname, []))
                _totals_now = dict(_fetcher_totals.get(_fname) or {})
                _meta_now = ({"year_slices": {"pubmed": list(_year_slices["pubmed"])},
                              "year_slices_capped": {"pubmed": dict(_year_slices_capped.get("pubmed") or {})}}
                             if _fname == "_fetch_pubmed" and _year_slices.get("pubmed") else {})
            _save_source_cache(_fname, _qhash, query, _links_now, totals=_totals_now, meta=_meta_now)
        # ── L'issue de CHAQUE source, nommée ─────────────────────────────────
        # `_queried` ne distinguait pas « a répondu » de « a échoué » : un fetcher qui
        # lève est tout de même passé par la boucle, donc la ligne de couverture
        # annonçait « 12/12 sources interrogées » pour une recherche où la moitié
        # avaient échoué, et le PRISMA ne nommait que celles qui avaient rapporté
        # quelque chose. Cinq issues, exclusives, et aucune muette.
        for _fn in source_funcs:
            _n = _fn.__name__
            if _n in _cached:
                _outcome = "cached"
            elif _n in _fetcher_skipped:
                _outcome = "skipped"
            elif _n in _fetcher_errors:
                _outcome = "error"
            elif _n in _fetcher_cut or _n not in _returned:
                # Coupée par le budget : soit elle n'est pas revenue du tout, soit elle a
                # cessé de paginer en cours de route. Les deux se disent de la même
                # façon, parce que le lecteur doit en tirer la même conclusion : cette
                # source n'a pas été lue jusqu'au bout. Les enregistrements qu'elle a
                # tout de même rapportés restent dans le tableau par source ; c'est son
                # compte qui n'est pas un total, et son zéro qui ne vaut pas « rien ».
                _outcome = "cut_by_budget"
            else:
                # Ce que la source a RENVOYÉ, et non les lignes qu'on en a insérées.
                # `_returned` porte le compteur du fetcher, qui ne s'incrémente que sur une
                # ligne NOUVELLE : une source dont les 500 notices étaient déjà toutes en
                # bibliothèque rendait donc 0, et l'issue calculée était `empty`, c'est-à-dire
                # « cette base n'a rien sur le sujet ». La carte affichait « openalex 500 ·
                # aucun résultat », un compte et son démenti côte à côte.
                with _counter_lock:
                    _recs = sum(int(_ident_records.get(_k) or 0)
                                for _k in source_record_keys(_n))
                _outcome = "ok" if _recs > 0 else "empty"
            _fetcher_outcome[_n] = _outcome
            if _outcome in SOURCE_OUTCOMES_COUNTED:
                _queried.add(_n)
            # ── La position de lecture, gardée ou oubliée selon l'issue ──────────
            # Coupée : on garde où elle en était, pour que la prochaine reprise
            # continue au lieu de relire les mêmes pages. Terminée : on oublie, sinon
            # une relance de la même question repartirait du milieu.
            with _counter_lock:
                _pos = _fetch_pos.get(_n)
            _prev = _resume_state.get(_n) or {}
            _act = cursor_action(_outcome, _pos is not None, bool(_prev))
            if _act == "save":
                _save_fetch_cursor(scenario_id, _n, _qhash, _pos[0], _pos[1],
                                   int(_prev.get("rounds") or 0) + 1)
            elif _act == "clear":
                _clear_fetch_cursor(scenario_id, _n, _qhash)
        t_elapsed = _time.time() - t_start
        logger.info(f"Populate {scenario_id}: fédération terminée en {t_elapsed:.1f}s "
                    f"({len(_cached)} fetcher(s) depuis le cache, {len(_completed_ok)} mémorisé(s)) ; "
                    f"issues = {_outcome_summary(_fetcher_outcome)}")
    else:
        logger.info(f"Populate {scenario_id}: include_live=False - base locale uniquement")

    ingested = _ingested_total[0]
    errors = _errors_total[0]

    # ── Corpus = correspondance BOOLÉENNE (ou multi-sous-requêtes) sur base enrichie ─
    # Après ingestion des articles live, on recalcule l'appartenance au corpus via
    # EXACTEMENT le même helper que la recherche : _boolean_corpus_ids en mono-requête,
    # _multi_query_corpus_ids (union/intersection) en multi-sous-requêtes. Le corpus
    # devient donc strictement « résultat de la requête sur base locale ∪ live ».
    # allow_empty=True en multi : une intersection légitimement vide DOIT vider le corpus.
    # L'ensemble APPARIÉ par cette recherche (None tant qu'il n'est pas assemblé), et les
    # sources dont la réponse n'est pas complète : les chiffres PRISMA les lisent plus bas.
    _matched: list[int] | None = None
    _incomplete_sources: list[str] = []
    try:
        # Gel : à partir d'ici, ce qui arrive encore des sources ne fait plus partie de
        # cette recherche (cf. _corpus_frozen). Le corpus et ses chiffres sont figés ensemble.
        with _corpus_lock:
            _corpus_frozen[0] = True
            # Les enregistrements que les sources ont RENVOYÉS, recoupements compris : le
            # « identifiés » du PRISMA. Lu APRÈS le gel, et sous les deux verrous dans
            # l'ordre du reste du module (corpus puis compteurs) : lu avant, une source
            # lente pouvait encore lier des notices entre la lecture et le gel, et le
            # `total_found` servi au job était plus petit que le `records_identified`
            # servi dans la même réponse. `ingested`, lui, ne compte que les lignes
            # nouvelles : servi sous ce nom, il annonçait moins que ce que la recherche
            # avait identifié.
            with _counter_lock:
                total_found = sum(_ident_records.values())
                # Le même instant pour les compteurs du job : lus plus tôt, ou relus après
                # le scoring, ils ne décrivaient pas la même recherche que `total_found`.
                ingested = _ingested_total[0]
                errors = _errors_total[0]
                _sources_snapshot = dict(
                    (_user_scenario_populate_jobs.get(scenario_id) or {}).get("sources") or {})
        # Appartenance = re-match booléen LOCAL (base locale ∪ live) pour les sources par
        # mots-clés + la base existante…
        if _sub_queries:
            _final_ids = _multi_query_corpus_ids(_sub_queries, _combinator, filters)
        else:
            _final_ids = _boolean_corpus_ids(_boolean, filters)
        _local_capped[0] = _local_capped[0] or bool(getattr(_final_ids, "capped", False))
        # … ∪ SOURCE-UNION : les docs des sources booléennes-natives (PubMed, Europe PMC,
        # préprints EPMC) qui ont appliqué la VRAIE requête booléenne sont inclus DIRECTEMENT,
        # sans re-filtrage local (qui supprimait leurs correspondances MeSH/texte-intégral -
        # « 109 PubMed → 6 »). Leurs notices sans résumé restent, signalées comme les autres.
        # …SAUF en INTERSECTION multi-requêtes : le corpus doit matcher TOUTES les facettes,
        # or un doc booléen-natif ne matche que la requête PRINCIPALE (les fetchers live
        # interrogent la requête principale) → l'unir casserait l'intersection. On garde
        # alors le re-match local strict. L'intersection peut venir du combinateur
        # GLOBAL ou d'un « ET » posé sur UNE facette (bouton par sous-requête) : ne
        # tester que le combinateur global laissait passer ces docs et le « ET » entre
        # deux requêtes booléennes n'était pas appliqué au corpus.
        _union_native = not (_sub_queries and _facets_intersect(_sub_queries, _combinator))
        _n_native = len(_bool_native_ids) if _union_native else 0
        if _union_native:
            _final_ids = list(set(_final_ids) | _bool_native_ids)
        # Les critères propres de la recherche (bornes d'années, préprints exclus), que la
        # correspondance locale applique déjà et que l'union native n'a pas vus : l'ensemble
        # APPARIÉ, c'est-à-dire ce que cette recherche dit appartenir au corpus.
        _matched = _ids_within_criteria(_final_ids, filters)
        # ③ La fédération a-t-elle abouti ? Lu sur l'issue de chaque source, en plus des
        # deux compteurs : un fetcher qui lève hors de son propre `except` finit en `error`
        # sans passer par `_mark_source_error`.
        _incomplete_sources = sorted({_source_label(_n) for _n, _o in _fetcher_outcome.items()
                                      if _o in ("error", "cut_by_budget")})
        _fetch_ok = (not _fed_incomplete[0]) and _source_errors[0] == 0 and not _incomplete_sources
        # Ce qui peut être RETIRÉ du corpus :
        #  • fédération aboutie, ou base locale seule : tout lien que la recherche n'apparie
        #    pas, sauf ceux qu'un relecteur a jugés (_set_scenario_corpus) ;
        #  • fédération incomplète : rien de ce que le corpus avait AVANT la recherche,
        #    tant que ses critères le laissent (années, préprints). Une source en échec ne
        #    dit rien des liens qu'elle n'a pas ramenés ; la carte PRISMA compte ces liens
        #    gardés, nomme les sources, et demande de relancer. Seuls les liens que cette
        #    recherche a posés en route et qu'elle n'apparie pas sont retirés.
        # Un résultat vide ne vide le corpus qu'en multi-requêtes (une intersection
        # légitimement vide), en base locale seule (déterministe), ou après une
        # fédération aboutie ; sinon il ne touche à rien.
        _allow_empty = bool(_sub_queries) or (not include_live) or _fetch_ok
        _keep: list[int] | None = None
        if not _fetch_ok:
            _before = _previous_links
            if _before is None:
                # Relevé manqué au départ : les liens présents, au plus large.
                with engine.connect() as _pbc:
                    _before = [int(x) for x in _pbc.execute(text(
                        "SELECT document_id FROM article_scenarios WHERE scenario_id = :sid"),
                        {"sid": scenario_id}).scalars().all()]
            _keep = _ids_within_criteria(_before, filters)
        _n_corpus = _set_scenario_corpus(scenario_id, _matched, allow_empty=_allow_empty, keep=_keep)
        # « assemblé, AVANT nettoyage » : les lignes doublons sont retirées juste après -
        # le corpus retenu (article_count, « passés au screening » du PRISMA) est
        # journalisé plus bas avec les chiffres PRISMA. Étiqueté « final » auparavant, ce
        # nombre était lu comme le corpus.
        logger.info(f"Populate {scenario_id}: corpus assemblé (avant nettoyage) = {_n_corpus} docs "
                    f"(re-match local ∪ {_n_native} docs booléens-natifs PubMed/EPMC/préprints ; "
                    f"{'multi ' + '/'.join(_facet_ops(_sub_queries, _combinator)) if _sub_queries else 'mono'})"
                    + (f" ; fédération incomplète ({', '.join(_incomplete_sources) or 'budget'}) : "
                       f"les {len(_keep)} liens d'avant la recherche restent" if _keep is not None else ""))
    except Exception as _e_corpus:
        logger.warning(f"Rebuild corpus {scenario_id}: {_e_corpus}")

    try:
        # ── Les notices SANS résumé RESTENT au corpus ────────────────────────
        # Elles en étaient retirées ici (2 434 sur HPAI_last : les notices Crossref,
        # Semantic Scholar et OpenAlex que les éditeurs livrent sans résumé, les lettres,
        # les éditoriaux, le PubMed ancien), et le backfill qui pouvait leur trouver un
        # résumé passait après sans jamais les relier. Elles restent, signalées
        # (scenario_store.has_abstract_sql, lu à la lecture), hors du sous-ensemble
        # pertinent tant qu'un relecteur ne les inclut pas : le titre se juge à la main.
        # Un résumé comblé plus tard les rend éligibles d'elles-mêmes.

        # Les préprints d'une recherche qui les exclut, et ce qui sort de ses bornes
        # d'années, ne sont plus retirés ici après coup : l'assemblage ne lie que ce qui
        # satisfait ces critères (_ids_within_criteria), avant la dédup, et ne garde de
        # l'ancien corpus, quand la fédération n'a pas abouti, que ce qui les satisfait
        # aussi. Un lien qu'un relecteur a jugé reste, comme partout ailleurs.

        # ── Dédup INTRA-scénario : un seul lien par article distinct ─────────
        # Fige le compte de façon déterministe (cf. _dedup_scenario_links). Une copie
        # sans résumé gardée au détriment d'une copie complète du même article reçoit le
        # résumé de l'autre : la dédup le recopie sur le survivant.
        _n_dup_rows = _dedup_scenario_links(scenario_id)

        # ── Mettre à jour article_count (avant rerank) ──────────────────────
        # NB : on ne marque PAS encore populate_status='done' ici - le scoring
        # n'a pas tourné. Le marquer prématurément faisait paraître "prêt" un
        # scénario dont les similarity_score restaient NULL (compteurs divergents).
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE user_scenarios
                SET article_count = (
                    SELECT COUNT(DISTINCT ars.document_id) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                ),
                updated_at = NOW()
                WHERE id = :sid
            """), {"sid": scenario_id})

        # ── Chiffres PRISMA « identification » de CETTE recherche ────────────
        # Enregistrements par source (base locale comprise), documents distincts,
        # doublons (recoupements + lignes fusionnées), retirés pour d'autres raisons.
        # Stockés sur le scénario : le PRISMA les lit au lieu de compter un flag
        # `is_duplicate` que rien ne pose (→ « doublons retirés : 0 » à vie).
        _pi_corpus_total: int | None = None
        _pi_figures: dict | None = None
        try:
            with engine.connect() as _pc:
                _corpus_now = _pc.execute(text("""
                    SELECT COUNT(DISTINCT ars.document_id) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                """), {"sid": scenario_id}).scalar() or 0
            with _counter_lock:
                _recs_snapshot = dict(_ident_records)
                _totals_snapshot = dict(_source_totals)
                _ids_snapshot = list(_ident_docs)
                _year_slices_snapshot = {k: list(v) for k, v in _year_slices.items()}
                _year_slices_capped_snapshot = {k: dict(v) for k, v in _year_slices_capped.items()}
            # Ventilation des documents identifiés mais ABSENTS du corpus : préprints quand
            # la recherche les exclut (écartés avant la dédup), ou non liés (source par
            # mots-clés hors requête booléenne, ou ligne fusionnée par la dédup, d'où la
            # soustraction). Les notices sans résumé n'y sont plus : elles sont gardées, et
            # comptées à part parmi les passées au screening (`kept_no_abstract`).
            # « Passé au screening PAR cette recherche » = lié ET apparié par elle : un
            # document identifié qui n'est resté que parce que le corpus l'avait déjà
            # (fédération incomplète) ou qu'un relecteur en avait décidé compte parmi les
            # retraits, et revient sur la ligne des liens gardés (_kept_link_counts).
            _kept = {"reviewed": 0, "added": 0, "previous": 0}
            if _matched is not None:
                _kept = _kept_link_counts(scenario_id, _matched)
            _kept_no_abs, _not_linked, _pp_out = 0, 0, 0
            if _ids_snapshot:
                _is_pp = preprint_sql("d") if _no_preprints else "FALSE"
                _here = ("(EXISTS (SELECT 1 FROM article_scenarios a"
                         " WHERE a.scenario_id = :sid AND a.document_id = d.id)"
                         + (" AND d.id = ANY(CAST(:matched AS bigint[])))" if _matched is not None else ")"))
                with engine.connect() as _bc:
                    _br = _bc.execute(text(f"""
                        SELECT COUNT(*) FILTER (WHERE {_is_pp} AND NOT {_here})     AS preprints,
                               COUNT(*) FILTER (WHERE NOT {_is_pp} AND NOT {_here}) AS not_linked
                        FROM literature_document d
                        WHERE d.id = ANY(CAST(:ids AS bigint[]))
                    """), {"sid": scenario_id, "ids": _ids_snapshot,
                           **({"matched": list(_matched)} if _matched is not None else {})}).mappings().first()
                _pp_out = int(_br["preprints"] or 0)
                _not_linked = int(_br["not_linked"] or 0)
            with engine.connect() as _kc2:
                _kept_no_abs = int(_kc2.execute(text(f"""
                    SELECT COUNT(DISTINCT ars.document_id) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                      AND NOT {has_abstract_sql('d')}
                """), {"sid": scenario_id}).scalar() or 0)
            _figures = _prisma_identification_figures(
                _recs_snapshot, len(_ids_snapshot), _n_dup_rows or 0, int(_corpus_now),
                method="populate", federation_incomplete=bool(_fed_incomplete[0]),
                kept_no_abstract=_kept_no_abs,
                removed_not_matching=max(0, _not_linked - int(_n_dup_rows or 0)),
                # L'issue de chaque source lancée, et le plafond qui a servi : sans eux,
                # le tableau d'identification taisait les sources en échec ou coupées et
                # annonçait quand même un nombre de sources interrogées.
                source_outcomes=(dict(_fetcher_outcome) if include_live else {}),
                per_source_cap=int(_cap_strategy),
                keyword_fallback_sources=(list(_kw_fallback) if include_live else []),
                keyword_fallback_queries=(dict(_fallback_queries) if include_live else {}),
                source_error_reasons=(dict(_fetcher_errors) if include_live else {}),
                source_totals=(_totals_snapshot if include_live else {}),
                # Le mode, et le plafond de chaque source quand il diffère : Crossref, classé
                # par pertinence, garde le sien (RANKED_PASS_MAX) ; le passage titre et
                # résumé d'OpenAlex a le plafond commun. PubMed n'en a plus de propre :
                # au-delà des 10 000 identifiants qu'eutils rend par recherche, sa requête
                # est découpée par années (`year_slices`).
                search_mode=_search_mode,
                # Et l'endpoint classique de Semantic Scholar, qui ne rend pas plus de
                # 1 000 notices.
                source_caps=(({"crossref": int(_cap_ranked)}
                              if _cap_ranked < int(_cap_strategy) else {})
                             | dict(_source_caps_run)
                             if include_live else {}),
                title_abstract_sources=(sorted(_title_abstract_sources) if include_live else []),
                unranked_sources=(sorted(_unranked_sources) if include_live else []),
                split_queries=(dict(_split_queries) if include_live else {}),
                # PubMed au-delà de 10 000 notices : ses tranches d'années, et celles d'une
                # seule année encore au-delà (gardées à 10 000, leurs plus pertinentes).
                year_slices=(_year_slices_snapshot if include_live else {}),
                year_slices_capped=(_year_slices_capped_snapshot if include_live else {}),
                # L'exclusion des préprints, et ses trois moyens : serveurs non interrogés,
                # exclusion dans la requête, retrait avant le screening.
                preprints_excluded=_no_preprints,
                removed_preprints=_pp_out,
                preprints_excluded_at_source=(list(_preprints_at_source) if include_live else []),
                preprint_sources_not_searched=(list(_preprint_servers_skipped) if include_live else []),
                # Les facettes OU qu'aucune base n'a reçues, et la borne de la
                # correspondance locale quand elle a mordu.
                facets_skipped=(list(_facets_skipped) if include_live else []),
                local_match_capped=bool(_local_capped[0]),
                # Passés au screening sans que cette recherche les apparie : le corpus
                # d'avant une fédération incomplète (et les sources qui l'ont rendue
                # incomplète), puis les liens qu'un relecteur a jugés.
                kept_previous_links=_kept["previous"],
                kept_previous_failed_sources=list(_incomplete_sources),
                kept_reviewed_links=_kept["reviewed"],
                kept_added_links=_kept.get("added", 0))
            _store_prisma_identification(scenario_id, _figures)
            # Le même total pour tout le monde : le statut du job expose le corpus
            # RETENU (= article_count = « passés au screening » du PRISMA), et non le
            # seul compteur d'ingestion (local + nouveaux documents), qui n'est pas un
            # total de corpus et se lisait comme tel. Repris dans l'état final ci-dessous.
            _pi_corpus_total = int(_corpus_now)
            _pi_figures = _figures
            if _pipeline_callback is None:
                _job_now = _user_scenario_populate_jobs.get(scenario_id)
                if _job_now is not None:
                    _job_now["corpus_total"] = _pi_corpus_total
                    _job_now["prisma_identification"] = _pi_figures
            logger.info(f"Populate {scenario_id}: PRISMA identification = "
                        f"{_figures['records_identified']} enregistrements, "
                        f"{_figures['duplicates_removed']} doublons, "
                        f"{_figures['unique_records']} uniques, "
                        f"{_figures['removed_other_reasons']} retirés (autres raisons), "
                        f"{_figures['kept_previous_links']} gardés de l'ancien corpus, "
                        f"{_figures['kept_reviewed_links']} gardés pour une décision, "
                        f"{_figures['records_screened']} au screening.")
        except Exception as _e_pi:
            logger.warning(f"Populate {scenario_id}: chiffres PRISMA non stockés: {_e_pi}")

        # ── Scores sémantiques (cosinus) - SANS suppression ─────────────────
        # Soft filter : le seuil filtre l'affichage et l'aval, JAMAIS par
        # suppression. Réduire le seuil fait donc réapparaître des articles.
        _set_phase("scoring")
        _n_scored = 0
        _scoring_failed = False
        _auto_ok = [False]           # corpus scoré → le pipeline complet peut enchaîner
        try:
            _backfill_title_abstract_chunks(scenario_id)  # docs liés sans chunk résumé
            _n_scored = _run_semantic_rerank_inline(scenario_id, query)
            logger.info(f"Post-populate scoring {scenario_id}: {_n_scored} articles scorés (cosinus, aucune suppression).")
        except Exception as _e_rr:
            _scoring_failed = True
            logger.warning(f"scoring post-populate {scenario_id}: {_e_rr}")

        # ── Honnêteté de l'état : trois issues DISTINCTES, pas deux ──────────
        # `_run_semantic_rerank_inline` avale ses erreurs et renvoie 0 (OpenAI
        # indisponible, quota épuisé). Deux états ne suffisaient pas à le dire :
        #   - 'done'     : corpus construit ET scoré, tout est utilisable ;
        #   - 'unranked' : corpus construit, AUCUN score produit. Les articles sont
        #                  là et se lisent, mais l'ordre et le seuil ne veulent rien
        #                  dire tant qu'ils ne sont pas calculés ;
        #   - 'error'    : la construction du corpus elle-même a échoué (posé
        #                  ailleurs : lancement, redémarrage).
        # Le marquer 'error' jetait un corpus parfaitement lisible : sur une panne
        # OpenAI, l'utilisateur voyait « la construction du corpus a échoué » alors
        # que ses articles étaient en base. 'done' mentait dans l'autre sens.
        _scoring_state = "ok"
        try:
            with engine.begin() as conn:
                _scorable = conn.execute(text(f"""
                    SELECT COUNT(DISTINCT ars.document_id) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                      AND {has_abstract_sql('d')}
                """), {"sid": scenario_id}).scalar() or 0
                _ok = (not _scoring_failed) and (_n_scored > 0 or _scorable == 0)
                _auto_ok[0] = bool(_ok)
                _scoring_state = "ok" if _ok else "unranked"
                conn.execute(text("""
                    UPDATE user_scenarios SET populate_status = :st, updated_at = NOW() WHERE id = :sid
                """), {"st": "done" if _ok else "unranked", "sid": scenario_id})
                if not _ok:
                    logger.warning(f"Populate {scenario_id}: scoring n'a produit aucun score "
                                   f"({_scorable} articles scorables) → populate_status='unranked' "
                                   f"(corpus construit, non classé).")
        except Exception as _e_st:
            logger.warning(f"populate_status update {scenario_id}: {_e_st}")

        # ── Le corpus est scoré (cosinus) → on le publie MAINTENANT ──────────
        # Le cross-encoder Cohere (plus lent) et les visualisations tournent
        # ENSUITE en arrière-plan : l'utilisateur voit les résultats ordonnés par
        # cosinus immédiatement, puis la liste se réordonne quand le rerank arrive
        # (rerank_status). Plus d'attente synchrone sur l'API Cohere.
        _cohere_enabled = bool(os.getenv("COHERE_API_KEY"))
        if _pipeline_callback is None:
            # L'instantané pris au gel, en COPIE : le dict vivant du job était publié tel
            # quel, et tout ce qui l'aurait encore modifié réécrivait le statut final.
            _sources_final = dict(_sources_snapshot if _sources_snapshot is not None else
                                  _user_scenario_populate_jobs.get(scenario_id, {}).get("sources", {}))
            _src_parts = [f"{src}: {cnt}" for src, cnt in _sources_final.items() if cnt > 0]
            _src_summary = " | ".join(_src_parts) if _src_parts else "aucune source"
            _user_scenario_populate_jobs[scenario_id] = {
                # Le statut SERVI à l'interface dit la même chose que celui écrit en base.
                # Il laissait « done » quand le scoring n'avait produit aucun score : la
                # page affichait une recherche terminée dont tous les scores sont NULL
                # (classement et seuil sans signification), sans un mot nulle part.
                "status": "done" if _auto_ok[0] else "unranked",
                "phase": "done",
                "search_mode": _search_mode,
                "exclude_preprints": _no_preprints,
                **({} if _auto_ok[0] else {
                    # PAS une erreur de recherche : le corpus est construit et lisible,
                    # c'est son CLASSEMENT qui manque. L'interface l'affiche donc avec un
                    # avertissement, au lieu de jeter des articles qui sont bien là.
                    "scoring": {
                        "ok": False,
                        "reason_code": "no_scores",
                        "reason": "Le scoring sémantique n'a produit aucun score : corpus "
                                  "construit mais non classé (clé OpenAI indisponible ?). "
                                  "L'ordre des articles et le seuil de similarité ne "
                                  "s'appliquent pas tant qu'il n'a pas tourné.",
                    },
                }),
                # Sources rejouées depuis le cache : l'état final remplace le dict du job,
                # la liste serait perdue et la page de résultats ne pourrait plus dire
                # pourquoi la recherche a pris trois secondes au lieu de trois minutes.
                "cached_sources": sorted(_cached) if include_live else [],
                "rerank_status": "running" if _cohere_enabled else "skipped",
                "ingested": ingested,
                # `ingested` = base locale + NOUVEAUX documents : un compteur de travail,
                # pas un total de corpus. Le corpus retenu est `corpus_total`.
                "corpus_total": _pi_corpus_total,
                "prisma_identification": _pi_figures,
                "errors": errors,
                "total_found": total_found,
                "sources": _sources_final,
                # Ce message est la ligne qu'un relecteur citerait pour justifier la
                # couverture de la recherche : il dit ce qui a RÉELLEMENT été interrogé,
                # pas « 13 sources » en toutes circonstances (base locale seule, budget
                # de fédération épuisé, sources rejouées depuis le cache).
                "message": (
                    f"{ingested} articles ingérés, {errors} erreurs. "
                    + ("Base locale uniquement (sources externes non interrogées)."
                       if not include_live else
                       f"Sources interrogées : {len(_queried)}/{len(source_funcs)} "
                       f"({_src_summary})"
                       + _coverage_caveat(_fetcher_outcome)
                       + (f" ; préprints exclus (non interrogés : {', '.join(_preprint_servers_skipped)})"
                          if _no_preprints else "")
                       + (f" ; {len(_cached)} rejouée(s) depuis le cache de la recherche identique."
                          if _cached else "."))
                ),
                # L'issue de chaque source, servie telle quelle : la ligne de couverture
                # résume, le panneau peut nommer.
                "source_outcomes": {_source_label(k): v for k, v in _fetcher_outcome.items()},
                # La correspondance locale a atteint sa borne (LIMIT sans ORDER BY) : le
                # corpus est alors un sous-ensemble arbitraire des correspondances.
                "local_match_capped": bool(_local_capped[0]),
            }

        # Arrière-plan : cross-encoder (réordonne le sous-ensemble pertinent) puis
        # clustering UMAP/HDBSCAN + knowledge graph (cache DB). Réservé au chemin
        # /populate (le pipeline complet a ses propres étapes).
        # Le garde `if _pipeline_callback is None` réservait ce bloc au chemin /populate,
        # donc le cross-encoder n'était JAMAIS appelé quand le pipeline complet tournait,
        # c'est-à-dire sur le chemin normal. Huit scénarios de production n'ont aucun
        # score de rerank pour cette seule raison, et l'étape 4 du pipeline, qui s'appelle
        # « rerank », est un cosinus qui n'écrit jamais rerank_score : l'utilisateur
        # voyait donc une étape de rerank réussir sans qu'un seul score soit produit.
        if True:
            def _post_done_bg(_sid, _query):
                try:
                    # Après une recherche, la question n'a pas changé : seuls les articles
                    # que la recherche vient d'ajouter n'ont pas de note. Les renvoyer tous
                    # à Cohere coûtait un corpus entier d'appels payants par recherche.
                    from .relevance import needs_full_score_pass, note_full_score_pass
                    _full = needs_full_score_pass(_sid, _query, "rerank")
                    _ce = _run_cross_encoder_rerank(_sid, _query, only_missing=not _full)
                    if _full and not _ce.get("stopped") and not _ce.get("skipped_no_key"):
                        note_full_score_pass(_sid, _query, "rerank")
                    if _ce.get("scored"):
                        logger.info(f"Post-populate cross-encoder {_sid}: {_ce['scored']}/"
                                    f"{_ce['candidates']} notés, {_ce['batches_failed']} lots en échec"
                                    f" ({'complet' if _full else 'manques seulement'}).")
                except Exception as _ece:
                    logger.warning(f"cross-encoder arrière-plan {_sid}: {_ece}")
                finally:
                    _job = _user_scenario_populate_jobs.get(_sid)
                    if _job is not None:
                        _job["rerank_status"] = "done"
                try:
                    # clustering → cache DB, résumés dans la langue de l'interface qui a
                    # lancé la recherche (sinon la première ouverture attendait le LLM)
                    _run_clustering_background(_sid, True, lang)
                except Exception as _e1:
                    logger.warning(f"Précalcul clustering {_sid}: {_e1}")
                try:
                    _precompute_user_kg(_sid)                # knowledge graph → cache DB
                except Exception as _e2:
                    logger.warning(f"Précalcul KG {_sid}: {_e2}")
                # Enrichissement COMPLET automatique (embeddings, PICO, métadonnées, résumés,
                # brief, variables et modèle, actions, carte des concepts) dès que le corpus
                # est construit et scoré : un scénario est prêt sans qu'on ait à l'épingler.
                # AUTO_PIPELINE_AFTER_SEARCH=0 pour s'en tenir à la recherche seule ;
                # `auto_pipeline=False` pour cette recherche-ci (une recherche dont on
                # enrichit soi-même le sous-ensemble pertinent, par exemple).
                if _auto_ok[0] and _auto_pipeline_after_search() and auto_pipeline:
                    try:
                        from .scenarios import _launch_full_pipeline
                        _st = _launch_full_pipeline(_sid, lang=lang)
                        logger.info(f"Auto full-pipeline après recherche {_sid}: {_st}")
                    except Exception as _e3:
                        logger.warning(f"Auto full-pipeline {_sid}: {_e3}")
            try:
                import threading as _vth
                _vth.Thread(target=_post_done_bg, args=(scenario_id, query), daemon=True).start()
            except Exception as _e_viz:
                logger.warning(f"Tâches arrière-plan {scenario_id}: {_e_viz}")

        logger.info(f"Populate user_scenario {scenario_id}: {ingested} articles ingérés "
                    f"({len(source_funcs)} sources, parallèle).")
        return ingested

    except Exception as e:
        logger.error(f"Populate user_scenario {scenario_id} fatal: {e}", exc_info=True)
        if _pipeline_callback is None:
            _user_scenario_populate_jobs[scenario_id] = {
                "status": "error",
                "error": str(e),
                "ingested": _ingested_total[0],
            }
        return 0


def _run_user_scenario_full_pipeline(scenario_id: str, query: str, filters: dict,
                                     max_results: int = LIVE_MAX_PER_SOURCE, lang: str | None = None) -> None:
    # max_results : MÊME plafond par source que le populate appelé par l'API
    # (LIVE_MAX_PER_SOURCE, 2000 par défaut). Il valait 500 ici, d'où deux corpus
    # différents pour la même requête selon qu'elle partait de l'interface ou de l'API
    # (25 140 contre 30 511 documents sur « (AI OR ML OR DL) AND infection »).
    """
    Pipeline complet d'enrichissement pour un scénario utilisateur.
    Ordre optimal :
    1. ingest    – Ingestion multi-sources (PubMed+OpenAlex+Crossref+EuropePMC+Preprints+SemanticScholar+DOAJ+ClinicalTrials.gov+CORE+arXiv+OpenAIRE)
    2. fulltext  – Récupération full-text (PMC→EuropePMC→Unpaywall) pendant que les IDs sont frais
    3. embed     – Embeddings sur title+abstract+fulltext chunks (contenu enrichi)
    4. rerank    – Score cosinus via pgvector (pas de re-embedding API)
    5. pico      – Extraction PICO (LLM) : le résumé, plus les méthodes et les résultats du
                   texte intégral quand il existe, et le résultat principal de l'article ;
                   puis ce résultat pour les PICO pertinents écrits avant ce champ
    6. metadata  – Extraction métadonnées étude (LLM)
    7. clustering – K-means sur embeddings pgvector
    """
    from .evidence import _generate_evidence_brief_llm  # lazy: evidence is loaded after this module
    from .variables import _generate_variables_from_pico  # lazy: variables is loaded after this module
    # Ces deux noms n'étaient importés nulle part dans ce module : l'étape cross-encoder
    # (depuis #322) et les étapes PICO et métadonnées (depuis #278) levaient un NameError à
    # chaque pipeline complet, rattrapé par le try de l'étape et rangé en « error ». Le
    # PICO, extrait aussi en arrière-plan à l'ingestion, ne manquait guère ; les
    # MÉTADONNÉES, elles, n'avaient pas d'autre chemin automatique : sur HPAI_last, 23 %
    # des articles en portaient (type d'étude, effectif, risque de biais, quality_score),
    # et les niveaux de preuve du brief se calculaient sur ce quart.
    # `tests/test_no_undefined_names.py` passe désormais tout le paquet au crible.
    from .relevance import _run_cross_encoder_rerank, _run_semantic_rerank_inline  # lazy: relevance is loaded after this module
    import time as _time

    # L'étape 4 s'appelait « rerank » et n'était qu'un cosinus sur les embeddings
    # stockés : elle n'écrivait JAMAIS rerank_score. L'utilisateur voyait donc une étape
    # de rerank réussir sans qu'un seul score de rerank existe, et le cross-encoder, lui,
    # ne tournait que sur le chemin /populate. Deux étapes, nommées pour ce qu'elles font.
    STEP_ORDER = ["ingest", "embed", "semantic_scoring", "cross_encoder", "fulltext",
                  "pico", "metadata", "extraction", "clustering", "knowledge_graph",
                  "evidence", "variables", "actions"]

    def update_step(step: str, status: str, **kwargs):
        job = _user_scenario_pipeline_jobs.get(scenario_id, {})
        job["current_step"] = step
        job["steps"] = job.get("steps", {})
        job["steps"][step] = {"status": status, **kwargs}
        job["overall_status"] = "running"
        _user_scenario_pipeline_jobs[scenario_id] = job
        step_idx = STEP_ORDER.index(step) if step in STEP_ORDER else 0
        progress = int((step_idx / len(STEP_ORDER)) * 100)
        try:
            with engine.begin() as _conn:
                _conn.execute(text("""
                    UPDATE user_scenarios
                    SET pipeline_status = 'running',
                        pipeline_step = :step,
                        pipeline_progress = :progress,
                        pipeline_started_at = COALESCE(pipeline_started_at, NOW())
                    WHERE id = :sid
                """), {"step": step, "progress": progress, "sid": scenario_id})
        except Exception as _e:
            logger.warning(f"update_step DB write failed: {_e}")
        logger.info(f"Pipeline {scenario_id} [{step}]: {status} {kwargs}")

    def ingest_callback(event: str, value):
        if event == "pubmed_found":
            update_step("ingest", "running", found=value)

    # Le seuil AU DÉPART de ce passage : s'il a changé en route (quelqu'un a déplacé le
    # curseur pendant que le pipeline tournait), les étapes déjà faites l'ont été sur un
    # autre sous-ensemble, et il faut repasser. Sans cela, appliquer un seuil pendant un
    # pipeline laissait un corpus analysé moitié sur l'ancien, moitié sur le nouveau.
    try:
        _thr_at_start = scenario_threshold_info(scenario_id)["threshold"]
    except Exception:                                        # noqa: BLE001
        _thr_at_start = None
    _user_scenario_pipeline_jobs[scenario_id] = {
        "overall_status": "running",
        "current_step": "ingest",
        "lang": lang or "fr",
        "threshold_at_start": _thr_at_start,
        "steps": {k: {"status": "pending"} for k in STEP_ORDER},
    }

    try:
        # ── Étape 1 : Ingestion multi-sources ────────────────────────────────────
        update_step("ingest", "running")
        # Si le scénario a DÉJÀ été peuplé (populate terminé - typiquement par la
        # recherche, ensuite sauvegardée en scénario), on NE RE-FÉDÈRE PAS. Une 2ᵉ
        # fédération (live, non déterministe) recalculerait le corpus booléen sur une
        # base entre-temps enrichie par les threads d'arrière-plan de la 1ʳᵉ
        # fédération → le compteur dérivait (recherche = 7, carte = 8). On réutilise
        # le corpus existant et on passe directement à l'enrichissement.
        with engine.connect() as _psc:
            _ps_row = _psc.execute(text(
                "SELECT populate_status, "
                "(SELECT COUNT(DISTINCT document_id) FROM article_scenarios WHERE scenario_id = :sid) AS n "
                "FROM user_scenarios WHERE id = :sid"
            ), {"sid": scenario_id}).mappings().first()
        _corpus_ready = bool(_ps_row and _ps_row.get("populate_status") == "done" and (_ps_row.get("n") or 0) > 0)
        if _corpus_ready:
            ingested = int(_ps_row["n"])
            logger.info(f"Pipeline {scenario_id}: corpus déjà construit ({ingested} docs, populate=done) - "
                        f"fédération sautée pour éviter la dérive du compteur.")
        else:
            ingested = _run_user_scenario_populate(
                scenario_id, query, filters, max_results, _pipeline_callback=ingest_callback
            )
        # `ingested` from populate is a raw cumulative counter (includes ON CONFLICT duplicates
        # and DB-cache articles). Get the real unique count from DB as the source of truth.
        with engine.connect() as _ic:
            _real_ingested = _ic.execute(text(
                "SELECT COUNT(DISTINCT document_id) FROM article_scenarios WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).scalar() or 0
        update_step("ingest", "done", ingested=_real_ingested, api_results_raw=ingested)

        if _real_ingested == 0:
            _user_scenario_pipeline_jobs[scenario_id]["overall_status"] = "done"
            # Le détail de ce que chaque source a fait est sur la carte PRISMA : un nombre de
            # sources écrit ici en dur était faux dès qu'une source échouait ou n'était pas
            # lancée (recherche sans préprints).
            _user_scenario_pipeline_jobs[scenario_id]["message"] = (
                "Aucun article trouvé (le détail par source est sur la carte PRISMA).")
            return


        # ── Étape 3 : Embeddings (title_abstract + fulltext_section - contenu enrichi) ────────
        update_step("embed", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from llm_usage import MeteredOpenAI as _OAI_emb
                _emb_client = _OAI_emb(api_key=openai_key)
                with engine.connect() as _conn_emb:
                    _chunks_to_embed = _conn_emb.execute(text("""
                        SELECT c.id, c.document_id, c.content
                        FROM document_chunk c
                        JOIN article_scenarios ars ON ars.document_id = c.document_id
                        WHERE ars.scenario_id = :sid
                          AND c.embedding IS NULL
                          AND c.chunk_type IN ('title_abstract', 'fulltext_section')
                          AND LENGTH(c.content) > 20
                        ORDER BY c.id
                    """), {"sid": scenario_id}).mappings().fetchall()
                _emb_total = len(_chunks_to_embed)
                _emb_docs_total = len({r["document_id"] for r in _chunks_to_embed})
                _emb_done = 0
                _emb_docs_done: set = set()
                _emb_errors = 0
                _emb_batch_size = 100
                for _bi in range(0, _emb_total, _emb_batch_size):
                    _batch = _chunks_to_embed[_bi:_bi + _emb_batch_size]
                    try:
                        _texts = [_truncate_to_tokens(r["content"]) for r in _batch]
                        _emb_resp = _emb_client.embeddings.create(
                            model=_model("embedding"),
                            input=_texts
                        )
                        # Batch all updates in a single transaction (not one per chunk)
                        with engine.begin() as _conn_upd:
                            for _k, _emb_data in enumerate(_emb_resp.data):
                                _vec_str = "[" + ",".join(str(x) for x in _emb_data.embedding) + "]"
                                _conn_upd.execute(text("""
                                    UPDATE document_chunk
                                    SET embedding = CAST(:vec AS vector)
                                    WHERE id = :cid
                                """), {"vec": _vec_str, "cid": _batch[_k]["id"]})
                                _emb_done += 1
                                _emb_docs_done.add(_batch[_k]["document_id"])
                    except Exception as _emb_e:
                        _emb_errors += len(_batch)
                        logger.warning(f"Embed batch {_bi}: {_emb_e}")
                    update_step("embed", "running",
                                docs_done=len(_emb_docs_done), docs_total=_emb_docs_total,
                                chunks_done=_emb_done, chunks_total=_emb_total,
                                pct=round(_emb_done / _emb_total * 100, 1) if _emb_total > 0 else 0)
                update_step("embed", "done",
                            docs_embedded=len(_emb_docs_done), docs_total=_emb_docs_total,
                            chunks_embedded=_emb_done, chunks_total=_emb_total,
                            errors=_emb_errors)
            else:
                update_step("embed", "skipped", reason="Clé OpenAI non configurée")
        except Exception as _emb_ex:
            update_step("embed", "error", error=str(_emb_ex))

        # ── Étape 4 : Score SÉMANTIQUE via pgvector (cosinus sur embeddings stockés) ───
        # Ce qui est écrit ici est `similarity_score`, et rien d'autre. Le cross-encoder
        # est l'étape 5, séparée, parce que c'est elle qui écrit `rerank_score`.
        update_step("semantic_scoring", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                # LA fonction de score de la recherche, et non une copie. Celle-ci plongeait
                # `query[:2000]`, la requête BRUTE avec ses tags PubMed, quand la recherche
                # plonge `embedding_text_for_query` (requête entière, sans tags, #327) : deux
                # vecteurs de requête pour un même scénario, selon le dernier chemin passé.
                # Mesuré sur HPAI_last : les 6 453 articles communs aux deux chemins ont
                # tous changé de score (−0,024 en moyenne), et le seuil réglé pour garder
                # 1 000 articles n'en gardait plus que 288 après une recherche relancée.
                # Tout rebalayer, ou seulement combler les manques ? Le score est le
                # cosinus question-article : à question inchangée, un article déjà scoré
                # donnerait exactement le même nombre. Le pipeline rebalayait pourtant
                # tout à chaque passage, et il repart à chaque redémarrage de l'API.
                from .relevance import needs_full_score_pass, note_full_score_pass
                _full_sem = needs_full_score_pass(scenario_id, query, "semantic")
                n_reranked = _run_semantic_rerank_inline(scenario_id, query,
                                                         missing_only=not _full_sem)
                if _full_sem:
                    note_full_score_pass(scenario_id, query, "semantic")
                update_step("semantic_scoring", "done", updated=n_reranked, full_pass=_full_sem)

                # Seuil SÉMANTIQUE = SOFT : on ne supprime JAMAIS d'article du
                # corpus. Le corpus = résultat INTÉGRAL de la requête booléenne
                # (base locale ∪ live) ; le seuil ne fait que distinguer, à
                # l'affichage et en aval (page scénario, modèle), les articles
                # « au-dessus du seuil » (mis en avant) des « sous le seuil »
                # (conservés). Voir get_user_scenario_corpus (above/below_threshold).
                # On recalcule simplement article_count sur le corpus complet.
                try:
                    with engine.begin() as _ac:
                        _ac.execute(text("""
                            UPDATE user_scenarios
                            SET article_count = (
                                SELECT COUNT(DISTINCT ars.document_id)
                                FROM article_scenarios ars
                                JOIN literature_document d ON d.id = ars.document_id
                                WHERE ars.scenario_id = :sid
                                  AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                            )
                            WHERE id = :sid
                        """), {"sid": scenario_id})
                except Exception as _ce:
                    logger.warning(f"Post-rerank article_count update {scenario_id}: {_ce}")
            else:
                update_step("semantic_scoring", "skipped", reason="Clé OpenAI non configurée")
        except Exception as e:
            update_step("semantic_scoring", "error", error=str(e))

        # ── Étape 5 : Cross-encoder (c'est CELLE-CI qui écrit rerank_score) ────────────
        # Elle manquait au pipeline complet, c'est-à-dire au chemin normal : huit
        # scénarios de production n'avaient aucun score de rerank pour cette seule
        # raison, et le second seuil ne pouvait donc rien filtrer chez eux.
        update_step("cross_encoder", "running")
        try:
            if os.getenv("COHERE_API_KEY"):
                # Celle-ci appelle une API PAYANTE, un lot d'articles à la fois : la
                # relancer en entier sur un corpus déjà noté, c'est renvoyer 81 766
                # articles à Cohere pour réécrire les mêmes nombres. À question
                # inchangée, on ne note donc que ce qui n'a pas de note.
                from .relevance import needs_full_score_pass, note_full_score_pass
                _full_ce = needs_full_score_pass(scenario_id, query, "rerank")
                _ce_res = _run_cross_encoder_rerank(scenario_id, query, only_missing=not _full_ce)
                if _ce_res.get("skipped_no_key"):
                    update_step("cross_encoder", "skipped", reason="Clé Cohere non configurée")
                else:
                    if _full_ce and not _ce_res.get("stopped"):
                        note_full_score_pass(scenario_id, query, "rerank")
                    # `stopped` : le fournisseur a refusé et le balayage s'est arrêté avant
                    # la fin (les articles non notés restent sans score, donc gardés).
                    update_step("cross_encoder", "done",
                                scored=_ce_res.get("scored", 0),
                                candidates=_ce_res.get("candidates", 0),
                                batches_failed=_ce_res.get("batches_failed", 0),
                                stopped=_ce_res.get("stopped"),
                                full_pass=_full_ce)
            else:
                update_step("cross_encoder", "skipped", reason="Clé Cohere non configurée")
        except Exception as _ce_ex:
            update_step("cross_encoder", "error", error=str(_ce_ex))

        # ── Étape 5 bis : le texte intégral en accès ouvert, dans la portée du pipeline ──
        # Après les scores : la portée « relevant » se lit sur eux, et ils ne lisent que le
        # titre et le résumé (étape 4), si bien que le texte n'avait rien à faire avant.
        # L'étape téléchargeait le texte de TOUT le scénario (17 510 articles pour HPAI_last,
        # dont 1 075 pertinents). Tant qu'une personne n'a pas choisi le seuil, le sous-
        # ensemble pertinent est celui du seuil de départ (0,45), qui peut prendre un tiers
        # d'un corpus : le texte intégral, le PICO et les métadonnées l'ATTENDENT, et le
        # statut du pipeline le dit (`waiting_threshold`).
        _enrich_scope = pipeline_enrich_scope()
        _enrich_waits = _enrich_scope == "relevant" and not scenario_threshold_info(scenario_id)["chosen"]
        _WAIT_REASON = ("Seuil de pertinence non choisi : réglez-le, puis relancez le pipeline "
                        "ou les lots du panneau d'enrichissement")
        if _enrich_waits:
            _user_scenario_pipeline_jobs[scenario_id]["waiting_threshold"] = True
            update_step("fulltext", "skipped", reason=_WAIT_REASON)
        else:
            update_step("fulltext", "running")
            try:
                _ft = download_fulltext(
                    rows_without_fulltext(scenario_id, _enrich_scope),
                    progress=lambda d, t, f: update_step("fulltext", "running", done=d, total=t, paywall=f,
                                                         pct=round(d / t * 100, 1) if t else 0))
                update_step("fulltext", "done", fetched=_ft["fetched"], total=_ft["total"],
                            paywall_or_failed=_ft["failed"], scope=_enrich_scope)
            except Exception as e:
                update_step("fulltext", "error", error=str(e))

        # ── Étape 6 : Extraction PICO ─────────────────────────────────────────────────────
        update_step("pico", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and not _enrich_waits:
                from llm_usage import MeteredOpenAI as _OAI
                from datetime import datetime, timezone
                from .pico_map import PicoUnreadable, extract_pico, pico_fulltext_chunks
                _client = _OAI(api_key=openai_key, timeout=90.0)
                # Le PICO est AFFICHÉ tel quel (onglet PICO, export CSV) : sa PROSE suit
                # donc la langue de l'interface, comme le reste du pipeline. `study_design`
                # et `finding_direction` restent en anglais : ce sont des identifiants que
                # le code regroupe et compte (digest, carte des concepts), et les traduire
                # scinderait « cohort » et « cohorte » en deux catégories pour le même devis.
                # Le prompt, le texte lu et la normalisation sont ceux de tous les
                # rédacteurs PICO (api/pico_map) : cette étape ne lisait que le résumé.
                _pico_lang = _llm_lang_directive(lang)
                with engine.connect() as conn:
                    pico_rows = conn.execute(text("""
                        SELECT ld.id, ld.title, ld.abstract
                        FROM literature_document ld
                        JOIN article_scenarios asn ON asn.document_id = ld.id
                        WHERE asn.scenario_id = :sid
                          AND ld.project_context = 'literev'
                          AND (""" + _enrich_gate() + """)
                          AND (ld.pico_json IS NULL OR (ld.pico_json->>'pico_confidence')::float < 0.5)
                          AND ld.abstract IS NOT NULL AND length(ld.abstract) > 50
                          AND COALESCE(ld.pico_attempts, 0) < 3  -- borne les échecs déterministes (token-bleed)
                        ORDER BY ld.id
                    """), {"sid": scenario_id}).mappings().fetchall()

                pico_extracted = 0
                pico_errors = 0
                for row in pico_rows:
                    try:
                        with engine.connect() as _ftc:
                            _chunks = pico_fulltext_chunks(_ftc, row["id"])
                        pico, _pico_src = extract_pico(_client, row["title"], row["abstract"], _chunks,
                                                       lang_directive=_pico_lang,
                                                       label=f"Pipeline PICO {row['id']}")
                    except PicoUnreadable as e:
                        # Réponse reçue mais inutilisable → COMPTE la tentative (borne le
                        # token-bleed d'une sortie déterministe malformée).
                        logger.warning(f"Pipeline PICO parse article {row['id']}: {e}")
                        try:
                            with engine.begin() as conn:
                                conn.execute(text(
                                    "UPDATE literature_document SET pico_attempts = COALESCE(pico_attempts, 0) + 1 WHERE id = :aid"
                                ), {"aid": row["id"]})
                        except Exception:
                            pass
                        pico_errors += 1
                        _time.sleep(0.05)
                        continue
                    except Exception as e:
                        logger.warning(f"Pipeline PICO API article {row['id']}: {e}")
                        pico_errors += 1
                        _time.sleep(0.05)
                        continue  # transitoire - ne PAS consommer une tentative
                    try:
                        with engine.begin() as conn:
                            conn.execute(text("""
                                UPDATE literature_document
                                SET pico_json = CAST(:pico AS jsonb), pico_extracted_at = :ts,
                                    pico_attempts = COALESCE(pico_attempts, 0) + 1
                                WHERE id = :article_id
                            """), {"pico": json.dumps(pico), "ts": datetime.now(timezone.utc), "article_id": row["id"]})
                        pico_extracted += 1
                    except Exception as e:
                        logger.warning(f"Pipeline PICO write article {row['id']}: {e}")
                        pico_errors += 1
                    _time.sleep(0.05)
                # Le résultat principal des articles PERTINENTS dont le PICO précède ce champ
                # (écrit avant qu'un rédacteur ne le demande) : une passe courte, une fois
                # par article et mise en cache, sur tous ceux qui en manquent. Le brief,
                # les actions et le spec, plus loin dans ce pipeline, comptent les
                # résultats sur le corpus complet.
                _findings: dict[str, Any] = {}
                try:
                    from .enrichment import run_findings_pass
                    _findings = run_findings_pass(scenario_id, lang_directive=_pico_lang)
                except Exception as _fe:
                    logger.warning(f"Pipeline résultats principaux {scenario_id}: {_fe}")
                    _findings = {"status": "error", "error": str(_fe)[:300]}
                # Total coverage from DB (includes previously extracted articles)
                with engine.connect() as _pico_stat_conn:
                    _pico_total_in_scenario = _pico_stat_conn.execute(text("""
                        SELECT COUNT(*) FROM article_scenarios WHERE scenario_id = :sid
                    """), {"sid": scenario_id}).scalar() or 0
                    _pico_total_with = _pico_stat_conn.execute(text("""
                        SELECT COUNT(*) FROM literature_document ld
                        JOIN article_scenarios ars ON ars.document_id = ld.id
                        WHERE ars.scenario_id = :sid AND ld.pico_json IS NOT NULL
                    """), {"sid": scenario_id}).scalar() or 0
                update_step("pico", "done",
                            extracted_this_run=pico_extracted,
                            total_with_pico=_pico_total_with,
                            total_articles=_pico_total_in_scenario,
                            pct=round(_pico_total_with / _pico_total_in_scenario * 100, 1) if _pico_total_in_scenario > 0 else 0,
                            errors=pico_errors,
                            findings_status=_findings.get("status"),
                            findings_filled=_findings.get("done", 0),
                            findings_failed=_findings.get("failed", 0))
            else:
                update_step("pico", "skipped",
                            reason=_WAIT_REASON if _enrich_waits else "Clé OpenAI non configurée")
        except Exception as e:
            update_step("pico", "error", error=str(e))

        # ── Étape 7 : Extraction métadonnées ─────────────────────────────────
        update_step("metadata", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and not _enrich_waits:
                from llm_usage import MeteredOpenAI as _OAI2
                from datetime import datetime, timezone
                from .pico_map import json_completion
                # Un délai, comme tous les autres appels : sans lui, une requête pendue
                # bloquait l'étape, et le pipeline avec elle.
                _client2 = _OAI2(api_key=openai_key, timeout=90.0)
                # Seul `primary_outcome` est de la prose affichée : il suit la langue de
                # l'interface. Tous les autres champs sont des VALEURS D'ÉNUMÉRATION que le
                # code compare et regroupe (study_type → study_design, setting, funding,
                # bias_risk, country en ISO2) : elles restent en anglais.
                system_prompt_meta = (
                    "You are a biomedical librarian. Extract metadata from this article and return ONLY valid JSON:\n"
                    '{"study_type":"RCT|Cohort|Case-control|Cross-sectional|Systematic review|Meta-analysis|Case report|Editorial|Other",'
                    '"sample_size":null,"country":"ISO2 or null","setting":"hospital|prehospital|community|other|null",'
                    '"primary_outcome":"brief description or null","funding":"public|industry|mixed|not reported",'
                    '"bias_risk":"low|moderate|high|unclear","metadata_confidence":0.0-1.0}\n'
                    "Return ONLY the JSON. Every field except \"primary_outcome\" is an "
                    "identifier: keep it in English, exactly from the lists above. Do not "
                    "use the em dash character."
                    + _llm_lang_directive(lang)
                    + " That language applies to \"primary_outcome\" only."
                )
                with engine.connect() as conn:
                    meta_rows = conn.execute(text("""
                        SELECT ld.id, ld.title, ld.abstract, ld.source, ld.year,
                               ld.citation_count, ld.open_access,
                               ld.study_design, ld.sample_size
                        FROM literature_document ld
                        JOIN article_scenarios asn ON asn.document_id = ld.id
                        WHERE asn.scenario_id = :sid
                          AND ld.project_context = 'literev'
                          AND (""" + _enrich_gate() + """)
                          AND (ld.metadata_json IS NULL OR ld.metadata_json = '{}'::jsonb)
                        ORDER BY ld.id
                    """), {"sid": scenario_id}).mappings().fetchall()

                meta_extracted = 0
                meta_errors = 0
                for row in meta_rows:
                    try:
                        def _ask_meta(messages):
                            return _client2.chat.completions.create(
                                model=_model("bulk"),
                                messages=messages,
                                temperature=0.1,
                                max_tokens=300,
                                response_format={"type": "json_object"},
                            )

                        # Coupée au plafond, la réponse est redemandée une fois, plus courte,
                        # au lieu de remonter en erreur d'analyse JSON.
                        metadata = json.loads(json_completion(
                            _ask_meta,
                            [{"role": "system", "content": system_prompt_meta},
                             {"role": "user", "content": f"Title: {row['title']}\n\nAbstract: {(row['abstract'] or '')[:2000]}"}],
                            f"Pipeline metadata {row['id']}"))
                        metadata["metadata_confidence"] = float(metadata.get("metadata_confidence", 0.5))
                        # Colonnes structurées (study_design / sample_size) et quality_score
                        # déterministe, par l'écriture commune au lot /metadata/extract.
                        with engine.begin() as conn:
                            conn.execute(text(METADATA_UPDATE_SQL),
                                         _metadata_update_params(row["id"], metadata, row))
                        meta_extracted += 1
                    except Exception as e:
                        logger.warning(f"Pipeline metadata article {row['id']}: {e}")
                        meta_errors += 1
                    _time.sleep(0.05)
                # Total coverage from DB (includes previously extracted articles)
                with engine.connect() as _meta_stat_conn:
                    _meta_total_in_scenario = _meta_stat_conn.execute(text("""
                        SELECT COUNT(*) FROM article_scenarios WHERE scenario_id = :sid
                    """), {"sid": scenario_id}).scalar() or 0
                    _meta_total_with = _meta_stat_conn.execute(text("""
                        SELECT COUNT(*) FROM literature_document ld
                        JOIN article_scenarios ars ON ars.document_id = ld.id
                        WHERE ars.scenario_id = :sid
                          AND ld.metadata_json IS NOT NULL AND ld.metadata_json != '{}'::jsonb
                    """), {"sid": scenario_id}).scalar() or 0
                update_step("metadata", "done",
                            extracted_this_run=meta_extracted,
                            total_with_metadata=_meta_total_with,
                            total_articles=_meta_total_in_scenario,
                            pct=round(_meta_total_with / _meta_total_in_scenario * 100, 1) if _meta_total_in_scenario > 0 else 0,
                            errors=meta_errors)
            else:
                update_step("metadata", "skipped",
                            reason=_WAIT_REASON if _enrich_waits else "Clé OpenAI non configurée")
        except Exception as e:
            update_step("metadata", "error", error=str(e))

        # ── Étape 8 : Extraction structurée ───────────────────────────────────
        # Les observations (sexe, age, profession, EPI, cas et populations, avec leur
        # citation) ne s'extrayaient QUE sur un bouton. Un relecteur qui posait un seuil
        # voyait donc, indéfiniment, « extraire les 472 articles restants », et tout ce
        # qui lit l'extraction (le brief, la géographie, les estimations groupées, les
        # réponses de l'assistant) portait sur une extraction partielle sans le dire.
        # C'est maintenant une étape comme les autres : TOUS les articles pertinents qui
        # n'en ont pas encore, jamais un échantillon, et le pipeline l'ATTEND pour que
        # les étapes suivantes lisent ce qu'elle vient d'écrire.
        update_step("extraction", "running")
        try:
            from .extraction import extraction_status, start_extraction
            _ex0 = extraction_status(scenario_id)
            if _enrich_waits:
                update_step("extraction", "skipped", reason=_WAIT_REASON)
            elif not os.getenv("OPENAI_API_KEY"):
                update_step("extraction", "skipped", reason="Clé OpenAI non configurée")
            elif not _ex0.get("n_pending"):
                update_step("extraction", "done", relevant=_ex0.get("n_relevant", 0),
                            extracted=_ex0.get("n_extracted", 0), pending=0,
                            observations=_ex0.get("n_observations", 0))
            else:
                # Le nom du scénario sert d'indice de maladie dans le prompt, comme
                # pour le bouton : sans lui l'étape extrairait autrement que lui.
                try:
                    _ex_hint = (_get_user_scenario_or_404(scenario_id) or {}).get("name")
                except Exception:                            # noqa: BLE001
                    _ex_hint = None
                _ex_out = start_extraction(scenario_id, _ex_hint, 0)
                if _ex_out.get("status") in ("no_llm", "paused"):
                    update_step("extraction", "skipped",
                                reason=_ex_out.get("reason") or _ex_out.get("status"))
                else:
                    # Attente bornée : une extraction qui dure (plusieurs milliers
                    # d'articles) ne doit pas retenir le pipeline pour toujours. Passé le
                    # budget, elle CONTINUE en arrière-plan et l'étape le dit, au lieu de
                    # se déclarer finie.
                    _ex_budget = _extraction_wait_budget_s()
                    _ex_t0 = _time.time()
                    _ex = _ex0
                    while True:
                        _time.sleep(3)
                        _ex = extraction_status(scenario_id)
                        if not _ex.get("running"):
                            break
                        _job = _ex.get("job") or {}
                        _tot, _dn = int(_job.get("total") or 0), int(_job.get("done") or 0)
                        update_step("extraction", "running", done=_dn, total=_tot,
                                    failed=int(_job.get("failed") or 0),
                                    pct=round(_dn / _tot * 100, 1) if _tot else 0)
                        if _time.time() - _ex_t0 > _ex_budget:
                            break
                    update_step("extraction",
                                "running" if _ex.get("running") else "done",
                                relevant=_ex.get("n_relevant", 0),
                                extracted=_ex.get("n_extracted", 0),
                                pending=_ex.get("n_pending", 0),
                                observations=_ex.get("n_observations", 0),
                                still_running=bool(_ex.get("running")))
        except Exception as e:
            update_step("extraction", "error", error=str(e))

        # ── Étape 9 : Clustering (UMAP+HDBSCAN avec fallback KMeans) ────────────
        update_step("clustering", "running")
        try:
            # Clustering du pipeline sur le SOUS-ENSEMBLE PERTINENT (≥ seuil sémantique
            # OU inclus manuellement ; jamais exclus) - cohérent avec le clustering à la
            # demande et le knowledge graph.
            _thr = _get_scenario_threshold(scenario_id)
            cl_docs, _cl_total = _clustering_docs(scenario_id, _thr)   # plafonné (CLUSTER_MAX_DOCS)

            if len(cl_docs) >= 5:
                texts = [f"{d['title']} {d['abstract'] or ''}" for d in cl_docs]

                # ── Embeddings → UMAP → HDBSCAN (cœur partagé _cluster_core) ──
                _cc = _cluster_core(cl_docs, texts, openai_key=None,
                                    allow_openai_embeddings=False, tfidf_min_df=1)
                labels = _cc["labels"]
                embedding_2d = _cc["embedding_2d"]
                method_used = _cc["method"]
                feature_names = _cc["feature_names"]
                X_dense = _cc["X_dense"]
                logger.info(f"Pipeline clustering {scenario_id}: {len(cl_docs)} docs, "
                            f"source={_cc['embedding_source']}, method={method_used}")

                n_clusters = len(set(int(l) for l in labels if int(l) != -1))

                # ── Persister en DB ───────────────────────────────────────────
                with engine.begin() as _cl_conn:
                    for _cl_idx, _cl_doc in enumerate(cl_docs):
                        _cl_id = int(labels[_cl_idx])
                        _cl_conn.execute(text("""
                            UPDATE article_scenarios
                            SET cluster_id = :cid, cluster_label = :clabel
                            WHERE scenario_id = :sid AND document_id = :did
                        """), {
                            "cid": _cl_id,
                            "clabel": f"Cluster {_cl_id + 1}" if _cl_id != -1 else "Non-classés",
                            "sid": scenario_id,
                            "did": _cl_doc["id"],
                        })

                # Cache de visualisation : MÊME helper que le calcul en arrière-plan
                # (plus de duplication) → DB durable (+ /tmp pour compat). AVEC les résumés,
                # dans la langue de l'interface qui a lancé le pipeline : la première
                # ouverture de l'onglet ne doit rien attendre (avant : « sans résumés, générés
                # à la première ouverture » - soit 12 appels LLM sous les yeux de l'utilisateur).
                _cl_payload = _build_clusters_payload(scenario_id, cl_docs, _cc, with_summaries=True,
                                                      openai_key=os.getenv("OPENAI_API_KEY"),
                                                      lang=lang or "fr", n_docs_total=_cl_total)
                _persist_clustering_result(scenario_id, _cl_payload)
                update_step("clustering", "done", n_clusters=n_clusters, n_docs=len(cl_docs),
                            n_docs_total=_cl_total, method=method_used)
            else:
                update_step("clustering", "skipped", reason=f"Corpus insuffisant ({len(cl_docs)} articles)")
        except Exception as e:
            update_step("clustering", "error", error=str(e))

        # ── Knowledge graph (cache DB) - visualisation prête ──────────────────
        try:
            update_step("knowledge_graph", "running")
            _precompute_user_kg(scenario_id)
            # Carte des concepts : annotation LLM (une fois par article) des articles
            # pertinents sans concepts, puis cache - l'onglet n'a rien à calculer.
            from .knowledge_graph import _precompute_concept_graph  # lazy: keeps the import list short
            _cg = _precompute_concept_graph(scenario_id, extract=True) or {}
            update_step("knowledge_graph", "done", concepts_articles=_cg.get("n_with_concepts"),
                        concept_nodes=len(_cg.get("nodes") or []))
        except Exception as _e_kg:
            logger.warning(f"Précalcul KG pipeline {scenario_id}: {_e_kg}")
            update_step("knowledge_graph", "error", error=str(_e_kg))

        # ── Evidence Brief (narratif LLM, mis en cache) - prêt sans clic ───────
        try:
            update_step("evidence", "running")
            _generate_evidence_brief_llm(scenario_id, lang=lang or "fr")
            update_step("evidence", "done")
        except Exception as _e_ev:
            logger.warning(f"Evidence brief pipeline {scenario_id}: {_e_ev}")
            update_step("evidence", "error", error=str(_e_ev))

        # ── Variables & Modèle (spec déterministe : outcome, features, algorithme,
        # data_template, paramètres SEIR) - le scénario est « modèle-prêt » d'emblée.
        #
        # La SEULE étape des onze qui ne serve qu'à un scénario prédictif, et la seule
        # qui coûte un appel de modèle pour une question qui n'en attend pas. Une revue
        # de littérature la saute : la décision est prise AVANT la dépense, au lieu
        # d'être découverte après par les écrans qui refusent ensuite de s'ouvrir.
        # La lecture de la nature est DANS le try de l'étape : elle ouvre une connexion,
        # et une connexion peut échouer. Laissée dehors, son exception remontait au
        # gestionnaire du pipeline entier, qui saute alors non seulement cette étape
        # mais toute la fin de la course, si bien qu'un run complet de onze étapes
        # n'écrivait jamais son « done » pour un SELECT d'une ligne.
        try:
            from .scenario_store import CAP_MODEL, scenario_can
            _wants_model = scenario_can(scenario_id, CAP_MODEL)
        except Exception as _e_kind:                                  # noqa: BLE001
            # On ne sait pas : on fait comme avant, c'est-à-dire tout. Une porte ne
            # doit pas retirer une moitié de l'application sur une panne de lecture.
            logger.warning(f"Nature du scénario illisible pour {scenario_id}: {_e_kind}")
            _wants_model = True
        if not _wants_model:
            update_step("variables", "skipped",
                        reason="Revue de littérature : pas de variables prédictives")
            logger.info(f"Variables & modèle ignorés pour {scenario_id} : revue de littérature")
        else:
            try:
                update_step("variables", "running")
                _generate_variables_from_pico(scenario_id, lang=lang or "fr")
                # Projection SEIR par défaut (365 j, 300 tirages) calculée et mise en cache ICI :
                # l'onglet Modèle l'affiche sans simuler sous les yeux de l'utilisateur.
                try:
                    from .seir import _precompute_seir_projection  # lazy: seir is loaded after this module
                    _precompute_seir_projection(scenario_id)
                except Exception as _e_seir:
                    logger.warning(f"Précalcul SEIR {scenario_id}: {_e_seir}")
                update_step("variables", "done")
            except Exception as _e_var:
                logger.warning(f"Génération variables pipeline {scenario_id}: {_e_var}")
                update_step("variables", "error", error=str(_e_var))

        # ── Actions recommandées (carte du tableau de bord) : générées ICI, dans la
        # langue de l'interface - elles l'étaient à la première ouverture de la carte.
        try:
            update_step("actions", "running")
            from .actions import _generate_recommended_actions  # lazy: actions is loaded after this module
            _generate_recommended_actions(scenario_id, lang=lang or "fr")
            update_step("actions", "done")
        except Exception as _e_act:
            logger.warning(f"Actions recommandées pipeline {scenario_id}: {_e_act}")
            update_step("actions", "error", error=str(_e_act))

        # ── Fin du pipeline ───────────────────────────────────────────────────
        _user_scenario_pipeline_jobs[scenario_id]["overall_status"] = "done"
        _user_scenario_pipeline_jobs[scenario_id]["message"] = (
            "Pipeline terminé, sauf le texte intégral, le PICO et les métadonnées, qui attendent "
            "le seuil de pertinence : réglez-le, puis relancez." if _enrich_waits else "Pipeline terminé."
        )
        # Persister pipeline_status = done et mettre à jour article_count (source de vérité = DB)
        try:
            with engine.begin() as _conn:
                _final_count = _conn.execute(text("""
                    SELECT COUNT(DISTINCT ars.document_id) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                """), {"sid": scenario_id}).scalar() or 0
                _conn.execute(text("""
                    UPDATE user_scenarios
                    SET pipeline_status = 'done',
                        pipeline_step = 'done',
                        pipeline_progress = 100,
                        article_count = :cnt,
                        updated_at = NOW()
                    WHERE id = :sid
                """), {"sid": scenario_id, "cnt": _final_count})
            _user_scenario_pipeline_jobs[scenario_id]["message"] = (
                f"{_final_count} articles dans le corpus."
            )
        except Exception as _e:
            logger.warning(f"Pipeline final DB update failed: {_e}")
        # ── Vérification finale : liste = en-tête = PRISMA = étape sémantique ──
        # Pendant le pipeline ces lectures divergent (copie stockée, chiffres figés à
        # la recherche, base en direct) ; à la fin elles doivent coïncider. Exposé dans
        # le statut du pipeline (bannière de la page) et journalisé.
        try:
            _cc = _scenario_counts(scenario_id)
            _user_scenario_pipeline_jobs[scenario_id]["counts"] = _cc
            if _cc["consistent"]:
                logger.info(f"Pipeline complet {scenario_id}: compteurs cohérents - "
                            f"{_cc['corpus_links']} articles (liste, en-tête, PRISMA, étape 2).")
            else:
                logger.warning(f"Pipeline complet {scenario_id}: compteurs INCOHÉRENTS - {_cc['mismatches']}")
        except Exception as _e_cc:
            logger.warning(f"Pipeline {scenario_id}: vérification des compteurs impossible: {_e_cc}")
        logger.info(f"Pipeline complet {scenario_id}: terminé.")

        # Le seuil a-t-il bougé pendant ce passage ? Alors ce qui vient d'être calculé
        # l'a été sur le sous-ensemble d'avant : on repasse UNE fois, sur le nouveau.
        # Le garde-fou contre la boucle est le seuil lui-même : la relance repart du
        # seuil courant, donc le passage suivant ne trouvera plus de différence.
        try:
            _thr_now = scenario_threshold_info(scenario_id)["threshold"]
            if _thr_at_start is not None and _thr_now is not None \
                    and abs(float(_thr_now) - float(_thr_at_start)) > 1e-9:
                from .scenarios import _launch_full_pipeline  # lazy: ordre de chargement
                logger.info(f"Pipeline {scenario_id}: seuil passé de {_thr_at_start} à {_thr_now} "
                            f"pendant le passage, relance sur le nouveau sous-ensemble.")
                _launch_full_pipeline(scenario_id, lang=lang)
        except Exception as _e_relaunch:                     # noqa: BLE001 - jamais bloquant
            logger.warning(f"Pipeline {scenario_id}: relance après changement de seuil: {_e_relaunch}")

    except Exception as e:
        logger.error(f"Pipeline user_scenario {scenario_id} fatal: {e}", exc_info=True)
        _user_scenario_pipeline_jobs[scenario_id]["overall_status"] = "error"
        _user_scenario_pipeline_jobs[scenario_id]["error"] = str(e)
        # Persister l'échec en DB (symétrique du succès) : sinon la carte lit
        # pipeline_status='running' indéfiniment et le scénario est relancé comme
        # « orphelin » au redémarrage.
        try:
            with engine.begin() as _conn:
                _conn.execute(text("""
                    UPDATE user_scenarios
                    SET pipeline_status = 'failed', updated_at = NOW()
                    WHERE id = :sid
                """), {"sid": scenario_id})
        except Exception as _e2:
            logger.warning(f"Pipeline failure DB update {scenario_id}: {_e2}")


# ─── PIPELINE COMPLET AVEC BRIEF LLM ─────────────────────────────────────────

@app.post("/scenarios/{scenario_id}/full-pipeline")
def trigger_full_pipeline_with_brief(scenario_id: str, lang: str | None = Query(None),
                                     _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Déclenche le pipeline complet incluant :
    1. Reranking sémantique
    2. Génération Evidence Brief LLM
    3. Génération Variables & Modèle
    Fonctionne pour GESICA et user_scenarios.

    `lang` : langue de production, comme tous les autres déclencheurs. Sans elle, cet
    endpoint régénérait en français (et, comme il force la régénération du brief,
    REMPLAÇAIT un brief anglais existant par sa version française).
    """
    from .relevance import _run_semantic_rerank_inline  # lazy: relevance is loaded after this module
    from .evidence import _generate_evidence_brief_llm  # lazy: evidence is loaded after this module
    from .variables import _generate_variables_from_pico  # lazy: variables is loaded after this module
    import threading

    if scenario_id.startswith("usr-"):
        row = _get_user_scenario_or_404(scenario_id)
        query = row["query"]
    else:
        meta = _get_db_gesica_scenario_or_404(scenario_id)
        nl = meta.get("nl_queries") or []
        query = nl[0] if nl else _gesica_title(meta)

    _lang = _norm_lang(lang) or "fr"

    def _run():
        logger.info(f"Full pipeline with brief: {scenario_id} (lang={_lang})")
        # 1. Reranking
        _run_semantic_rerank_inline(scenario_id, query)
        # 2. Evidence Brief LLM
        _generate_evidence_brief_llm(scenario_id, force=True, lang=_lang)
        # 3. Variables & Modèle - sautées pour une revue de littérature, comme dans
        #    le pipeline complet : la même question, la même réponse.
        from .scenario_store import CAP_MODEL, scenario_can  # lazy: ordre de chargement
        if scenario_can(scenario_id, CAP_MODEL):
            _generate_variables_from_pico(scenario_id, lang=_lang)
        logger.info(f"Full pipeline with brief done: {scenario_id}")

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "steps": ["rerank", "evidence_brief", "variables"]}
