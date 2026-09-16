"""Corpus build (populate) and the full enrichment pipeline of a scenario.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import Depends
from sqlalchemy import text, bindparam

from .core import POPULATE_FEDERATION_BUDGET, app, engine, logger, require_api_key
from .documents import (
    _coerce_int,
    _compute_quality_score,
    _normalize_doi,
    _strategy_is_degraded,
    _truncate_to_tokens,
    sanitize_db_text,
)
from .scenario_store import _get_scenario_threshold, _get_user_scenario_or_404
from .search import (
    LIVE_MAX_PER_SOURCE,
    _boolean_corpus_ids,
    _boolean_to_arxiv,
    _boolean_to_s2,
    _dedup_scenario_links,
    _facet_ops,
    _facets_intersect,
    _generate_search_strategy,
    _looks_boolean,
    _multi_query_corpus_ids,
    _normalize_sub_queries,
    _parse_boolean_ast,
    _plain_keywords,
    _prisma_identification_figures,
    _search_local_doc_ids,
    _set_scenario_corpus,
    _store_prisma_identification,
    _strip_field_tags,
    _tokenize_boolean,
    _widen_boolean_for_or_facets,
)
from .sources import (
    _NCBI_LAST,
    _NCBI_LOCK,
    _NCBI_MIN_INTERVAL,
    _ingest_doc_direct,
    _ncbi_get,
    _parse_arxiv,
    _parse_biorxiv,
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


def _auto_pipeline_after_search() -> bool:
    """Le pipeline complet d'enrichissement enchaîne-t-il automatiquement après une
    recherche (corpus construit ET scoré) ? Oui par défaut ; AUTO_PIPELINE_AFTER_SEARCH=0
    pour s'en tenir à la recherche (les onglets calculent alors à l'ouverture)."""
    return os.getenv("AUTO_PIPELINE_AFTER_SEARCH", "1").strip().lower() not in ("0", "false", "no", "off")
from .scenarios import _scenario_counts, _user_scenario_pipeline_jobs, _user_scenario_populate_jobs

def _run_user_scenario_populate(
    scenario_id: str,
    query: str,
    filters: dict,
    max_results: int = 500,
    _pipeline_callback=None,
    include_live: bool = True,
    lang: str | None = None,
) -> int:
    """
    Construit le corpus d'un scénario = résultat de la REQUÊTE BOOLÉENNE sur
    (base locale ∪ articles récupérés en direct). Les sources live ne servent qu'à
    ENRICHIR la base ; l'appartenance au corpus est ensuite décidée UNIQUEMENT par
    la correspondance booléenne (_boolean_corpus_ids) — la même que la recherche.
    Plafond : LIVE_MAX_PER_SOURCE articles par source. include_live=False = base
    locale seulement. Retourne le nombre total d'articles ingérés.
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
            "sources": {
                "db_cache": 0, "pubmed": 0, "openalex": 0, "crossref": 0,
                "europepmc": 0, "preprint": 0, "semantic_scholar": 0, "doaj": 0,
                "clinicaltrials": 0, "core": 0, "arxiv": 0, "openaire": 0,
                "biorxiv": 0, "medrxiv": 0
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
    _bool_native_ids: set = set()   # SOURCE-UNION — voir _link_to_scenario
    # Comptabilité PRISMA « identification » : enregistrements ramenés PAR SOURCE (un
    # article renvoyé par trois sources = trois enregistrements) et documents DISTINCTS
    # derrière eux. Leur différence est le nombre de doublons — que la dédup à
    # l'ingestion (index uniques) absorbait jusqu'ici sans laisser de trace.
    _ident_records: dict[str, int] = {}
    _ident_docs: set = set()
    # ③ Santé de la fédération, pour décider si un corpus PEUT rétrécir / se vider.
    #  • _source_errors  : nb de sources ayant échoué (except top-level d'un fetcher) ;
    #  • _fed_incomplete : True si le budget fédération a été dépassé (sources coupées).
    # Un « zéro » n'est fiable — donc on autorise un corpus vide — que si la fédération
    # a réussi (aucune source en erreur ET pas de timeout). Sinon on garde l'ancien
    # corpus, pour ne pas l'effacer sur une panne passagère d'une source.
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

    def _link_to_scenario(doc_id, boolean_native=False, source=None):
        with _corpus_lock:
            if _corpus_frozen[0]:
                return None            # arrivé après l'assemblage : pas dans cette recherche
            return _link_to_scenario_unlocked(doc_id, boolean_native, source)

    def _link_to_scenario_unlocked(doc_id, boolean_native=False, source=None):
        # Comptabilité PRISMA : un enregistrement par source qui a renvoyé l'article,
        # que la ligne soit nouvelle ou déjà connue — c'est justement le recoupement
        # entre sources (et avec la base locale) qui fait le doublon.
        if doc_id is not None and source:
            with _counter_lock:
                _ident_records[source] = _ident_records.get(source, 0) + 1
                _ident_docs.add(doc_id)
        # NE LIE PLUS pendant la fédération : ingérer un article live ne l'ajoute PAS
        # d'office au corpus. L'appartenance est recalculée après ingestion via la
        # correspondance booléenne — sinon le corpus gonflait avec des résultats live
        # (mots-clés) ne correspondant pas à la requête.
        # EXCEPTION — SOURCE-UNION : les sources BOOLÉENNES-NATIVES (PubMed, Europe PMC,
        # préprints EPMC) ont appliqué la VRAIE requête booléenne (MeSH / texte intégral).
        # On mémorise leurs docs pour les INCLURE directement dans le corpus, sans les
        # re-filtrer localement (le re-filtrage titre+résumé perdait leurs correspondances
        # MeSH sans phrase littérale dans le résumé — d'où « 109 PubMed → 6 »). Les
        # sources par MOTS-CLÉS, elles, restent re-filtrées (leur tri est lâche).
        if boolean_native and doc_id is not None:
            _bool_native_ids.add(doc_id)
            # Lien INCRÉMENTAL : un doc booléen-natif appartient au corpus par source-union
            # (confirmé tel quel à l'assemblage final). On le lie DÈS l'ingestion pour que
            # le compteur du corpus grandisse EN DIRECT pendant la recherche (172 → … →
            # total), au lieu de sauter d'un coup à la fin. Score NULL ici → renseigné au
            # scoring. best-effort : l'assemblage final refait l'union de toute façon.
            try:
                with engine.begin() as _lc:
                    _lc.execute(text(
                        "INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                        "VALUES (:d, :s, NULL) ON CONFLICT (document_id, scenario_id) DO NOTHING"),
                        {"d": doc_id, "s": scenario_id})
            except Exception:
                pass
        return None

    def _inc(source_name, count=1, err=0):
        with _counter_lock:
            _ingested_total[0] += count
            _errors_total[0] += err
            if _pipeline_callback is None:
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

    # ── Étape 0 : Linking depuis la base locale (séquentiel, rapide) ─────────
    # Le CORPUS est défini par une correspondance LEXICALE (requête booléenne),
    # indépendante du seuil sémantique : base locale ∪ nouvelles références live.
    # Le seuil sémantique n'intervient QUE dans la page scénario pour sélectionner
    # le sous-ensemble pertinent (_get_above_threshold_articles).
    local_linked = 0
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
                _gen = _generate_search_strategy(query)
                _boolean = _gen.get("general") or query
                _pubmed_q = _gen.get("pubmed") or _boolean
                if not _strategy_is_degraded(_gen, query):
                    with engine.begin() as _sc2:
                        _sc2.execute(text("UPDATE user_scenarios SET search_strategy = CAST(:s AS jsonb) WHERE id = :id"),
                                     {"s": json.dumps(_gen), "id": scenario_id})
        except Exception as _be:
            logger.warning(f"Populate {scenario_id} boolean strategy: {_be}")
        # Facettes UNIES (OU) d'une recherche multi-facettes : fédérées AUSSI, en
        # élargissant les requêtes live (« principal OR facette »). Les facettes ET
        # restent re-matchées localement (sous-ensemble du principal).
        if _sub_queries:
            try:
                _boolean, _pubmed_q, _n_or = _widen_boolean_for_or_facets(_boolean, _pubmed_q, _sub_queries, _combinator)
                if _n_or:
                    logger.info(f"Populate {scenario_id}: fédération élargie à {_n_or} facette(s) OU — {_boolean[:160]!r}")
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
        _portable_bool = _strip_field_tags(_boolean).strip()
        _bool_is_real = bool(_portable_bool) and _looks_boolean(_portable_bool)
        _send_bool = _bool_is_real and len(_portable_bool) <= 1200
        _bool_query = _portable_bool if _send_bool else _plain_q          # OpenAlex/DOAJ/CORE/CT
        _arxiv_q, _arxiv_native = f"all:{_plain_q}", False                # arXiv : syntaxe dédiée
        if _bool_is_real:
            try:
                _ax = _boolean_to_arxiv(_parse_boolean_ast(_tokenize_boolean(_portable_bool)))
                if _ax and len(_ax) <= 1200:
                    _arxiv_q, _arxiv_native = _ax, True
            except Exception:
                pass
        # PubMed RECALL : la requête MeSH générée par le LLM (_pubmed_q) est parfois
        # BEAUCOUP plus étroite que le booléen général — p. ex. 35 résultats contre 306
        # pour le même booléen collé sur le site PubMed. On interroge donc PubMed sur
        # l'UNION « (MeSH) OR (booléen portable) » : on garde les correspondances MeSH
        # ET les correspondances de phrase (all-fields). Ne peut qu'AJOUTER des résultats.
        if (_bool_is_real and _portable_bool and _pubmed_q
                and _portable_bool not in _pubmed_q
                and len(_pubmed_q) + len(_portable_bool) <= 1900):
            _pubmed_q = f"({_pubmed_q}) OR ({_portable_bool})"
        _local_ids = (_multi_query_corpus_ids(_sub_queries, _combinator, filters)
                      if _sub_queries
                      else _search_local_doc_ids(_boolean, "boolean", filters, limit=100_000))

        if _local_ids:
            with engine.begin() as _lc2:
                # RESET du corpus à la correspondance booléenne. Sans cela,
                # l'accumulation ON CONFLICT DO NOTHING ne retire jamais les liens
                # devenus obsolètes (ex. un ancien match lexical OR-de-tous-les-mots
                # qui avait gonflé le corpus à des dizaines de milliers d'articles).
                # Le corpus = EXACTEMENT le résultat de la requête booléenne (base
                # locale) ∪ les nouvelles références live ajoutées plus bas.
                _lc2.execute(
                    text("DELETE FROM article_scenarios WHERE scenario_id = :sid "
                         "AND document_id NOT IN :ids").bindparams(
                             bindparam("ids", expanding=True)),
                    {"sid": scenario_id, "ids": list(_local_ids)},
                )
                # Insertion en masse (un seul aller-retour) : le corpus local doit
                # être lié quasi instantanément, sans une requête par document.
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

    # ── Étape 1 : Interrogation parallèle des 13 sources externes ────────────
    # (13 sources nommées = 12 fetchers ; bioRxiv+medRxiv partagent un fetcher, et
    #  Europe PMC sert 2 facettes — europepmc + préprints. Cf. source_funcs plus bas.)

    def _fetch_pubmed():
        count = 0
        try:
            # Throttle partagé eutils (verrou global + clé API + retry 429) : sinon
            # PubMed se faisait évincer par PROSPERO/Cochrane (mêmes serveurs eutils,
            # 3 req/s sans clé) → esearch 429 → total_found=0 → 0 article ingéré.
            r = _ncbi_get(
                f"{ENTREZ_BASE}/esearch.fcgi",
                # sort=pub_date → l'ensemble historique est trié du plus récent au
                # plus ancien ; efetch récupère donc d'abord les articles récents.
                {"db": "pubmed", "term": _pubmed_q, "retmax": 0, "sort": "pub_date",
                 "retmode": "json", "usehistory": "y", "email": EMAIL},
                timeout=30,
            )
            r.raise_for_status()
            search_result = r.json()["esearchresult"]
            total_found = int(search_result.get("count", 0))
            web_env = search_result.get("webenv", "")
            query_key = search_result.get("querykey", "1")
            if _pipeline_callback:
                _pipeline_callback("pubmed_found", total_found)
            effective_max = min(max_results, total_found)
            n_batches = math.ceil(effective_max / BATCH_SIZE) if effective_max > 0 else 0
            for batch_idx in range(n_batches):
                # Budget fédération dépassé → on ARRÊTE la pagination PubMed (comme
                # toutes les autres sources). Sans ce garde, PubMed continuait à
                # ingérer APRÈS la reconstruction finale du corpus (liens
                # boolean_native écrits trop tard) → membres non scorés + divergence
                # de article_count.
                if _time.time() >= _fed_deadline[0]:
                    break
                retstart = batch_idx * BATCH_SIZE
                retmax_batch = min(BATCH_SIZE, effective_max - retstart)
                if retmax_batch <= 0 or retstart >= total_found:
                    break
                try:
                    _ef_data = {"db": "pubmed", "WebEnv": web_env, "query_key": query_key,
                                "retstart": retstart, "retmax": retmax_batch,
                                "rettype": "xml", "retmode": "xml", "email": EMAIL}
                    _ef_key = os.getenv("NCBI_API_KEY")
                    if _ef_key:
                        _ef_data["api_key"] = _ef_key
                    # Espacer le DÉMARRAGE de la requête vis-à-vis des autres appels
                    # eutils (verrou partagé), sans tenir le verrou pendant le POST
                    # (lent) afin de ne pas sérialiser PROSPERO/Cochrane.
                    with _NCBI_LOCK:
                        _ef_wait = (0.11 if _ef_key else _NCBI_MIN_INTERVAL) - (_time.time() - _NCBI_LAST[0])
                        if _ef_wait > 0:
                            _time.sleep(_ef_wait)
                        _NCBI_LAST[0] = _time.time()
                    r2 = _requests.post(f"{ENTREZ_BASE}/efetch.fcgi", data=_ef_data, timeout=90)
                    r2.raise_for_status()
                except Exception as _e_fetch:
                    logger.warning(f"PubMed efetch batch {batch_idx}: {_e_fetch}")
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
                    year_text = (article_elem.findtext(".//PubDate/Year")
                                 or article_elem.findtext(".//ArticleDate/Year") or "")
                    year = int(year_text[:4]) if year_text[:4].isdigit() else None
                    authors_list = []
                    for author in article_elem.findall(".//AuthorList/Author"):
                        last = author.findtext("LastName") or ""
                        first = author.findtext("ForeName") or ""
                        if last:
                            authors_list.append(f"{last} {first}".strip())
                    authors = "; ".join(authors_list[:6]) if authors_list else None
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
                            # external_id « pmid:<id> » — MÊME format que la recherche
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
            _source_errors[0] += 1
        return ("pubmed", count)

    def _fetch_openalex():
        count = 0
        try:
            _oa_page = 1
            _oa_fetched = 0
            _oa_limit = min(max_results, max_results)
            while _oa_fetched < _oa_limit:
                if _time.time() >= _fed_deadline[0]:
                    break  # budget fédération dépassé — on arrête de paginer
                _oa_batch = min(200, _oa_limit - _oa_fetched)
                oa_resp = _requests.get(
                    "https://api.openalex.org/works",
                    # sort=relevance_score:desc → quand on plafonne à max_results, on garde
                    # les 2000 LES PLUS PERTINENTS (BM25 OpenAlex) et non les plus récents.
                    # OpenAlex ordonne par pertinence par défaut sous `search` ; on l'explicite.
                    params={"search": _bool_query, "per_page": _oa_batch, "page": _oa_page,
                            "sort": "relevance_score:desc", "mailto": "literev@gesica.ch"},
                    timeout=20,
                )
                oa_resp.raise_for_status()
                _oa_results = oa_resp.json().get("results", [])
                if not _oa_results:
                    break
                for work in _oa_results:
                    ext_id = work.get("id", "").split("/")[-1]
                    title = work.get("title") or ""
                    if not ext_id or not title:
                        continue
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
                        _link_to_scenario(doc_id, boolean_native=_send_bool, source="openalex")   # source-union si booléen
                        if _new:
                            count += 1
                            _inc("openalex")
                    except Exception:
                        _inc("openalex", 0, 1)
                _oa_fetched += len(_oa_results)
                if len(_oa_results) < _oa_batch or _oa_fetched >= _oa_limit:
                    break
                _oa_page += 1
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"OpenAlex populate {scenario_id}: {_e}")
            _source_errors[0] += 1
        return ("openalex", count)

    def _fetch_crossref():
        count = 0
        try:
            _cr_offset = 0
            _cr_fetched = 0
            _cr_limit = min(max_results, max_results)
            _cr_rows = min(1000, _cr_limit)   # max Crossref : 10× moins d'allers-retours
            while _cr_fetched < _cr_limit:
                if _time.time() >= _fed_deadline[0]:
                    break  # budget fédération dépassé — on arrête de paginer
                cr_resp = _requests.get(
                    "https://api.crossref.org/works",
                    # NB : PAS de sort=published desc ici (contrairement aux autres
                    # sources). Les dates Crossref sont peu fiables : un tri par date
                    # remonte des enregistrements à dates erronées (ex. « 2121 ») en
                    # tête. On garde donc le tri par pertinence (défaut), qui place les
                    # articles les plus pertinents — pas les plus faussement récents.
                    params={"query": _plain_q, "rows": _cr_rows, "offset": _cr_offset,
                            "mailto": "literev@gesica.ch"},
                    timeout=20,
                )
                cr_resp.raise_for_status()
                _cr_items = cr_resp.json().get("message", {}).get("items", [])
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
                    year = None
                    created = item.get("created", {}).get("date-parts", [])
                    if created and created[0]:
                        year = created[0][0]
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
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"Crossref populate {scenario_id}: {_e}")
            _source_errors[0] += 1
        return ("crossref", count)

    def _fetch_europepmc():
        count = 0
        try:
            _ep_cursor_mark = "*"
            _ep_fetched = 0
            _ep_limit = min(max_results, max_results)
            _ep_page_size = 1000   # max Europe PMC : moins d'allers-retours → fédération plus rapide
            while _ep_fetched < _ep_limit:
                if _time.time() >= _fed_deadline[0]:
                    break  # budget fédération dépassé — on arrête de paginer
                ep_resp = _requests.get(
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    # Pas de tri par date : on laisse le tri par PERTINENCE (défaut Europe PMC)
                    # → au plafond de 2000, on garde les plus pertinents et non les plus récents.
                    params={"query": _boolean, "format": "json", "pageSize": _ep_page_size,
                            "resultType": "core",
                            "cursorMark": _ep_cursor_mark},
                    timeout=20,
                )
                ep_resp.raise_for_status()
                _ep_data = ep_resp.json()
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
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"EuropePMC populate {scenario_id}: {_e}")
            _source_errors[0] += 1
        return ("europepmc", count)

    def _fetch_preprints():
        # Préprints (bioRxiv, medRxiv, Research Square, …) via Europe PMC (SRC:PPR) :
        # recherche par mots-clés RÉELLE. L'API biorxiv ne fait que dates/DOI, d'où
        # l'ancien scan des 90 derniers jours filtré côté client (~0 résultat).
        count = 0
        try:
            _pp_cursor = "*"
            _pp_fetched = 0
            _pp_query = f"({_boolean}) AND (SRC:PPR)"
            while _pp_fetched < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break  # budget fédération dépassé
                _pp_resp = _requests.get(
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    params={"query": _pp_query, "format": "json", "pageSize": 100,
                            "resultType": "core", "sort": "P_PDATE_D desc",
                            "cursorMark": _pp_cursor},
                    timeout=20,
                )
                _pp_resp.raise_for_status()
                _pp_data = _pp_resp.json()
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
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"Préprints (Europe PMC) populate {scenario_id}: {_e}")
            _source_errors[0] += 1
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
        _s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        _hdrs = {"x-api-key": _s2_key} if _s2_key else {}
        _fields = "title,abstract,year,externalIds,url"
        # Vrai booléen → endpoint BULK (opérateurs AND/OR + jusqu'à 10M via token) +
        # source-union. Sinon → endpoint classique (relevance, offset≤1000) + mots-clés.
        _s2_bool = None
        if _bool_is_real:
            try:
                _s2_bool = _boolean_to_s2(_parse_boolean_ast(_tokenize_boolean(_portable_bool)))
            except Exception:
                _s2_bool = None
        try:
            if _s2_bool:
                _tok, _fetched = None, 0
                while _fetched < max_results:
                    if _time.time() >= _fed_deadline[0]:
                        break
                    _p = {"query": _s2_bool, "fields": _fields}
                    if _tok:
                        _p["token"] = _tok
                    _r = _requests.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",
                                       params=_p, headers=_hdrs, timeout=25)
                    if _r.status_code == 429:
                        _time.sleep(2)
                        continue
                    _r.raise_for_status()
                    _payload = _r.json()
                    _n = len(_payload.get("data") or [])
                    count += _ingest_parsed("semantic_scholar", _parse_semantic_scholar(_payload),
                                            boolean_native=True)
                    _fetched += _n
                    _tok = _payload.get("token")
                    if not _tok or _n == 0:
                        break
                    _time.sleep(0.3)
            else:
                _off, _cap = 0, min(max_results, 1000)   # search classique : offset+limit ≤ 1000
                while _off < _cap:
                    if _time.time() >= _fed_deadline[0]:
                        break
                    _bulk = min(100, _cap - _off)
                    _r = _requests.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        params={"query": _plain_q, "offset": _off, "limit": _bulk, "fields": _fields},
                        headers=_hdrs, timeout=20,
                    )
                    if _r.status_code == 429:
                        _time.sleep(2)
                        continue
                    _r.raise_for_status()
                    _payload = _r.json()
                    _got = len(_payload.get("data") or [])
                    count += _ingest_parsed("semantic_scholar", _parse_semantic_scholar(_payload))
                    if _got < _bulk:
                        break
                    _off += _bulk
                    _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"Semantic Scholar populate {scenario_id}: {_e}")
            _source_errors[0] += 1
        return ("semantic_scholar", count)

    def _fetch_doaj():
        count = 0
        try:
            import urllib.parse as _ulib
            _page, _fetched = 1, 0
            while _fetched < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break
                _r = _requests.get(
                    f"https://doaj.org/api/search/articles/{_ulib.quote(_bool_query, safe='')}",
                    params={"pageSize": 100, "page": _page}, timeout=20,
                )
                _r.raise_for_status()
                _payload = _r.json()
                _n = len(_payload.get("results") or [])
                if _n == 0:
                    break
                count += _ingest_parsed("doaj", _parse_doaj(_payload), boolean_native=_send_bool)
                _fetched += _n
                if _n < 100:
                    break
                _page += 1
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"DOAJ populate {scenario_id}: {_e}")
            _source_errors[0] += 1
        return ("doaj", count)

    def _fetch_clinicaltrials():
        count = 0
        try:
            _ct_token, _ct_fetched = None, 0
            while _ct_fetched < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break
                _params = {"query.term": _bool_query, "pageSize": 100, "format": "json"}
                if _ct_token:
                    _params["pageToken"] = _ct_token
                _r = _requests.get("https://clinicaltrials.gov/api/v2/studies", params=_params, timeout=20)
                _r.raise_for_status()
                _payload = _r.json()
                _n = len(_payload.get("studies") or [])
                if _n == 0:
                    break
                count += _ingest_parsed("clinicaltrials", _parse_clinicaltrials(_payload), boolean_native=_send_bool)
                _ct_fetched += _n
                _ct_token = _payload.get("nextPageToken")
                if not _ct_token:
                    break
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"ClinicalTrials.gov populate {scenario_id}: {_e}")
            _source_errors[0] += 1
        return ("clinicaltrials", count)

    def _fetch_core():
        # CORE exige une clé API (gratuite). Sans clé → source ignorée proprement
        # (comme NCBI_API_KEY : optionnelle, le déploiement reste vert).
        _core_key = os.getenv("CORE_API_KEY")
        if not _core_key:
            logger.info(f"CORE populate {scenario_id}: CORE_API_KEY absent — source ignorée.")
            return ("core", 0)
        count = 0
        try:
            _core_offset, _core_fetched = 0, 0
            while _core_fetched < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break
                _bulk = min(100, max_results - _core_fetched)
                _r = _requests.post(
                    "https://api.core.ac.uk/v3/search/works",
                    headers={"Authorization": f"Bearer {_core_key}"},
                    json={"q": _bool_query, "limit": _bulk, "offset": _core_offset},
                    timeout=25,
                )
                if _r.status_code == 429:
                    _time.sleep(3)
                    continue
                _r.raise_for_status()
                _payload = _r.json()
                _n = len(_payload.get("results") or [])
                if _n == 0:
                    break
                count += _ingest_parsed("core", _parse_core(_payload), boolean_native=_send_bool)
                _core_fetched += _n
                _core_offset += _bulk
                if _n < _bulk:
                    break
                _time.sleep(0.3)
        except Exception as _e:
            logger.warning(f"CORE populate {scenario_id}: {_e}")
            _source_errors[0] += 1
        return ("core", count)

    def _fetch_arxiv():
        count = 0
        try:
            _ax_start = 0
            while _ax_start < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break
                _bulk = min(100, max_results - _ax_start)
                _r = _requests.get(
                    "http://export.arxiv.org/api/query",
                    params={"search_query": _arxiv_q, "start": _ax_start, "max_results": _bulk},
                    timeout=25,
                )
                _r.raise_for_status()
                _docs = _parse_arxiv(_r.text)
                if not _docs:
                    break
                count += _ingest_parsed("arxiv", _docs, boolean_native=_arxiv_native)
                if len(_docs) < _bulk:
                    break
                _ax_start += _bulk
                _time.sleep(3)      # arXiv demande ≥3 s entre requêtes
        except Exception as _e:
            logger.warning(f"arXiv populate {scenario_id}: {_e}")
            _source_errors[0] += 1
        return ("arxiv", count)

    def _fetch_openaire():
        count = 0
        # Migration vers l'API Graph v2 : l'ancien /search/publications a été RETIRÉ le
        # 2026-05-31. `search=` accepte AND/OR/NOT + parenthèses + guillemets → source-union
        # quand on a un vrai booléen (sinon mots-clés). Pagination par CURSEUR.
        _oa_q = _portable_bool if _bool_is_real else _plain_q
        _oa_native = bool(_bool_is_real)
        try:
            _cursor, _fetched = "*", 0
            while _fetched < max_results:
                if _time.time() >= _fed_deadline[0]:
                    break
                _r = _requests.get(
                    "https://api.openaire.eu/graph/v2/researchProducts",
                    params={"search": _oa_q, "pageSize": 50, "cursor": _cursor},
                    headers={"Accept": "application/json"}, timeout=25,
                )
                _r.raise_for_status()
                _payload = _r.json()
                _docs = _parse_openaire_graph(_payload)
                count += _ingest_parsed("openaire", _docs, boolean_native=_oa_native)
                _fetched += len(_docs)
                _next = (_payload.get("header") or {}).get("nextCursor")
                if not _docs or not _next or _next == _cursor:
                    break
                _cursor = _next
                _time.sleep(0.5)
        except Exception as _e:
            logger.warning(f"OpenAIRE (Graph API v2) populate {scenario_id}: {_e}")
            _source_errors[0] += 1
        return ("openaire", count)

    def _fetch_biorxiv_medrxiv():
        # L'API bioRxiv/medRxiv n'offre PAS de recherche par mots-clés. On scanne une
        # fenêtre RÉCENTE (les ~45 derniers jours) DEPUIS le curseur 0 vers aujourd'hui
        # — de sorte que la couverture porte réellement sur les préprints récents (le
        # bug précédent scannait le DÉBUT d'une fenêtre de 18 mois → les plus VIEUX,
        # jamais pertinents). Puis filtrage lexical (_parse_biorxiv). Chaque serveur
        # ingère sous sa propre source ("biorxiv" / "medrxiv"), distinctes des
        # "Preprints" (facette Europe PMC).
        count = 0
        try:
            from datetime import date as _date, timedelta as _td
            _terms = [re.sub(r"[^a-z0-9]", "", w) for w in (_plain_q or "").lower().split()]
            _terms = [t for t in _terms if len(t) >= 4]
            try:
                _to = _date.today()
                _win = f"{(_to - _td(days=45)).isoformat()}/{_to.isoformat()}"
            except Exception:
                _win = "2026-05-01/2026-12-31"
            for _server in ("biorxiv", "medrxiv"):
                _cursor, _pages = 0, 0
                while _pages < 30:          # borne dure ; les ~45 j récents tiennent dedans
                    if _time.time() >= _fed_deadline[0]:
                        break
                    _r = _requests.get(
                        f"https://api.biorxiv.org/details/{_server}/{_win}/{_cursor}/json", timeout=20)
                    _r.raise_for_status()
                    _payload = _r.json()
                    _coll = _payload.get("collection") or []
                    if not _coll:
                        break
                    count += _ingest_parsed(_server, _parse_biorxiv(_payload, _terms, _server))
                    _pages += 1
                    _total = int((_payload.get("messages") or [{}])[0].get("total", 0) or 0)
                    _cursor += len(_coll)
                    if _cursor >= _total or len(_coll) < 100:
                        break
                    _time.sleep(0.4)
        except Exception as _e:
            logger.warning(f"bioRxiv/medRxiv populate {scenario_id}: {_e}")
            _source_errors[0] += 1
        return ("biorxiv_medrxiv", count)

    # Lancer toutes les sources en parallèle
    source_funcs = [
        _fetch_pubmed, _fetch_openalex, _fetch_crossref,
        _fetch_europepmc, _fetch_preprints,
        _fetch_semantic_scholar, _fetch_doaj, _fetch_clinicaltrials, _fetch_core,
        _fetch_arxiv, _fetch_openaire, _fetch_biorxiv_medrxiv,
    ]
    t_start = _time.time()
    if include_live:
        _set_phase("federation")
        # Garde-temps : passé ce délai, les boucles de pagination des sources lentes
        # s'arrêtent (cf. _fed_deadline) et on poursuit avec le corpus partiel.
        _fed_deadline[0] = t_start + POPULATE_FEDERATION_BUDGET
        # IMPORTANT — on N'UTILISE PAS `with ThreadPoolExecutor(...)` : sa sortie
        # appelle shutdown(wait=True), qui attend TOUTES les sources (jusqu'à ~5 min
        # quand OpenAlex/Crossref paginent vers 2000), annulant de fait le budget.
        # On gère l'executor manuellement et on l'arrête SANS attendre.
        executor = ThreadPoolExecutor(max_workers=12)
        try:
            futures = {executor.submit(fn): fn.__name__ for fn in source_funcs}
            try:
                # Budget global : ne pas attendre indéfiniment une source lente.
                for future in as_completed(futures, timeout=POPULATE_FEDERATION_BUDGET):
                    try:
                        src_name, src_count = future.result()
                        logger.info(f"Populate {scenario_id} [{src_name}]: {src_count} articles ingérés")
                    except Exception as _fe:
                        logger.warning(f"Populate {scenario_id} source future error: {_fe}")
            # CRITIQUE — sur Python <3.11, as_completed lève
            # concurrent.futures.TimeoutError (≠ TimeoutError natif). Sans
            # _FuturesTimeout dans le except, l'exception remontait, le bloc
            # plantait, et la reconstruction du corpus + le scoring + le passage à
            # « done » étaient SAUTÉS → corpus figé sur la base locale, statut
            # bloqué sur « running », résultats live perdus.
            except (TimeoutError, _FuturesTimeout):
                _fed_incomplete[0] = True   # fetch partiel → corpus non autorisé à rétrécir
                _done = sum(1 for _f in futures if _f.done())
                logger.warning(
                    f"Populate {scenario_id}: budget fédération {POPULATE_FEDERATION_BUDGET:.0f}s dépassé — "
                    f"{_done}/{len(futures)} sources terminées ; poursuite avec le corpus partiel "
                    f"(les sources lentes continuent en arrière-plan)."
                )
        finally:
            # wait=False : ne PAS bloquer sur les sources lentes. cancel_futures=True
            # annule celles qui n'ont pas démarré ; celles en cours s'arrêteront au
            # prochain tour de pagination grâce à _fed_deadline.
            executor.shutdown(wait=False, cancel_futures=True)
        t_elapsed = _time.time() - t_start
        logger.info(f"Populate {scenario_id}: fédération terminée en {t_elapsed:.1f}s")
    else:
        logger.info(f"Populate {scenario_id}: include_live=False — base locale uniquement")

    ingested = _ingested_total[0]
    errors = _errors_total[0]
    total_found = ingested  # Approximation — PubMed callback met à jour séparément

    # ── Corpus = correspondance BOOLÉENNE (ou multi-sous-requêtes) sur base enrichie ─
    # Après ingestion des articles live, on recalcule l'appartenance au corpus via
    # EXACTEMENT le même helper que la recherche : _boolean_corpus_ids en mono-requête,
    # _multi_query_corpus_ids (union/intersection) en multi-sous-requêtes. Le corpus
    # devient donc strictement « résultat de la requête sur base locale ∪ live ».
    # allow_empty=True en multi : une intersection légitimement vide DOIT vider le corpus.
    try:
        # Gel : à partir d'ici, ce qui arrive encore des sources ne fait plus partie de
        # cette recherche (cf. _corpus_frozen). Le corpus et ses chiffres sont figés ensemble.
        with _corpus_lock:
            _corpus_frozen[0] = True
        # Appartenance = re-match booléen LOCAL (base locale ∪ live) pour les sources par
        # mots-clés + la base existante…
        if _sub_queries:
            _final_ids = _multi_query_corpus_ids(_sub_queries, _combinator, filters)
        else:
            _final_ids = _boolean_corpus_ids(_boolean, filters)
        # … ∪ SOURCE-UNION : les docs des sources booléennes-natives (PubMed, Europe PMC,
        # préprints EPMC) qui ont appliqué la VRAIE requête booléenne sont inclus DIRECTEMENT,
        # sans re-filtrage local (qui supprimait leurs correspondances MeSH/texte-intégral —
        # « 109 PubMed → 6 »). La règle « pas de résumé → exclu » s'applique quand même après.
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
        # ③ Autorise un corpus vide/réduit :
        #  • multi-requêtes : une intersection légitimement vide DOIT vider (inchangé) ;
        #  • base locale seule (pas de live) : le match booléen local est déterministe ;
        #  • mono-requête + live : SEULEMENT si la fédération a réussi (fetch_ok) — sinon
        #    un « zéro » peut venir d'une panne passagère → on garde l'ancien corpus.
        _fetch_ok = (not _fed_incomplete[0]) and _source_errors[0] == 0
        _allow_empty = bool(_sub_queries) or (not include_live) or _fetch_ok
        _n_corpus = _set_scenario_corpus(scenario_id, _final_ids, allow_empty=_allow_empty)
        # « assemblé, AVANT nettoyage » : les liens vers les documents sans résumé et
        # les lignes doublons sont retirés juste après — le corpus retenu (article_count,
        # « passés au screening » du PRISMA) est journalisé plus bas avec les chiffres
        # PRISMA. Étiqueté « final » auparavant, ce nombre était lu comme le corpus.
        logger.info(f"Populate {scenario_id}: corpus assemblé (avant nettoyage) = {_n_corpus} docs "
                    f"(re-match local ∪ {_n_native} docs booléens-natifs PubMed/EPMC/préprints ; "
                    f"{'multi ' + '/'.join(_facet_ops(_sub_queries, _combinator)) if _sub_queries else 'mono'})")
    except Exception as _e_corpus:
        logger.warning(f"Rebuild corpus {scenario_id}: {_e_corpus}")

    try:
        # ── Règle qualité : articles SANS abstract retirés du corpus ─────────
        try:
            with engine.begin() as conn:
                _removed = conn.execute(text("""
                    DELETE FROM article_scenarios a
                    USING literature_document d
                    WHERE a.document_id = d.id AND a.scenario_id = :sid
                      AND (d.abstract IS NULL OR length(TRIM(d.abstract)) < 30)
                """), {"sid": scenario_id}).rowcount
            if _removed:
                logger.info(f"Populate {scenario_id}: {_removed} articles sans abstract retirés.")
        except Exception as _e_noabs:
            logger.warning(f"Suppression articles sans abstract {scenario_id}: {_e_noabs}")

        # ── Dédup INTRA-scénario : un seul lien par article distinct ─────────
        # APRÈS le retrait des articles sans abstract, pour ne jamais garder par
        # erreur une copie sans résumé au détriment d'une copie complète du même
        # article. Fige le compte de façon déterministe (cf. _dedup_scenario_links).
        _n_dup_rows = _dedup_scenario_links(scenario_id)

        # ── Mettre à jour article_count (avant rerank) ──────────────────────
        # NB : on ne marque PAS encore populate_status='done' ici — le scoring
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
                _ids_snapshot = list(_ident_docs)
            # Ventilation des documents identifiés mais ABSENTS du corpus : sans résumé
            # (règle qualité, appliquée avant la dédup — donc les lignes fusionnées par
            # la dédup ont toutes un résumé et sont à soustraire de l'autre poche), ou
            # avec résumé mais non liés (source par mots-clés hors requête booléenne).
            _no_abs, _not_linked = 0, 0
            if _ids_snapshot:
                with engine.connect() as _bc:
                    _br = _bc.execute(text("""
                        SELECT COUNT(*) FILTER (WHERE d.abstract IS NULL OR length(TRIM(d.abstract)) < 30) AS no_abstract,
                               COUNT(*) FILTER (WHERE NOT (d.abstract IS NULL OR length(TRIM(d.abstract)) < 30)
                                                  AND NOT EXISTS (SELECT 1 FROM article_scenarios a
                                                                   WHERE a.scenario_id = :sid AND a.document_id = d.id)) AS not_linked
                        FROM literature_document d
                        WHERE d.id = ANY(CAST(:ids AS bigint[]))
                    """), {"sid": scenario_id, "ids": _ids_snapshot}).mappings().first()
                _no_abs = int(_br["no_abstract"] or 0)
                _not_linked = int(_br["not_linked"] or 0)
            _figures = _prisma_identification_figures(
                _recs_snapshot, len(_ids_snapshot), _n_dup_rows or 0, int(_corpus_now),
                method="populate", federation_incomplete=bool(_fed_incomplete[0]),
                removed_no_abstract=_no_abs,
                removed_not_matching=max(0, _not_linked - int(_n_dup_rows or 0)))
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
                        f"{_figures['records_screened']} au screening.")
        except Exception as _e_pi:
            logger.warning(f"Populate {scenario_id}: chiffres PRISMA non stockés: {_e_pi}")

        # ── Scores sémantiques (cosinus) — SANS suppression ─────────────────
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

        # ── Honnêteté de l'état : 'done' SEULEMENT si le scoring a réellement
        # produit des scores. _run_semantic_rerank_inline avale ses erreurs et
        # renvoie 0 (ex. OpenAI indisponible) ; on détecte ce cas et on marque
        # 'error' plutôt que 'done' pour ne pas afficher un état prêt trompeur.
        try:
            with engine.begin() as conn:
                _scorable = conn.execute(text("""
                    SELECT COUNT(DISTINCT ars.document_id) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                      AND d.abstract IS NOT NULL AND length(TRIM(d.abstract)) >= 30
                """), {"sid": scenario_id}).scalar() or 0
                _ok = (not _scoring_failed) and (_n_scored > 0 or _scorable == 0)
                _auto_ok[0] = bool(_ok)
                conn.execute(text("""
                    UPDATE user_scenarios SET populate_status = :st, updated_at = NOW() WHERE id = :sid
                """), {"st": "done" if _ok else "error", "sid": scenario_id})
                if not _ok:
                    logger.warning(f"Populate {scenario_id}: scoring n'a produit aucun score "
                                   f"({_scorable} articles scorables) → populate_status='error'.")
        except Exception as _e_st:
            logger.warning(f"populate_status update {scenario_id}: {_e_st}")

        # ── Le corpus est scoré (cosinus) → on le publie MAINTENANT ──────────
        # Le cross-encoder Cohere (plus lent) et les visualisations tournent
        # ENSUITE en arrière-plan : l'utilisateur voit les résultats ordonnés par
        # cosinus immédiatement, puis la liste se réordonne quand le rerank arrive
        # (rerank_status). Plus d'attente synchrone sur l'API Cohere.
        _cohere_enabled = bool(os.getenv("COHERE_API_KEY"))
        if _pipeline_callback is None:
            _sources_final = _user_scenario_populate_jobs.get(scenario_id, {}).get("sources", {})
            _src_parts = [f"{src}: {cnt}" for src, cnt in _sources_final.items() if cnt > 0]
            _src_summary = " | ".join(_src_parts) if _src_parts else "aucune source"
            _user_scenario_populate_jobs[scenario_id] = {
                "status": "done",
                "phase": "done",
                "rerank_status": "running" if _cohere_enabled else "skipped",
                "ingested": ingested,
                # `ingested` = base locale + NOUVEAUX documents : un compteur de travail,
                # pas un total de corpus. Le corpus retenu est `corpus_total`.
                "corpus_total": _pi_corpus_total,
                "prisma_identification": _pi_figures,
                "errors": errors,
                "total_found": total_found,
                "sources": _sources_final,
                "message": f"{ingested} articles ingérés depuis 13 sources ({_src_summary}), {errors} erreurs.",
            }

        # Arrière-plan : cross-encoder (réordonne le sous-ensemble pertinent) puis
        # clustering UMAP/HDBSCAN + knowledge graph (cache DB). Réservé au chemin
        # /populate (le pipeline complet a ses propres étapes).
        if _pipeline_callback is None:
            def _post_done_bg(_sid, _query):
                try:
                    _n_ce = _run_cross_encoder_rerank(_sid, _query)
                    if _n_ce:
                        logger.info(f"Post-populate cross-encoder {_sid}: {_n_ce} articles réordonnés.")
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
                # AUTO_PIPELINE_AFTER_SEARCH=0 pour s'en tenir à la recherche seule.
                if _auto_ok[0] and _auto_pipeline_after_search():
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

        logger.info(f"Populate user_scenario {scenario_id}: {ingested} articles ingérés (13 sources, parallèle).")
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
    1. ingest    – Ingestion multi-sources (PubMed+OpenAlex+Crossref+EuropePMC+Preprints+SemanticScholar+DOAJ+ClinicalTrials.gov+CORE+arXiv+OpenAIRE+medRxiv+bioRxiv)
    2. fulltext  – Récupération full-text (PMC→EuropePMC→Unpaywall) pendant que les IDs sont frais
    3. embed     – Embeddings sur title+abstract+fulltext chunks (contenu enrichi)
    4. rerank    – Score cosinus via pgvector (pas de re-embedding API)
    5. pico      – Extraction PICO (LLM, utilise fulltext si dispo)
    6. metadata  – Extraction métadonnées étude (LLM)
    7. clustering – K-means sur embeddings pgvector
    """
    from .evidence import _generate_evidence_brief_llm  # lazy: evidence is loaded after this module
    from .variables import _generate_variables_from_pico  # lazy: variables is loaded after this module
    import time as _time

    STEP_ORDER = ["ingest", "fulltext", "embed", "rerank", "pico", "metadata",
                  "clustering", "knowledge_graph", "evidence", "variables", "actions"]

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

    _user_scenario_pipeline_jobs[scenario_id] = {
        "overall_status": "running",
        "current_step": "ingest",
        "lang": lang or "fr",
        "steps": {k: {"status": "pending"} for k in STEP_ORDER},
    }

    try:
        # ── Étape 1 : Ingestion multi-sources ────────────────────────────────────
        update_step("ingest", "running")
        # Si le scénario a DÉJÀ été peuplé (populate terminé — typiquement par la
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
            logger.info(f"Pipeline {scenario_id}: corpus déjà construit ({ingested} docs, populate=done) — "
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
            _user_scenario_pipeline_jobs[scenario_id]["message"] = "Aucun article trouvé (13 sources interrogées)."
            return

        # ── Étape 2 : Full-text multi-sources (avant embedding pour enrichir les chunks) (PMC → EuropePMC → Unpaywall → bioRxiv → Semantic Scholar → OpenAlex) ──
        update_step("fulltext", "running")
        try:
            import re as _re
            import subprocess as _subprocess
            import tempfile as _tempfile
            import xml.etree.ElementTree as _ET_ft
            import requests as _requests

            _NCBI_BASE_FT = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            _EPMC_BASE_FT = "https://www.ebi.ac.uk/europepmc/webservices/rest"
            _UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "literev@gesica.ch")
            _CHUNK_SIZE_FT = 4000
            _CHUNK_OVERLAP_FT = 400

            def _ft_get(url, params=None, timeout=20):
                for _att in range(3):
                    try:
                        _r = _requests.get(url, params=params, timeout=timeout,
                                           headers={"User-Agent": "LiteRev-Evidence/1.0"})
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
                _skip = {"ref-list","ack","fn-group","glossary","app-group","notes","bio","author-notes"}
                try:
                    _root = _ET_ft.fromstring(xml_str)
                except _ET_ft.ParseError:
                    xml_str = _re.sub(r"&(?!amp;|lt;|gt;|apos;|quot;)", "&amp;", xml_str)
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
                _txt = _re.sub(r"\s+", " ", " ".join(_parts)).strip()
                return _txt if len(_txt) > 50 else None

            def _extract_pdf_text_ft(pdf_url):
                try:
                    _r = _requests.get(pdf_url, timeout=30, stream=True,
                                       headers={"User-Agent": "LiteRev-Evidence/1.0"})
                    if _r.status_code != 200:
                        return None
                    _ct = _r.headers.get("content-type", "")
                    if "pdf" not in _ct.lower() and not pdf_url.lower().endswith(".pdf"):
                        _txt = _re.sub(r"<[^>]+>", " ", _r.text)
                        _txt = sanitize_db_text(_re.sub(r"\s+", " ", _txt)).strip()
                        return _txt if len(_txt) > 500 else None
                    with _tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as _f:
                        for _chunk in _r.iter_content(chunk_size=8192):
                            _f.write(_chunk)
                        _tmp = _f.name
                    _res = _subprocess.run(["pdftotext", "-layout", _tmp, "-"],
                                           capture_output=True, text=True, timeout=30)
                    os.unlink(_tmp)
                    if _res.returncode == 0 and _res.stdout.strip():
                        # pdftotext émet des NUL sur certains PDF mal formés ; sans ce
                        # nettoyage l'INSERT du chunk échoue et l'article perd son texte.
                        _txt = sanitize_db_text(_re.sub(r"\s+", " ", _res.stdout)).strip()
                        return _txt if len(_txt) > 500 else None
                except Exception:
                    pass
                return None

            def _resolve_pmcid_ft(ext_id, pmid_val, doi_val):
                if ext_id and "PMC" in ext_id.upper():
                    _m = _re.search(r"(\d{5,10})", ext_id)
                    if _m:
                        return f"PMC{_m.group(1)}"
                if ext_id and _re.match(r"^PMC\d+$", ext_id.upper()):
                    return ext_id.upper()
                _cand = None
                if ext_id and ext_id.isdigit():
                    _cand = ext_id
                elif pmid_val:
                    _cand = str(pmid_val).replace("PMID:", "").strip()
                if _cand:
                    _r2 = _ft_get(f"{_NCBI_BASE_FT}/esummary.fcgi",
                                  params={"db": "pubmed", "id": _cand, "retmode": "json"})
                    if _r2 and _r2.status_code == 200:
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
                    _r3 = _ft_get("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
                                  params={"ids": doi_val, "format": "json",
                                          "tool": "literev", "email": _UNPAYWALL_EMAIL})
                    if _r3 and _r3.status_code == 200:
                        try:
                            _recs = _r3.json().get("records", [])
                            if _recs and _recs[0].get("pmcid"):
                                return _recs[0]["pmcid"]
                        except Exception:
                            pass
                return None

            def _chunk_text_ft(text_str):
                # Découpage SÉMANTIQUE par phrases : on coupe aux frontières de
                # phrases (et non au milieu d'un mot/d'une idée), puis on fusionne
                # en chunks d'environ _CHUNK_SIZE_FT caractères avec un recouvrement
                # au niveau de la phrase. Plus cohérent que l'ancienne fenêtre brute.
                text_str = _re.sub(r"\s+", " ", text_str).strip()
                if not text_str:
                    return []
                _sentences = _re.split(r"(?<=[.!?])\s+", text_str)
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
                with engine.begin() as _c:
                    _c.execute(text(
                        "DELETE FROM document_chunk WHERE document_id = :did "
                        "AND chunk_type IN ('fulltext_section', 'full_text')"
                    ), {"did": doc_id})
                    for _i, _chunk_text in enumerate(chunks):
                        _meta = json.dumps({"source": source_label, "chunk_index": _i})
                        _c.execute(text("""
                            INSERT INTO document_chunk
                                (document_id, content, chunk_index, chunk_type, chunk_weight, metadata_json)
                            VALUES (:did, :content, :idx, 'fulltext_section', 1.0, CAST(:meta AS jsonb))
                        # Filet de sécurité au POINT D'ÉCRITURE : l'extracteur nettoie
                        # déjà, mais cette fonction est appelée avec du texte d'autres
                        # provenances (PMC, EuropePMC…) et un seul NUL fait échouer toute
                        # la transaction — DELETE compris, donc zéro chunk conservé.
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
                            _batch = _new_chunks[_bi:_bi+50]
                            _texts = [_truncate_to_tokens(r["content"]) for r in _batch]
                            _emb_resp = emb_client.embeddings.create(
                                model="text-embedding-3-small", input=_texts)
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

            # Récupérer tous les documents du scénario sans full-text
            with engine.connect() as conn:
                ft_rows = conn.execute(text("""
                    SELECT ld.id, ld.external_id, ld.doi, ld.pmid, ld.source
                    FROM literature_document ld
                    JOIN article_scenarios asn ON asn.document_id = ld.id
                    WHERE asn.scenario_id = :sid
                      AND ld.project_context = 'literev'
                      AND (ld.has_fulltext IS NULL OR ld.has_fulltext = false)
                    ORDER BY ld.id
                """), {"sid": scenario_id}).mappings().fetchall()

            ft_fetched = 0
            ft_errors = 0
            ft_total = len(ft_rows)

            # Initialiser le client OpenAI pour l'embedding des chunks fulltext
            _ft_emb_client = None
            try:
                from llm_usage import MeteredOpenAI as _OAI_ft
                _ft_emb_client = _OAI_ft(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception:
                pass

            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            
            _ft_lock = threading.Lock()
            _ft_done_count = 0
            
            def _process_ft_row(row):
                _ext_id = row["external_id"] or ""
                _doi = _normalize_doi(row["doi"] or "") or ""
                _pmid = row["pmid"] or ""
                _source = row["source"] or ""
                _fulltext = None
                _source_used = None
                _ft_fail_reasons = []
                
                try:
                    # Source 1 : PMC
                    _pmcid = _resolve_pmcid_ft(_ext_id, _pmid, _doi)
                    if _pmcid:
                        _r_pmc = _ft_get(f"{_EPMC_BASE_FT}/{_pmcid}/fullTextXML", timeout=30)
                        if _r_pmc and _r_pmc.status_code == 200 and _r_pmc.text.strip().startswith("<"):
                            _fulltext = _parse_pmc_xml_ft(_r_pmc.text)
                            if _fulltext and len(_fulltext) > 500:
                                _source_used = f"europepmc:{_pmcid}"
                            else:
                                _ft_fail_reasons.append(f"europepmc:{_pmcid}:xml_parse_empty")
                        elif _r_pmc:
                            _ft_fail_reasons.append(f"europepmc:{_pmcid}:http_{_r_pmc.status_code}")
                        
                        if not _fulltext:
                            _pmcid_num = _pmcid.replace("PMC", "")
                            _r_ncbi = _ft_get(f"{_NCBI_BASE_FT}/efetch.fcgi",
                                             params={"db": "pmc", "id": _pmcid_num,
                                                     "rettype": "full", "retmode": "xml"}, timeout=30)
                            if _r_ncbi and _r_ncbi.status_code == 200 and _r_ncbi.text.strip().startswith("<"):
                                _fulltext = _parse_pmc_xml_ft(_r_ncbi.text)
                                if _fulltext and len(_fulltext) > 500:
                                    _source_used = f"pmc:{_pmcid}"
                                else:
                                    _fulltext = None
                                    _ft_fail_reasons.append(f"pmc:{_pmcid}:xml_parse_empty")
                            elif _r_ncbi:
                                _ft_fail_reasons.append(f"pmc:{_pmcid}:http_{_r_ncbi.status_code}")
                    else:
                        _ft_fail_reasons.append("pmcid:not_resolved")
                    
                    # Source 2 : Unpaywall
                    if not _fulltext and _doi and _doi.startswith("10."):
                        _r_uw = _ft_get(f"https://api.unpaywall.org/v2/{_doi}",
                                        params={"email": _UNPAYWALL_EMAIL})
                        if _r_uw and _r_uw.status_code == 200:
                            try:
                                _uw_data = _r_uw.json()
                                _pdf_url = None
                                _best = _uw_data.get("best_oa_location") or {}
                                _pdf_url = _best.get("url_for_pdf") or _best.get("url")
                                if not _pdf_url:
                                    for _loc in _uw_data.get("oa_locations", []):
                                        if _loc.get("url_for_pdf"):
                                            _pdf_url = _loc["url_for_pdf"]
                                            break
                                if _pdf_url:
                                    _fulltext = _extract_pdf_text_ft(_pdf_url)
                                    if _fulltext and len(_fulltext) > 500:
                                        _source_used = "unpaywall"
                                    else:
                                        _fulltext = None
                                        _ft_fail_reasons.append("unpaywall:pdf_empty")
                                else:
                                    _ft_fail_reasons.append(f"unpaywall:not_oa(is_oa={_uw_data.get('is_oa')})")
                            except Exception as _uw_e:
                                _ft_fail_reasons.append(f"unpaywall:parse_error:{_uw_e}")
                        elif _r_uw:
                            _ft_fail_reasons.append(f"unpaywall:http_{_r_uw.status_code}")
                        else:
                            _ft_fail_reasons.append("unpaywall:no_doi" if not _doi else "unpaywall:timeout")
                    elif not _doi:
                        _ft_fail_reasons.append("unpaywall:skipped_no_doi")
                    
                    # Source 3 : bioRxiv/medRxiv
                    if not _fulltext and _doi and _doi.startswith("10.1101/"):
                        for _srv in ["biorxiv", "medrxiv"]:
                            _r_bx = _ft_get(f"https://api.biorxiv.org/details/{_srv}/{_doi}/na/json")
                            if _r_bx and _r_bx.status_code == 200:
                                try:
                                    _coll = _r_bx.json().get("collection", [])
                                    if _coll:
                                        _pdf_url = f"https://www.{_srv}.org/content/{_doi}.full.pdf"
                                        _fulltext = _extract_pdf_text_ft(_pdf_url)
                                        if _fulltext and len(_fulltext) > 500:
                                            _source_used = _srv
                                            break
                                        else:
                                            _fulltext = None
                                            _ft_fail_reasons.append(f"{_srv}:pdf_empty")
                                    else:
                                        _ft_fail_reasons.append(f"{_srv}:not_found")
                                except Exception as _bx_e:
                                    _ft_fail_reasons.append(f"{_srv}:parse_error:{_bx_e}")
                    
                    # Source 4 : Semantic Scholar
                    if not _fulltext:
                        _ss_id = None
                        if _doi:
                            _ss_id = f"DOI:{_doi}"
                        elif _pmid:
                            _ss_id = f"PMID:{_pmid}"
                        elif _ext_id and _ext_id.upper().startswith("PMC"):
                            _ss_id = f"PMCID:{_ext_id}"
                        if _ss_id:
                            _r_ss = _ft_get(
                                f"https://api.semanticscholar.org/graph/v1/paper/{_ss_id}",
                                params={"fields": "openAccessPdf,abstract"},
                            )
                            if _r_ss and _r_ss.status_code == 200:
                                try:
                                    _ss_data = _r_ss.json()
                                    _oa_pdf = _ss_data.get("openAccessPdf")
                                    if _oa_pdf and _oa_pdf.get("url"):
                                        _fulltext = _extract_pdf_text_ft(_oa_pdf["url"])
                                        if _fulltext and len(_fulltext) > 500:
                                            _source_used = "semanticscholar"
                                        else:
                                            _fulltext = None
                                            _ft_fail_reasons.append("semanticscholar:pdf_empty")
                                    else:
                                        _ft_fail_reasons.append(f"semanticscholar:no_oa_pdf")
                                except Exception as _ss_e:
                                    _ft_fail_reasons.append(f"semanticscholar:parse_error:{_ss_e}")
                            elif _r_ss:
                                _ft_fail_reasons.append(f"semanticscholar:http_{_r_ss.status_code}")
                        else:
                            _ft_fail_reasons.append("semanticscholar:no_identifier")
                    
                    # Source 5 : OpenAlex
                    if not _fulltext and (_ext_id.startswith("W") or _doi):
                        _oa_work_url = (
                            f"https://api.openalex.org/works/{_ext_id}"
                            if _ext_id.startswith("W")
                            else f"https://api.openalex.org/works/doi:{_doi}"
                        )
                        _r_oa = _ft_get(_oa_work_url, params={"select": "open_access"})
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
                                        _ft_fail_reasons.append("openalex:pdf_empty")
                                else:
                                    _ft_fail_reasons.append(f"openalex:not_oa(is_oa={_oa_info.get('is_oa')})")
                            except Exception as _oa_e:
                                _ft_fail_reasons.append(f"openalex:parse_error:{_oa_e}")
                        elif _r_oa:
                            _ft_fail_reasons.append(f"openalex:http_{_r_oa.status_code}")
                    
                    if _fulltext and _source_used:
                        _chunks_ft = _chunk_text_ft(_fulltext)
                        if _chunks_ft:
                            _insert_fulltext_chunks_ft(row["id"], _chunks_ft, _source_used, _ft_emb_client)
                            return True, None
                        else:
                            logger.warning(f"Fulltext doc {row['id']}: text retrieved but produced 0 chunks (source={_source_used})")
                            return False, "0_chunks"
                    else:
                        logger.info(f"Fulltext unavailable doc {row['id']} (ext_id={_ext_id}, doi={_doi}): {' | '.join(_ft_fail_reasons) or 'no_sources_tried'}")
                        return False, "not_found"
                except Exception as _ft_e:
                    logger.warning(f"Fulltext doc {row['id']}: {_ft_e}")
                    return False, str(_ft_e)

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(_process_ft_row, row): row for row in ft_rows}
                for future in as_completed(futures):
                    try:
                        success, _ = future.result()
                        with _ft_lock:
                            _ft_done_count += 1
                            if success:
                                ft_fetched += 1
                            else:
                                ft_errors += 1
                            
                            if _ft_done_count % 10 == 0 or _ft_done_count == ft_total:
                                update_step("fulltext", "running",
                                            done=_ft_done_count, total=ft_total, paywall=ft_errors,
                                            pct=round((_ft_done_count) / ft_total * 100, 1) if ft_total > 0 else 0)
                    except Exception as e:
                        with _ft_lock:
                            _ft_done_count += 1
                            ft_errors += 1
                            logger.error(f"Error in fulltext worker: {e}")

            update_step("fulltext", "done", fetched=ft_fetched, total=ft_total,
                        paywall_or_failed=ft_errors)
        except Exception as e:
            update_step("fulltext", "error", error=str(e))

        # ── Étape 3 : Embeddings (title_abstract + fulltext_section — contenu enrichi) ────────
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
                            model="text-embedding-3-small",
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

        # ── Étape 4 : Rerank via pgvector (cosinus sur embeddings stockés) ──────────────
        update_step("rerank", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from llm_usage import MeteredOpenAI as _OAI_rr
                _rr_client = _OAI_rr(api_key=openai_key)
                _rr_resp = _rr_client.embeddings.create(
                    model="text-embedding-3-small", input=query[:2000])
                _q_vec = "[" + ",".join(str(x) for x in _rr_resp.data[0].embedding) + "]"
                with engine.begin() as _rr_conn:
                    _rr_result = _rr_conn.execute(text("""
                        UPDATE article_scenarios ars
                        SET similarity_score = sub.best_sim
                        FROM (
                            SELECT c.document_id,
                                   MAX(1.0 - (c.embedding <=> CAST(:q_vec AS vector))) AS best_sim
                            FROM document_chunk c
                            JOIN article_scenarios a ON a.document_id = c.document_id
                            WHERE a.scenario_id = :sid
                              AND c.embedding IS NOT NULL
                              AND c.chunk_type IN ('title_abstract', 'fulltext_section')
                            GROUP BY c.document_id
                        ) sub
                        WHERE ars.scenario_id = :sid
                          AND ars.document_id = sub.document_id
                    """), {"q_vec": _q_vec, "sid": scenario_id})
                n_reranked = _rr_result.rowcount
                update_step("rerank", "done", updated=n_reranked)

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
                update_step("rerank", "skipped", reason="Clé OpenAI non configurée")
        except Exception as e:
            update_step("rerank", "error", error=str(e))

        # ── Étape 5 : Extraction PICO ─────────────────────────────────────────────────────
        update_step("pico", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from llm_usage import MeteredOpenAI as _OAI
                from datetime import datetime, timezone
                _client = _OAI(api_key=openai_key, timeout=90.0)
                system_prompt_pico = (
                    "You are a systematic review expert. "
                    "Extract PICO elements and return ONLY valid JSON:\n"
                    '{"P":"Population","I":"Intervention","C":"Comparator or Not specified",'
                    '"O":"Outcome(s)","study_design":"RCT|Cohort|Systematic review|etc",'
                    '"pico_confidence":0.0-1.0,"pico_notes":""}\n'
                    "Be concise (max 2 sentences per field). Return ONLY the JSON."
                )
                with engine.connect() as conn:
                    pico_rows = conn.execute(text("""
                        SELECT ld.id, ld.title, ld.abstract
                        FROM literature_document ld
                        JOIN article_scenarios asn ON asn.document_id = ld.id
                        WHERE asn.scenario_id = :sid
                          AND ld.project_context = 'literev'
                          AND (ld.pico_json IS NULL OR (ld.pico_json->>'pico_confidence')::float < 0.5)
                          AND ld.abstract IS NOT NULL AND length(ld.abstract) > 50
                          AND COALESCE(ld.pico_attempts, 0) < 3  -- borne les échecs déterministes (token-bleed)
                        ORDER BY ld.id
                    """), {"sid": scenario_id}).mappings().fetchall()

                pico_extracted = 0
                pico_errors = 0
                for row in pico_rows:
                    try:
                        response = _client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {"role": "system", "content": system_prompt_pico},
                                {"role": "user", "content": f"Title: {row['title']}\n\nAbstract: {(row['abstract'] or '')[:3000]}"},
                            ],
                            temperature=0,
                            seed=42,
                            max_tokens=800,  # 400 tronquait le JSON verbeux → JSON invalide
                            response_format={"type": "json_object"},
                        )
                    except Exception as e:
                        logger.warning(f"Pipeline PICO API article {row['id']}: {e}")
                        pico_errors += 1
                        _time.sleep(0.05)
                        continue  # transitoire — ne PAS consommer une tentative
                    # Réponse reçue → COMPTE la tentative (borne le token-bleed) ;
                    # remplissage tolérant des clés plutôt que rejet en boucle.
                    try:
                        pico = json.loads(response.choices[0].message.content)
                        if not isinstance(pico, dict):
                            raise ValueError("réponse PICO non-dict")
                        for _k in ("P", "I", "C", "O"):
                            pico.setdefault(_k, "")
                        pico.setdefault("study_design", "non précisé")
                        try:
                            pico["pico_confidence"] = float(pico.get("pico_confidence", 0.3))
                        except (TypeError, ValueError):
                            pico["pico_confidence"] = 0.3
                        pico["pico_notes"] = pico.get("pico_notes", "")
                        with engine.begin() as conn:
                            conn.execute(text("""
                                UPDATE literature_document
                                SET pico_json = CAST(:pico AS jsonb), pico_extracted_at = :ts,
                                    pico_attempts = COALESCE(pico_attempts, 0) + 1
                                WHERE id = :article_id
                            """), {"pico": json.dumps(pico), "ts": datetime.now(timezone.utc), "article_id": row["id"]})
                        pico_extracted += 1
                    except Exception as e:
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
                            errors=pico_errors)
            else:
                update_step("pico", "skipped", reason="Clé OpenAI non configurée")
        except Exception as e:
            update_step("pico", "error", error=str(e))

        # ── Étape 6 : Extraction métadonnées ─────────────────────────────────
        update_step("metadata", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from llm_usage import MeteredOpenAI as _OAI2
                from datetime import datetime, timezone
                _client2 = _OAI2(api_key=openai_key)
                system_prompt_meta = (
                    "You are a biomedical librarian. Extract metadata from this article and return ONLY valid JSON:\n"
                    '{"study_type":"RCT|Cohort|Case-control|Cross-sectional|Systematic review|Meta-analysis|Case report|Editorial|Other",'
                    '"sample_size":null,"country":"ISO2 or null","setting":"hospital|prehospital|community|other|null",'
                    '"primary_outcome":"brief description or null","funding":"public|industry|mixed|not reported",'
                    '"bias_risk":"low|moderate|high|unclear","metadata_confidence":0.0-1.0}\n'
                    "Return ONLY the JSON."
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
                          AND (ld.metadata_json IS NULL OR ld.metadata_json = '{}'::jsonb)
                        ORDER BY ld.id
                    """), {"sid": scenario_id}).mappings().fetchall()

                meta_extracted = 0
                meta_errors = 0
                for row in meta_rows:
                    try:
                        response = _client2.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {"role": "system", "content": system_prompt_meta},
                                {"role": "user", "content": f"Title: {row['title']}\n\nAbstract: {(row['abstract'] or '')[:2000]}"},
                            ],
                            temperature=0.1,
                            max_tokens=300,
                            response_format={"type": "json_object"},
                        )
                        metadata = json.loads(response.choices[0].message.content)
                        metadata["metadata_confidence"] = float(metadata.get("metadata_confidence", 0.5))
                        # Renseigner les colonnes structurées depuis le JSON extrait
                        # (study_design / sample_size), puis calculer un quality_score
                        # déterministe — sinon l'évaluation GRADE buckette tout en « Faible ».
                        study_design = metadata.get("study_type") or row.get("study_design")
                        sample_size = _coerce_int(metadata.get("sample_size")) or row.get("sample_size")
                        quality_score = _compute_quality_score(
                            study_design=study_design,
                            year=row.get("year"),
                            sample_size=sample_size,
                            citation_count=row.get("citation_count"),
                            open_access=row.get("open_access"),
                            bias_risk=metadata.get("bias_risk"),
                        )
                        with engine.begin() as conn:
                            conn.execute(text("""
                                UPDATE literature_document
                                SET metadata_json = CAST(:meta AS jsonb),
                                    study_design = COALESCE(:study_design, study_design),
                                    sample_size = COALESCE(:sample_size, sample_size),
                                    quality_score = COALESCE(:quality_score, quality_score)
                                WHERE id = :article_id
                            """), {
                                "meta": json.dumps(metadata),
                                "study_design": study_design,
                                "sample_size": sample_size,
                                "quality_score": quality_score,
                                "article_id": row["id"],
                            })
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
                update_step("metadata", "skipped", reason="Clé OpenAI non configurée")
        except Exception as e:
            update_step("metadata", "error", error=str(e))

        # ── Étape 7 : Clustering (UMAP+HDBSCAN avec fallback KMeans) ────────────
        update_step("clustering", "running")
        try:
            # Clustering du pipeline sur le SOUS-ENSEMBLE PERTINENT (≥ seuil sémantique
            # OU inclus manuellement ; jamais exclus) — cohérent avec le clustering à la
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
                # à la première ouverture » — soit 12 appels LLM sous les yeux de l'utilisateur).
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

        # ── Knowledge graph (cache DB) — visualisation prête ──────────────────
        try:
            update_step("knowledge_graph", "running")
            _precompute_user_kg(scenario_id)
            # Carte des concepts : annotation LLM (une fois par article) des articles
            # pertinents sans concepts, puis cache — l'onglet n'a rien à calculer.
            from .knowledge_graph import _precompute_concept_graph  # lazy: keeps the import list short
            _cg = _precompute_concept_graph(scenario_id, extract=True) or {}
            update_step("knowledge_graph", "done", concepts_articles=_cg.get("n_with_concepts"),
                        concept_nodes=len(_cg.get("nodes") or []))
        except Exception as _e_kg:
            logger.warning(f"Précalcul KG pipeline {scenario_id}: {_e_kg}")
            update_step("knowledge_graph", "error", error=str(_e_kg))

        # ── Evidence Brief (narratif LLM, mis en cache) — prêt sans clic ───────
        try:
            update_step("evidence", "running")
            _generate_evidence_brief_llm(scenario_id, lang=lang or "fr")
            update_step("evidence", "done")
        except Exception as _e_ev:
            logger.warning(f"Evidence brief pipeline {scenario_id}: {_e_ev}")
            update_step("evidence", "error", error=str(_e_ev))

        # ── Variables & Modèle (spec déterministe : outcome, features, algorithme,
        # data_template, paramètres SEIR) — le scénario est « modèle-prêt » d'emblée.
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
        # langue de l'interface — elles l'étaient à la première ouverture de la carte.
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
            f"Pipeline terminé."
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
                logger.info(f"Pipeline complet {scenario_id}: compteurs cohérents — "
                            f"{_cc['corpus_links']} articles (liste, en-tête, PRISMA, étape 2).")
            else:
                logger.warning(f"Pipeline complet {scenario_id}: compteurs INCOHÉRENTS — {_cc['mismatches']}")
        except Exception as _e_cc:
            logger.warning(f"Pipeline {scenario_id}: vérification des compteurs impossible: {_e_cc}")
        logger.info(f"Pipeline complet {scenario_id}: terminé.")

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
def trigger_full_pipeline_with_brief(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Déclenche le pipeline complet incluant :
    1. Reranking sémantique
    2. Génération Evidence Brief LLM
    3. Génération Variables & Modèle
    Fonctionne pour GESICA et user_scenarios.
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

    def _run():
        logger.info(f"Full pipeline with brief: {scenario_id}")
        # 1. Reranking
        _run_semantic_rerank_inline(scenario_id, query)
        # 2. Evidence Brief LLM
        _generate_evidence_brief_llm(scenario_id, force=True)
        # 3. Variables & Modèle
        _generate_variables_from_pico(scenario_id)
        logger.info(f"Full pipeline with brief done: {scenario_id}")

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "steps": ["rerank", "evidence_brief", "variables"]}
