"""Batch LLM enrichment: PICO, metadata, full text, and the main finding of each article.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from fastapi import Depends, HTTPException
from sqlalchemy import text

from .core import (_env_int, _is_openai_quota_error, _trip_openai_cooldown, app, engine, logger,
                   require_api_key)
from .documents import METADATA_UPDATE_SQL, _llm_lang_directive, _metadata_update_params
from .pico_map import (PicoUnreadable, extract_finding, extract_pico, fulltext_excerpt,
                       json_completion, pico_fulltext_chunks)
from .scenario_store import (SCOPES, _get_scenario_threshold, _get_user_scenario_or_404,
                             relevance_order_sql, relevant_gate_sql, scenario_scope_sql)
from llm_usage import model_for as _model


def _scope(scope: Optional[str]) -> str:
    """Valide la portée d'un lot. Par défaut tout le scénario, comme avant."""
    value = (scope or "all").strip().lower()
    if value not in SCOPES:
        raise HTTPException(
            status_code=400,
            detail=f"Portée inconnue : {scope!r} (attendu : {', '.join(SCOPES)})")
    return value

# ─── Enrichissement LLM Batch ────────────────────────────────────────────────

@app.post("/pico/extract")
def extract_pico_batch(
    scenario_id: Optional[str] = None,
    limit: int = 100000,
    scope: Optional[str] = None,
    _: None = Depends(require_api_key),
):
    """
    Extrait le PICO pour un lot d'articles (un scénario, ou tout le corpus).
    Traite uniquement les articles sans PICO ou avec un PICO de faible confiance.

    `scope` : `all` pour tout le scénario, `relevant` pour son seul sous-ensemble
    pertinent. Sans scénario, la portée ne s'applique pas.
    """
    scope = _scope(scope)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=503, detail="Clé OpenAI non configurée")

    # Récupérer les articles sans PICO
    with engine.connect() as conn:
        if scenario_id:
            rows = conn.execute(text(f"""
                SELECT ld.id, ld.title, ld.abstract
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.project_context = 'literev'
                  AND ({scenario_scope_sql(scope)})
                  AND (ld.pico_json IS NULL OR (ld.pico_json->>'pico_confidence')::float < 0.5)
                  AND COALESCE(ld.pico_attempts, 0) < 3  -- borne les échecs déterministes (token-bleed)
                ORDER BY ld.id
                LIMIT :lim
            """), {"sid": scenario_id, "lim": limit}).mappings().fetchall()
        else:
            rows = conn.execute(text("""
                SELECT id, title, abstract
                FROM literature_document
                WHERE project_context = 'literev'
                  AND (pico_json IS NULL OR (pico_json->>'pico_confidence')::float < 0.5)
                  AND abstract IS NOT NULL AND length(abstract) > 50
                  AND COALESCE(pico_attempts, 0) < 3  -- borne les échecs déterministes (token-bleed)
                ORDER BY id
                LIMIT :lim
            """), {"lim": limit}).mappings().fetchall()

    extracted = 0
    skipped = 0
    errors = 0

    try:
        from llm_usage import MeteredOpenAI as _OAI
        from datetime import datetime, timezone
        _client = _OAI(api_key=openai_key, timeout=90.0)

        for row in rows:
            article_id = row["id"]
            title = row["title"] or ""
            abstract = row["abstract"] or ""
            if not abstract or len(abstract) < 50:
                skipped += 1
                continue
            # Le prompt, le texte lu (résumé + méthodes et résultats du texte intégral) et
            # la normalisation sont ceux de tous les rédacteurs PICO (api/pico_map).
            try:
                with engine.connect() as conn:
                    chunks = pico_fulltext_chunks(conn, article_id)
                pico, _source = extract_pico(_client, title, abstract, chunks,
                                             label=f"PICO batch {article_id}")
            except PicoUnreadable as e:
                # Une RÉPONSE inutilisable → on COMPTE la tentative (borne le token-bleed :
                # une sortie déterministe malformée ne sera pas ré-extraite à l'infini).
                logger.warning(f"PICO batch parse error article {article_id}: {e}")
                try:
                    with engine.begin() as conn:
                        conn.execute(text(
                            "UPDATE literature_document SET pico_attempts = COALESCE(pico_attempts, 0) + 1 WHERE id = :aid"
                        ), {"aid": article_id})
                except Exception:
                    pass
                errors += 1
                continue
            except Exception as e:
                # Appel en échec (quota/réseau) : TRANSITOIRE → on ne compte PAS de
                # tentative (réessai quand l'API est saine).
                logger.warning(f"PICO batch API error article {article_id}: {e}")
                errors += 1
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE literature_document
                        SET pico_json = CAST(:pico AS jsonb), pico_extracted_at = :ts,
                            pico_attempts = COALESCE(pico_attempts, 0) + 1
                        WHERE id = :article_id
                    """), {
                        "pico": json.dumps(pico),
                        "ts": datetime.now(timezone.utc),
                        "article_id": article_id,
                    })
                extracted += 1
            except Exception as e:
                logger.warning(f"PICO batch write error article {article_id}: {e}")
                errors += 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LLM batch: {str(e)}")

    return {
        "extracted": extracted,
        "skipped": skipped,
        "errors": errors,
        "message": f"{extracted} articles enrichis, {skipped} ignorés, {errors} erreurs",
    }


@app.post("/metadata/extract")
def extract_metadata_batch(
    scenario_id: Optional[str] = None,
    limit: int = 100000,
    scope: Optional[str] = None,
    _: None = Depends(require_api_key),
):
    """
    Enrichit les métadonnées (type d'étude, effectif, pays, risque de biais) via LLM pour
    un lot d'articles, et calcule le score de qualité déterministe qu'elles permettent :
    la même écriture que l'étape du pipeline (`_metadata_update_params`).

    `scope` : `all` pour tout le scénario, `relevant` pour son seul sous-ensemble
    pertinent.
    """
    scope = _scope(scope)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=503, detail="Clé OpenAI non configurée")

    # Ce que la ligne sait déjà entre dans le score : année, citations, accès ouvert, et
    # le devis ou l'effectif qu'une extraction précédente aurait laissés.
    with engine.connect() as conn:
        if scenario_id:
            rows = conn.execute(text(f"""
                SELECT ld.id, ld.title, ld.abstract, ld.source, ld.year, ld.citation_count,
                       ld.open_access, ld.study_design, ld.sample_size
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.project_context = 'literev'
                  AND ({scenario_scope_sql(scope)})
                  AND (ld.metadata_json IS NULL OR ld.metadata_json = '{{}}'::jsonb)
                ORDER BY ld.id
                LIMIT :lim
            """), {"sid": scenario_id, "lim": limit}).mappings().fetchall()
        else:
            rows = conn.execute(text("""
                SELECT id, title, abstract, source, year, citation_count,
                       open_access, study_design, sample_size
                FROM literature_document
                WHERE project_context = 'literev'
                  AND (metadata_json IS NULL OR metadata_json = '{}'::jsonb)
                  AND abstract IS NOT NULL AND length(abstract) > 30
                ORDER BY id
                LIMIT :lim
            """), {"lim": limit}).mappings().fetchall()

    extracted = 0
    skipped = 0
    errors = 0

    system_prompt = (
        "You are a biomedical librarian. Extract metadata from this article and return ONLY valid JSON:\n"
        '{"study_type":"RCT|Cohort|Case-control|Cross-sectional|Systematic review|Meta-analysis|Case report|Editorial|Other",'
        '"sample_size":null,"country":"ISO2 or null","setting":"hospital|prehospital|community|other|null",'
        '"primary_outcome":"brief description or null","funding":"public|industry|mixed|not reported",'
        '"bias_risk":"low|moderate|high|unclear","metadata_confidence":0.0-1.0}\n'
        "Do not use the em dash character. Return ONLY the JSON."
    )

    try:
        from llm_usage import MeteredOpenAI as _OAI
        _client = _OAI(api_key=openai_key, timeout=90.0)

        for row in rows:
            article_id = row["id"]
            title = row["title"] or ""
            abstract = row["abstract"] or ""
            if not title:
                skipped += 1
                continue
            try:
                def _ask(messages):
                    return _client.chat.completions.create(
                        model=_model("bulk"),
                        messages=messages,
                        temperature=0.1,
                        max_tokens=300,
                        response_format={"type": "json_object"},
                    )

                # Une réponse coupée au plafond est redemandée une fois, plus courte, au
                # lieu de remonter en erreur d'analyse JSON que seul le journal voyait.
                metadata = json.loads(json_completion(
                    _ask, [{"role": "system", "content": system_prompt},
                           {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract[:2000]}"}],
                    f"metadata batch {article_id}"))
                metadata["metadata_confidence"] = float(metadata.get("metadata_confidence", 0.5))
                with engine.begin() as conn:
                    conn.execute(text(METADATA_UPDATE_SQL),
                                 _metadata_update_params(article_id, metadata, row))
                extracted += 1
            except Exception as e:
                logger.warning(f"Metadata batch error article {article_id}: {e}")
                errors += 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LLM batch: {str(e)}")

    return {
        "extracted": extracted,
        "skipped": skipped,
        "errors": errors,
        "message": f"{extracted} articles enrichis, {skipped} ignorés, {errors} erreurs",
    }


@app.post("/fulltext/fetch")
def fetch_fulltext_batch(
    scenario_id: Optional[str] = None,
    limit: int = 100000,
    scope: Optional[str] = None,
    _: None = Depends(require_api_key),
):
    """
    Tente de récupérer le texte intégral (via DOI/URL) pour un lot d'articles.
    Utilise Unpaywall + CrossRef pour les accès ouverts.

    `scope` : `all` pour tout le scénario, `relevant` pour son seul sous-ensemble
    pertinent.
    """
    import urllib.request

    scope = _scope(scope)
    with engine.connect() as conn:
        if scenario_id:
            rows = conn.execute(text(f"""
                SELECT ld.id, ld.title, ld.doi, ld.url
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.project_context = 'literev'
                  AND ({scenario_scope_sql(scope)})
                  AND (ld.has_fulltext IS NULL OR ld.has_fulltext = false)
                  AND ld.doi IS NOT NULL
                ORDER BY ld.id
                LIMIT :lim
            """), {"sid": scenario_id, "lim": limit}).mappings().fetchall()
        else:
            rows = conn.execute(text("""
                SELECT id, title, doi, url
                FROM literature_document
                WHERE project_context = 'literev'
                  AND (has_fulltext IS NULL OR has_fulltext = false)
                  AND doi IS NOT NULL
                ORDER BY id
                LIMIT :lim
            """), {"lim": limit}).mappings().fetchall()

    fetched = 0
    not_available = 0
    errors = 0

    for row in rows:
        article_id = row["id"]
        doi = row["doi"]
        if not doi:
            not_available += 1
            continue
        try:
            # Tenter Unpaywall
            unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=literev@gesica.ch"
            req = urllib.request.Request(unpaywall_url, headers={"User-Agent": "LiteRev/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            oa_url = None
            if data.get("is_oa") and data.get("best_oa_location"):
                oa_url = data["best_oa_location"].get("url_for_pdf") or data["best_oa_location"].get("url")
            if oa_url:
                # ── Un LIEN n'est pas un texte ───────────────────────────────
                # On posait `has_fulltext = true` et on écrasait `url`, sans stocker une
                # ligne de texte. Le panneau annonçait ensuite « couverture texte
                # intégral 2 582 sur 9 493 » et un relecteur en concluait que le système
                # détenait le texte de 2 582 articles ; pour au moins 28 d'entre eux il
                # ne détenait qu'une adresse. L'extraction, elle, se fiait au même
                # drapeau pour décider qu'il y avait quelque chose à lire.
                # Le lien a sa propre colonne, et ne touche plus ni le drapeau ni `url`.
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE literature_document
                        SET oa_url = :url, oa_url_found_at = NOW(),
                            url = COALESCE(NULLIF(TRIM(COALESCE(url, '')), ''), :url)
                        WHERE id = :article_id
                    """), {"url": oa_url, "article_id": article_id})
                fetched += 1
            else:
                not_available += 1
        except Exception as e:
            logger.warning(f"Fulltext fetch error article {article_id}: {e}")
            errors += 1

    return {
        "fetched": fetched,
        "not_available": not_available,
        "errors": errors,
        "message": f"{fetched} textes intégraux récupérés, {not_available} non disponibles, {errors} erreurs",
    }


def _scope_counts_sql() -> str:
    """Les compteurs d'un scénario, pour les DEUX portées, en UNE instruction.

    Le panneau annonce ce qu'un lot va traiter avant qu'on le lance ; si le total et le
    reste à faire venaient de deux requêtes, il pourrait annoncer un chiffre et en
    traiter un autre. `todo` est le nombre d'articles qu'un lot prendrait réellement,
    c'est-à-dire ce que l'appel au modèle va coûter."""
    # « Fait » doit être le CONTRAIRE de « à faire », sinon la barre et la ligne
    # au-dessous se contredisent : une extraction de faible confiance est reprise par
    # le lot, elle comptait pourtant comme faite, et fait + à faire dépassait le total.
    # Un article épuisé (trois tentatives) n'est dans ni l'un ni l'autre : il est
    # bloqué, ce qui n'est ni un succès ni une dépense à venir.
    _weak_pico = "(ld.pico_json->>'pico_confidence')::float < 0.5"
    done_pico = f"ld.pico_json IS NOT NULL AND NOT ({_weak_pico})"
    todo_pico = (f"(ld.pico_json IS NULL OR {_weak_pico})"
                 " AND COALESCE(ld.pico_attempts, 0) < 3")
    done_meta = "ld.metadata_json IS NOT NULL AND ld.metadata_json != '{}'::jsonb"
    todo_meta = "ld.metadata_json IS NULL OR ld.metadata_json = '{}'::jsonb"
    # « Fait » pour le texte intégral veut dire qu'on DÉTIENT le texte : un morceau de
    # texte intégral en base, la même question que pose scenario_counts_sql. Le drapeau
    # `has_fulltext` répondait « un lien d'accès ouvert existe », si bien que ce panneau
    # et l'onglet Corpus annonçaient deux nombres différents pour la même chose.
    _ft_chunk = ("EXISTS (SELECT 1 FROM document_chunk c WHERE c.document_id = ld.id"
                 " AND c.chunk_type IN ('fulltext_section', 'full_text'))")
    done_ft = _ft_chunk
    todo_ft = f"NOT {_ft_chunk} AND ld.doi IS NOT NULL"
    cols = []
    for scope in SCOPES:
        gate = scenario_scope_sql(scope)
        cols.append(f"COUNT(*) FILTER (WHERE {gate}) AS {scope}_total")
        for name, done, todo in (("pico", done_pico, todo_pico),
                                 ("metadata", done_meta, todo_meta),
                                 ("fulltext", done_ft, todo_ft)):
            cols.append(f"COUNT(*) FILTER (WHERE ({gate}) AND ({done})) AS {scope}_{name}_done")
            cols.append(f"COUNT(*) FILTER (WHERE ({gate}) AND ({todo})) AS {scope}_{name}_todo")
    return (f"SELECT {', '.join(cols)}\n"
            f"FROM literature_document ld\n"
            f"JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid\n"
            f"WHERE ld.project_context = 'literev'")


@app.get("/enrichment/status")
def get_enrichment_status(scenario_id: Optional[str] = None, scope: Optional[str] = None):
    """Le statut d'enrichissement (PICO, métadonnées, texte intégral).

    Pour un scénario, les compteurs des DEUX portées sont renvoyés dans `by_scope`,
    pour que le panneau puisse dire ce que chaque choix traiterait avant de le lancer.
    Les champs de premier niveau décrivent la portée demandée, `all` par défaut, ce qui
    laisse inchangée la réponse servie jusqu'ici."""
    scope = _scope(scope)
    with engine.connect() as conn:
        if scenario_id:
            row = conn.execute(text(_scope_counts_sql()),
                               {"sid": scenario_id}).mappings().fetchone() or {}
            by_scope = {}
            for name in SCOPES:
                total = int(row.get(f"{name}_total") or 0)
                denom = total or 1
                by_scope[name] = {"total": total}
                for job in ("pico", "metadata", "fulltext"):
                    done = int(row.get(f"{name}_{job}_done") or 0)
                    by_scope[name][job] = {
                        "count": done,
                        "pct": round(done / denom * 100, 1),
                        "todo": int(row.get(f"{name}_{job}_todo") or 0),
                    }
            chosen = by_scope[scope]
            return {
                "scenario_id": scenario_id,
                "scope": scope,
                "total": chosen["total"],
                "pico": chosen["pico"],
                "metadata": chosen["metadata"],
                "fulltext": chosen["fulltext"],
                "by_scope": by_scope,
            }
        else:
            row = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE pico_json IS NOT NULL
                        AND NOT ((pico_json->>'pico_confidence')::float < 0.5)) as with_pico,
                    COUNT(*) FILTER (WHERE (pico_json IS NULL
                        OR (pico_json->>'pico_confidence')::float < 0.5)
                        AND COALESCE(pico_attempts, 0) < 3) as todo_pico,
                    COUNT(CASE WHEN metadata_json IS NOT NULL AND metadata_json != '{}'::jsonb THEN 1 END) as with_metadata,
                    COUNT(*) FILTER (WHERE metadata_json IS NULL
                        OR metadata_json = '{}'::jsonb) as todo_metadata,
                    COUNT(CASE WHEN has_fulltext = true THEN 1 END) as with_fulltext,
                    COUNT(*) FILTER (WHERE (has_fulltext IS NULL OR has_fulltext = false)
                        AND doi IS NOT NULL) as todo_fulltext
                FROM literature_document
                WHERE project_context = 'literev'
            """)).mappings().fetchone()

    total = row["total"] or 1
    return {
        "scenario_id": scenario_id,
        "scope": scope,
        "total": row["total"],
        "pico": {"count": row["with_pico"], "pct": round(row["with_pico"] / total * 100, 1),
                 "todo": row["todo_pico"]},
        "metadata": {"count": row["with_metadata"], "pct": round(row["with_metadata"] / total * 100, 1),
                     "todo": row["todo_metadata"]},
        "fulltext": {"count": row["with_fulltext"], "pct": round(row["with_fulltext"] / total * 100, 1),
                     "todo": row["todo_fulltext"]},
        "by_scope": None,
    }


# ─── Résultat principal des PICO déjà extraits (key_finding / finding_direction) ──
# Les rédacteurs PICO demandent désormais le résultat principal de chaque article
# (api/pico_map) ; les PICO écrits avant n'en ont pas. Cette passe le complète, une fois
# par article, mis en cache sur la ligne (fusionné dans pico_json), pour TOUS les articles
# pertinents qui en manquent : jamais un échantillon. Le titre et le résumé en entrée, les
# deux champs en sortie, un appel court par article. Lancée par le pipeline après l'étape
# PICO, et par POST /user-scenarios/{id}/findings/run.

#: Aucun plafond par défaut, comme pour toute extraction. FINDINGS_MAX_ARTICLES > 0 en pose
#: un par passe (secours d'exploitation, un jour où le budget LLM doit être tenu).
FINDINGS_MAX_ARTICLES = _env_int("FINDINGS_MAX_ARTICLES", 0, 0)
_FINDINGS_WORKERS = 4
#: Réponses inutilisables (déterministes à température 0) après lesquelles un article n'est
#: plus redemandé. Une panne de l'API, elle, n'est comptée à personne.
_FINDINGS_MAX_ATTEMPTS = 3
_findings_lock = threading.Lock()
_findings_jobs: dict[str, dict[str, Any]] = {}

_FINDINGS_ATTEMPTS_SQL = ("CASE WHEN jsonb_typeof(d.pico_json->'findings_attempts') = 'number' "
                          "THEN (d.pico_json->>'findings_attempts')::int ELSE 0 END")
#: Un PICO sans résultat principal : à compléter tant qu'il n'a pas épuisé ses tentatives.
_LACKS_FINDING_SQL = ("d.pico_json IS NOT NULL AND (d.pico_json->>'finding_direction') IS NULL")


def _findings_rows(scenario_id: str) -> list[dict]:
    """Les articles pertinents dont le PICO attend son résultat principal, TOUS (la porte
    de toute l'application), dans l'ordre de pertinence du corpus : un plafond
    d'exploitation prendrait les plus proches de la question, pas les plus petits
    identifiants."""
    thr = _get_scenario_threshold(scenario_id)
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(f"""
            SELECT d.id, d.title, d.abstract
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {relevant_gate_sql('d', 'ars', ':thr')}
              AND {_LACKS_FINDING_SQL} AND {_FINDINGS_ATTEMPTS_SQL} < :max_attempts
            ORDER BY {relevance_order_sql('d', 'ars', ':thr')}
        """), {"sid": scenario_id, "thr": thr,
               "max_attempts": _FINDINGS_MAX_ATTEMPTS}).mappings().all()]


def findings_status(scenario_id: str) -> dict[str, Any]:
    """Où en est le scénario, compté sur TOUS ses articles pertinents, en une instruction."""
    thr = _get_scenario_threshold(scenario_id)
    with engine.connect() as conn:
        row = dict(conn.execute(text(f"""
            SELECT COUNT(*) AS n_relevant,
                   COUNT(*) FILTER (WHERE d.pico_json IS NOT NULL) AS n_with_pico,
                   COUNT(*) FILTER (WHERE (d.pico_json->>'finding_direction') IS NOT NULL)
                       AS n_with_finding,
                   COUNT(*) FILTER (WHERE {_LACKS_FINDING_SQL}
                                      AND {_FINDINGS_ATTEMPTS_SQL} < :max_attempts) AS n_pending,
                   COUNT(*) FILTER (WHERE {_LACKS_FINDING_SQL}
                                      AND {_FINDINGS_ATTEMPTS_SQL} >= :max_attempts) AS n_given_up
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {relevant_gate_sql('d', 'ars', ':thr')}
        """), {"sid": scenario_id, "thr": thr,
               "max_attempts": _FINDINGS_MAX_ATTEMPTS}).mappings().first() or {})
    with _findings_lock:
        job = dict(_findings_jobs.get(scenario_id) or {})
    counts = {k: int(row.get(k) or 0) for k in ("n_relevant", "n_with_pico", "n_with_finding",
                                                 "n_pending", "n_given_up")}
    return {
        "scenario_id": scenario_id,
        **counts,
        # Sans PICO, pas de résultat à compléter : c'est le travail des rédacteurs PICO, et
        # le digest compte ces articles comme « pas encore lus pour leur résultat ».
        "n_without_pico": counts["n_relevant"] - counts["n_with_pico"],
        "running": bool(job.get("running")),
        "job": {k: job.get(k, 0) for k in ("total", "done", "failed")} if job else None,
        "error": job.get("error"),
        "max_articles": FINDINGS_MAX_ARTICLES,
    }


def _findings_client():
    """Le client LLM de la passe, en un seul endroit que les tests remplacent."""
    from llm_usage import MeteredOpenAI as _OAI
    return _OAI(timeout=90.0)


def _count_finding_attempt(article_id: int) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(text(f"""
                UPDATE literature_document AS d
                SET pico_json = d.pico_json
                    || jsonb_build_object('findings_attempts', {_FINDINGS_ATTEMPTS_SQL} + 1)
                WHERE d.id = :id AND d.pico_json IS NOT NULL
            """), {"id": article_id})
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"findings attempts {article_id}: {e}")


def _fill_findings(scenario_id: str, max_articles: int = 0, lang_directive: str = "") -> dict[str, int]:
    """La passe elle-même, le scénario déjà réservé (`_claim_findings`)."""
    rows = _findings_rows(scenario_id)
    cap = int(max_articles or FINDINGS_MAX_ARTICLES or 0)
    if cap > 0:
        rows = rows[:cap]
    with _findings_lock:
        _findings_jobs[scenario_id].update({"total": len(rows), "done": 0, "failed": 0})
    if not rows:
        return {"total": 0, "done": 0, "failed": 0}
    client = _findings_client()

    def _work(row: dict) -> bool:
        aid = row["id"]
        try:
            body, label, source = (row.get("abstract") or "").strip(), "Abstract", "abstract"
            if len(body) < 50:
                # Sans résumé, les méthodes et les résultats du texte intégral.
                with engine.connect() as conn:
                    body = fulltext_excerpt(pico_fulltext_chunks(conn, aid))
                label, source = "Full text, methods and results (excerpt)", "fulltext"
            if not body:
                raise PicoUnreadable("ni résumé ni texte intégral à lire")
            key_finding, direction = extract_finding(
                client, row.get("title"), body, body_label=label,
                lang_directive=lang_directive, label=f"finding {aid}")
        except PicoUnreadable as e:
            logger.warning(f"finding article {aid}: {e}")
            _count_finding_attempt(aid)
            return False
        except Exception as e:                               # noqa: BLE001 - appel en échec : transitoire
            if _is_openai_quota_error(e):
                _trip_openai_cooldown()
            logger.warning(f"finding article {aid}: {e}")
            return False
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE literature_document
                    SET pico_json = pico_json || CAST(:f AS jsonb)
                    WHERE id = :id AND pico_json IS NOT NULL
                """), {"id": aid, "f": json.dumps({"key_finding": key_finding,
                                                    "finding_direction": direction,
                                                    "finding_source": source}, ensure_ascii=False)})
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"finding article {aid}: {e}")
            return False
        return True

    with ThreadPoolExecutor(max_workers=_FINDINGS_WORKERS) as ex:
        for ok in ex.map(_work, rows):
            with _findings_lock:
                _findings_jobs[scenario_id]["done" if ok else "failed"] += 1
    with _findings_lock:
        return {k: _findings_jobs[scenario_id][k] for k in ("total", "done", "failed")}


def _claim_findings(scenario_id: str) -> bool:
    """Une passe à la fois par scénario : deux passes simultanées paieraient deux fois les
    mêmes articles."""
    with _findings_lock:
        if _findings_jobs.get(scenario_id, {}).get("running"):
            return False
        _findings_jobs[scenario_id] = {"running": True, "started_at": time.time(),
                                       "total": 0, "done": 0, "failed": 0}
        return True


def _release_findings(scenario_id: str, error: str | None = None) -> None:
    with _findings_lock:
        job = _findings_jobs.setdefault(scenario_id, {})
        job["running"] = False
        if error:
            job["error"] = error[:300]


def run_findings_pass(scenario_id: str, max_articles: int = 0, lang_directive: str = "") -> dict[str, Any]:
    """La passe, SYNCHRONE : pour le pipeline, qui l'enchaîne après l'étape PICO."""
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "no_llm"}
    if not _claim_findings(scenario_id):
        return {"status": "running"}
    try:
        out = _fill_findings(scenario_id, max_articles, lang_directive)
    except Exception as e:
        _release_findings(scenario_id, str(e))
        raise
    _release_findings(scenario_id)
    return {"status": "done", **out}


def start_findings(scenario_id: str, max_articles: int = 0, lang_directive: str = "") -> dict[str, Any]:
    """La passe en arrière-plan, une à la fois par scénario."""
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "no_llm"}
    if not _claim_findings(scenario_id):
        return {"status": "running"}

    def _run():
        try:
            _fill_findings(scenario_id, max_articles, lang_directive)
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"findings {scenario_id}: {e}")
            _release_findings(scenario_id, str(e))
            return
        _release_findings(scenario_id)

    threading.Thread(target=_run, daemon=True, name=f"findings-{scenario_id}").start()
    return {"status": "started"}


@app.post("/user-scenarios/{scenario_id}/findings/run")
def run_scenario_findings(scenario_id: str, max_articles: int = 0, lang: Optional[str] = None,
                          _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Complète, en arrière-plan, le résultat principal (`key_finding`,
    `finding_direction`) de TOUS les articles pertinents dont le PICO n'en a pas. Aucun
    plafond sauf `max_articles` ou FINDINGS_MAX_ARTICLES. `lang` : langue de la phrase de
    résultat (anglais sans) ; la direction reste un identifiant anglais.
    Suivi : GET /user-scenarios/{id}/findings/status."""
    _get_user_scenario_or_404(scenario_id)
    directive = _llm_lang_directive(lang) if isinstance(lang, str) and lang.strip() else ""
    out = start_findings(scenario_id, max(0, int(max_articles or 0)), directive)
    status = findings_status(scenario_id)
    return {**out, "n_relevant": status["n_relevant"], "n_pending": status["n_pending"]}


@app.get("/user-scenarios/{scenario_id}/findings/status")
def get_scenario_findings_status(scenario_id: str) -> dict[str, Any]:
    _get_user_scenario_or_404(scenario_id)
    return findings_status(scenario_id)
