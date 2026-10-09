"""Batch LLM enrichment: PICO, metadata, full text.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import Depends, HTTPException
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .scenario_store import SCOPES, scenario_scope_sql
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

    system_prompt = (
        "You are a systematic review expert. "
        "Extract PICO elements and return ONLY valid JSON:\n"
        '{"P":"Population","I":"Intervention","C":"Comparator or Not specified",'
        '"O":"Outcome(s)",'
        '"study_design":"UN SEUL de: Randomized controlled trial | Clinical trial | '
        'Non-randomized trial | Systematic review | Meta-analysis | Cohort study | '
        'Case-control study | Cross-sectional study | Surveillance | Case report | '
        'Modelling study | Qualitative study | Guideline | Narrative review | '
        'Preclinical | Not stated",'
        '# Pour une revue, PRÉCISE ce qu\'elle inclut quand l\'abstract le dit '
        '(ex. "Systematic review of cohort studies") : le niveau de preuve en dépend.'
        ''
        '"pico_confidence":0.0-1.0,"pico_notes":""}\n'
        "Be concise (max 2 sentences per field). Return ONLY the JSON."
    )

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
            # Appel LLM isolé : une erreur d'API (quota/réseau) est TRANSITOIRE
            # → on ne compte PAS de tentative (réessai quand l'API est saine).
            try:
                response = _client.chat.completions.create(
                    model=_model("bulk"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract[:3000]}"},
                    ],
                    temperature=0,
                    seed=42,
                    max_tokens=800,  # 400 tronquait le JSON des articles verbeux → JSON invalide
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.warning(f"PICO batch API error article {article_id}: {e}")
                errors += 1
                continue  # transitoire - ne PAS consommer une tentative
            # On a une RÉPONSE → on COMPTE la tentative quoi qu'il arrive (borne le
            # token-bleed : une sortie déterministe malformée ne sera pas ré-extraite
            # à l'infini). Remplissage tolérant des clés plutôt que rejet en boucle.
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
                    """), {
                        "pico": json.dumps(pico),
                        "ts": datetime.now(timezone.utc),
                        "article_id": article_id,
                    })
                extracted += 1
            except Exception as e:
                logger.warning(f"PICO batch parse error article {article_id}: {e}")
                # Réponse reçue mais JSON invalide → COMPTE la tentative (borne le token-bleed).
                try:
                    with engine.begin() as conn:
                        conn.execute(text(
                            "UPDATE literature_document SET pico_attempts = COALESCE(pico_attempts, 0) + 1 WHERE id = :aid"
                        ), {"aid": article_id})
                except Exception:
                    pass
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
    Enrichit les métadonnées (type d'étude, année, journal) via LLM pour un lot d'articles.

    `scope` : `all` pour tout le scénario, `relevant` pour son seul sous-ensemble
    pertinent.
    """
    scope = _scope(scope)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=503, detail="Clé OpenAI non configurée")

    with engine.connect() as conn:
        if scenario_id:
            rows = conn.execute(text(f"""
                SELECT ld.id, ld.title, ld.abstract, ld.source, ld.year
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
                SELECT id, title, abstract, source, year
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
        "Return ONLY the JSON."
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
                response = _client.chat.completions.create(
                    model=_model("bulk"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract[:2000]}"},
                    ],
                    temperature=0.1,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                metadata = json.loads(response.choices[0].message.content)
                metadata["metadata_confidence"] = float(metadata.get("metadata_confidence", 0.5))
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE literature_document
                        SET metadata_json = CAST(:meta AS jsonb)
                        WHERE id = :article_id
                    """), {
                        "meta": json.dumps(metadata),
                        "article_id": article_id,
                    })
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
