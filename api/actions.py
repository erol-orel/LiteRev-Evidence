"""Recommended actions generated from the evidence.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

from typing import Any

from fastapi import Query
from sqlalchemy import text

from .core import app, engine, logger
from .documents import _llm_lang_directive
from .gesica import _get_scenario_name
from .relevance import _get_above_threshold_articles

# ─── ACTIONS RECOMMANDÉES (carte tableau de bord, généralisé aux user scenarios) ─
_ACTIONS_JOBS: dict[str, dict] = {}


def _generate_recommended_actions(scenario_id: str, lang: str | None = None) -> list[str]:
    """
    Génère 4-5 actions opérationnelles/cliniques concrètes à partir de l'évidence
    du scénario (PICO des articles au-dessus du seuil), façon « Actions
    recommandées » des cartes GESICA. Cache dans scenario_settings.
    """
    import json as _json
    from llm_usage import MeteredOpenAI as _OAI

    # Articles AVEC PICO d'abord (contexte pour 20 d'entre eux), sinon tous les
    # pertinents ; seuls les 20 premiers portent leur résumé/PICO.
    pico_articles = _get_above_threshold_articles(scenario_id, full_rows=20, require_pico=True)
    base = pico_articles or _get_above_threshold_articles(scenario_id, full_rows=20)
    if not base:
        return []

    scenario_name = _get_scenario_name(scenario_id)
    ctx = []
    for a in base[:20]:
        pj = a.get("pico_json") or {}
        ctx.append({
            "title": (a.get("title") or "")[:120],
            "O": pj.get("outcome", pj.get("O", "")),
            "key_finding": pj.get("key_finding", pj.get("conclusion", "")),
        })

    # Les actions engagent le corpus ENTIER, pas les 20 articles reproduits : le digest
    # (agrégats sur TOUS les articles pertinents) précède, le verbatim illustre.
    from .digest import corpus_digest, digest_coverage_note, digest_to_prompt
    _digest = corpus_digest(scenario_id)
    _digest_block = digest_to_prompt(_digest, max_chars=1800)
    _coverage = digest_coverage_note(_digest, len(ctx))
    _n_total = _digest.get("n_articles") or len(base)

    system = ("Tu es un expert en aide à la décision en santé et en santé publique. "
              "À partir d'une revue de littérature, tu proposes des ACTIONS opérationnelles "
              "concrètes, spécifiques et actionnables (pas de généralités). Pas de tiret cadratin (em dash)."
              ) + _llm_lang_directive(lang)
    user = (f"Scénario : \"{scenario_name}\"\n\n"
            + (f"{_digest_block}\n\n{_coverage}\n\n" if _digest_block else f"Basé sur {_n_total} articles.\n\n")
            + f"Articles reproduits ({len(ctx)} les mieux établis) :\n"
            + f"{_json.dumps(ctx, ensure_ascii=False)[:6000]}\n\n"
            "Génère un JSON {\"recommended_actions\": [\"action 1\", ...]} avec 4 à 5 actions "
            "concrètes déduites de l'évidence du corpus complet. Retourne UNIQUEMENT le JSON.")
    try:
        client = _OAI(timeout=90.0)
        resp = client.chat.completions.create(
            model="gpt-4.1", temperature=0.2, max_tokens=700,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        data = _json.loads(resp.choices[0].message.content)
        actions = [str(x) for x in (data.get("recommended_actions") or []) if str(x).strip()][:6]
    except Exception as e:
        logger.error(f"Génération actions {scenario_id}: {e}", exc_info=True)
        return []

    _lang_norm = (lang or "fr")[:2].lower()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO scenario_settings (scenario_id, recommended_actions_json, recommended_actions_lang, actions_generated_at, updated_at)
            VALUES (:sid, CAST(:a AS jsonb), :lng, NOW(), NOW())
            ON CONFLICT (scenario_id) DO UPDATE
            SET recommended_actions_json = CAST(:a AS jsonb), recommended_actions_lang = :lng,
                actions_generated_at = NOW(), updated_at = NOW()
        """), {"sid": scenario_id, "a": _json.dumps(actions), "lng": _lang_norm})
    return actions


def _maybe_generate_actions(scenario_id: str, lang: str | None = None) -> bool:
    """Lance la génération des actions en arrière-plan, une fois par (scénario, langue).
    La clé de job inclut la langue : un changement de langue relance la génération
    (au lieu du garde-fou « une seule fois » qui figeait la 1re langue)."""
    import threading
    _job_key = f"{scenario_id}:{(lang or 'fr')[:2].lower()}"
    if _ACTIONS_JOBS.get(_job_key, {}).get("status") in ("running", "done"):
        return False
    _ACTIONS_JOBS[_job_key] = {"status": "running"}

    def _run():
        try:
            n = _generate_recommended_actions(scenario_id, lang=lang)
            _ACTIONS_JOBS[_job_key] = {"status": "done", "count": len(n)}
        except Exception as e:
            _ACTIONS_JOBS[_job_key] = {"status": "error", "error": str(e)}
            logger.warning(f"Actions job {scenario_id}: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return True


@app.get("/scenarios/{scenario_id}/recommended-actions")
def get_recommended_actions(scenario_id: str, lang: str | None = Query(None)) -> dict[str, Any]:
    """Actions recommandées (cache) ; génère en arrière-plan au 1er appel si absentes."""
    _lang_norm = (lang or "fr")[:2].lower()
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT recommended_actions_json, recommended_actions_lang, actions_generated_at FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).mappings().first()
    # Ne servir le cache que s'il est DANS LA LANGUE demandée. Les actions anciennes
    # sans langue enregistrée (NULL) sont considérées françaises.
    if (row and isinstance(row["recommended_actions_json"], list) and row["recommended_actions_json"]
            and (row["recommended_actions_lang"] or "fr") == _lang_norm):
        return {"status": "ready", "actions": row["recommended_actions_json"],
                "generated_at": row["actions_generated_at"].isoformat() if row["actions_generated_at"] else None}
    started = _maybe_generate_actions(scenario_id, lang=lang)
    job = _ACTIONS_JOBS.get(f"{scenario_id}:{_lang_norm}", {})
    if job.get("status") == "error":
        return {"status": "error", "actions": [], "error": job.get("error")}
    return {"status": "generating" if (started or job.get("status") == "running") else "empty", "actions": []}
