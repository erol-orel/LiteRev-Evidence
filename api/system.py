"""Health, LLM usage and filter options endpoints.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, Query
from sqlalchemy import text

import lexical_search as _lex
import llm_usage as _llm_usage

from .core import (
    RATE_LIMIT_EXPENSIVE_PER_MIN,
    RATE_LIMIT_GENERAL_PER_MIN,
    _process_stats,
    app,
    engine,
    logger,
    require_api_key,
)
from .schema_boot import _REQUIRED_TABLES, _SCHEMA_DDL_FAILURES

# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, Any]:
    """Santé du service - connexion À LA BASE *et* intégrité du schéma.

    `SELECT 1` seul mentait : sur une base incomplète, /health répondait « ok » pendant
    que /user-scenarios, /gesica/scenarios et /corpus/fulltext-stats renvoyaient 500 (le
    DDL de démarrage échoue ouvert par conception). On expose donc aussi l'état du
    schéma : `schema.ok` à false nomme les tables manquantes et le nombre d'instructions
    DDL écartées au démarrage.

    Le statut HTTP reste 200 même en mode dégradé : le smoke test de déploiement
    l'interroge, et faire échouer le déploiement sur une dégradation préexistante
    aggraverait la panne au lieu de la révéler. C'est `schema.ok` qu'il faut alerter."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        missing: list[str] = []
        try:
            present = {r[0] for r in conn.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))}
            missing = sorted(t for t in _REQUIRED_TABLES if t not in present)
        except Exception as _e:                      # ne jamais faire tomber /health
            logger.warning(f"health: contrôle du schéma indisponible: {_e}")
    schema_ok = not missing and not _SCHEMA_DDL_FAILURES
    out: dict[str, Any] = {
        "status": "ok", "database": "ok",
        "schema": {
            "ok": schema_ok,
            "missing_tables": missing,
            "ddl_failures": len(_SCHEMA_DDL_FAILURES),
        },
    }
    # Moteur de la recherche booléenne (plein texte une fois document_search
    # rempli, LIKE avant / en repli) : informatif, jamais bloquant.
    try:
        out["lexical_search"] = _lex.state()
    except Exception as _e:                          # noqa: BLE001
        out["lexical_search"] = {"error": str(_e)[:200]}
    # Mémoire / threads / uptime / pool DB du processus : un redémarrage récent
    # (uptime court) ou une mémoire proche de la limite se lisent ici.
    out["process"] = _process_stats()
    # Per-IP limits in force (RATE_LIMIT_*_PER_MIN): what a room sharing one IP gets.
    out["rate_limit"] = {"general_per_min": RATE_LIMIT_GENERAL_PER_MIN,
                         "expensive_per_min": RATE_LIMIT_EXPENSIVE_PER_MIN}
    # Which model is doing which job RIGHT NOW. The names are environment-overridable, so
    # the only way to know what a deployment is running is to ask it, and any measurement
    # of extraction quality is a measurement of these exact models.
    try:
        from llm_usage import models_in_use
        out["models"] = models_in_use()
    except Exception as _e:                          # noqa: BLE001
        out["models"] = {"error": str(_e)[:200]}
    # And in what SHAPE it is asked: the reasoning effort, whether the temperatures the
    # call sites ask for survive it, and any parameter the API has refused since the last
    # restart. `learned_repairs` not being empty means the model configuration is wrong
    # somewhere, which is otherwise visible only in the logs. POST /llm-selftest proves
    # the whole shape against the live API for one call per role.
    try:
        from llm_usage import request_policy
        out["llm_requests"] = request_policy()
    except Exception as _e:                          # noqa: BLE001
        out["llm_requests"] = {"error": str(_e)[:200]}
    if not schema_ok:
        # Visible dans la réponse, pas seulement dans les logs du serveur.
        out["schema"]["details"] = _SCHEMA_DDL_FAILURES[:10]
        # NB : `status` reste volontairement "ok" même ici. Ce n'est plus un aveu
        # d'impuissance : `schema.ok` EST désormais bloquant au déploiement (cf.
        # scripts/check_health.py, appelé par le smoke test de deploy.yml, activé après
        # confirmation que la production était saine - 39759fe : ok=true, 0 table
        # manquante, 0 DDL écartée).
        # La séparation est délibérée : `status` répond « le service tourne », et doit
        # rester vrai pour que le déploiement PORTANT LE CORRECTIF puisse aboutir ;
        # `schema.ok` répond « la base est complète », et c'est lui qui échoue le
        # déploiement. Les inverser rendrait une dégradation irréparable par déploiement.
        logger.warning(f"/health: schéma DÉGRADÉ - tables manquantes={missing}, "
                       f"DDL écartées={len(_SCHEMA_DDL_FAILURES)}")
    return out


@app.get("/llm-usage")
def get_llm_usage(hours: int = Query(24, ge=1, le=24 * 90),
                  _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Consommation OpenAI par usage et par modèle sur les `hours` dernières heures.

    LA question à laquelle l'application ne savait pas répondre : QUI dépense. Chaque
    ligne est un couple (fonction appelante:surface, modèle) - p. ex.
    `_background_enrichment_worker:chat` pour l'extraction PICO automatique, la plus
    grosse dépense potentielle (50 articles toutes les 30 s). Trié par tokens
    décroissants : la première ligne est celle à regarder.

    Protégé par la clé d'écriture : c'est de la donnée d'exploitation, pas du contenu."""
    return _llm_usage.summary(hours=hours)


@app.post("/llm-selftest")
def llm_selftest(_: None = Depends(require_api_key)) -> dict[str, Any]:
    """One real, minimal call per role: does this deployment's LLM configuration work?

    THE question a model switch leaves open, and the one nothing else here can answer.
    Every LLM call site in the application is wrapped in `except Exception` and degrades
    to "this feature is unavailable", so a model name or a parameter the API refuses
    produces no error page anywhere: PICO stops filling, briefs come back empty, the
    assistant apologises, /health stays green and the only trace is a line in the logs of
    whichever worker happened to run first.

    So this asks the API directly, in the shape the application actually sends (a token
    ceiling and a temperature, translated for the named model by
    `llm_usage.shape_chat_kwargs`), and reports per role what came back. `repairs_learned`
    is the interesting field: a parameter named there was refused by the live API, the
    call succeeded on the retry, and the mapping in `llm_usage._LEGACY_REQUEST_SHAPE`
    disagrees with reality for that model.

    About twenty tokens per role. Behind the write key because it spends money, and POST
    because it does."""
    from llm_usage import (LLMCallBlocked, MeteredOpenAI, model_for, models_in_use,
                           request_policy)

    def _repairs() -> dict:
        try:
            return request_policy().get("learned_repairs", {})
        except Exception:                            # noqa: BLE001
            return {}

    before = _repairs()
    key = os.getenv("OPENAI_API_KEY")
    roles: dict[str, Any] = {}

    def _attempt(role: str, call) -> dict[str, Any]:
        out: dict[str, Any] = {"model": model_for(role)}
        if not key:
            return {**out, "ok": False, "error": "OPENAI_API_KEY is not set"}
        try:
            resp = call(MeteredOpenAI(api_key=key, timeout=20.0, purpose="llm_selftest"))
        except LLMCallBlocked as exc:
            # Not a misconfiguration: the master switch or the daily budget said no.
            return {**out, "ok": False, "blocked": True, "error": str(exc)[:300]}
        except Exception as exc:                     # noqa: BLE001
            return {**out, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
        usage = getattr(resp, "usage", None)
        return {**out, "ok": True, "tokens": _llm_usage._usage_fields(usage)[2],
                "served_by": getattr(resp, "model", None), "response": resp}

    for role in ("bulk", "write", "chat"):
        def _chat(client, role=role):
            return client.chat.completions.create(
                model=model_for(role),
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                temperature=0,
                max_tokens=16,
            )
        got = _attempt(role, _chat)
        resp = got.pop("response", None)
        if resp is not None:
            try:
                got["answer"] = (resp.choices[0].message.content or "")[:80]
            except Exception as exc:                 # noqa: BLE001
                got["ok"] = False
                got["error"] = f"a response with no readable content: {exc}"[:300]
        roles[role] = got

    got = _attempt("embedding", lambda client: client.embeddings.create(
        model=model_for("embedding"), input="ok"))
    resp = got.pop("response", None)
    if resp is not None:
        try:
            # The dimension is the part that matters: document_chunk.embedding is
            # vector(1536), and a model of another width cannot be inserted into it.
            got["dimensions"] = len(resp.data[0].embedding)
        except Exception as exc:                     # noqa: BLE001
            got["ok"] = False
            got["error"] = f"a response with no readable vector: {exc}"[:300]
    roles["embedding"] = got

    after = _repairs()
    learned = {model: {p: r for p, r in params.items()
                       if before.get(model, {}).get(p) != r}
               for model, params in after.items()}
    return {
        "ok": all(r.get("ok") for r in roles.values()),
        "models": models_in_use(),
        "policy": request_policy(),
        "repairs_learned": {m: p for m, p in learned.items() if p},
        "roles": roles,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Filter options
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/filters-options")
def get_filter_options() -> dict[str, list[dict[str, Any]]]:
    fields = [
        ("source", "source"),
        ("source_type", "source_type"),
        ("disease_or_condition", "disease_or_condition"),
        ("scenario_type", "scenario_type"),
        ("geographic_scope", "geographic_scope"),
        ("evidence_category", "evidence_category"),
        ("year", "year"),
    ]
    out: dict[str, list[dict[str, Any]]] = {}

    # Normalisation des valeurs : fusionne les variantes avec tiret/underscore
    def _normalize_key(val: str) -> str:
        return val.lower().replace("-", "_").strip()

    def _make_label(val: str) -> str:
        return (
            str(val)
            .replace("_", " ")
            .replace("-", " ")
            .title()
            .replace("Covid 19", "COVID-19")
            .replace("Ems", "EMS")
            .replace("Ai", "AI")
            .replace("Uk", "UK")
            .replace("Usa", "USA")
        )

    # Pays/régions qui sont des combinaisons (contiennent virgule, 'and', chiffres+Countries)
    import re as _re
    def _is_singleton_geo(val: str) -> bool:
        v = str(val).strip()
        if _re.search(r'\d+\s+(Countries|Cities|Regions)', v, _re.IGNORECASE):
            return False
        if ',' in v or ' and ' in v.lower() or ' & ' in v:
            return False
        return True

    with engine.connect() as conn:
        for key, col in fields:
            extra_where = "AND year >= 1800 AND year <= EXTRACT(YEAR FROM CURRENT_DATE)::int" if key == "year" else ""
            rows = conn.execute(
                text(f"""
                    SELECT DISTINCT {col} AS value
                    FROM literature_document
                    WHERE {col} IS NOT NULL {extra_where}
                    ORDER BY {col}
                """)
            ).mappings().all()

            seen_normalized: dict[str, dict[str, str]] = {}  # normalized_key -> {value, label}
            for row in rows:
                value = row["value"]
                if value is None:
                    continue

                # Filtrer les scénarios usr-XXXX dans scenario_type
                if key == "scenario_type" and str(value).startswith("usr-"):
                    continue

                # Pour geographic_scope : ne garder que les pays/régions singletons
                if key == "geographic_scope" and not _is_singleton_geo(str(value)):
                    continue

                if key == "year":
                    label = str(value)
                    norm = str(value)
                else:
                    label = _make_label(str(value))
                    norm = _normalize_key(str(value))

                # Dédoublonnage par clé normalisée (ex: systematic-review == systematic_review)
                if norm not in seen_normalized:
                    seen_normalized[norm] = {"value": value, "label": label}

            out[key] = list(seen_normalized.values())
    return out
