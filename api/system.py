"""Health, LLM usage and filter options endpoints.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

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
