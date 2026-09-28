"""Scenario lookups shared by every domain: existence, threshold, display name.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from sqlalchemy import text

from .core import engine

# ── Helpers internes ──────────────────────────────────────────────────────────

def _get_user_scenario_or_404(scenario_id: str) -> dict[str, Any]:
    """Retourne la ligne user_scenarios ou lève 404."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, name, query, mode, filters, result_count, pinned, folder_id, created_at, updated_at,
                   search_strategy, populate_status, pipeline_status, pipeline_step,
                   pipeline_progress, pipeline_started_at, article_count, is_system,
                   sub_queries, combinator
            FROM user_scenarios WHERE id = :id
        """), {"id": scenario_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Scénario utilisateur '{scenario_id}' non trouvé")
    return dict(row)

DEFAULT_SIMILARITY_THRESHOLD = 0.45


def _get_scenario_threshold(scenario_id: str) -> float:
    """Retourne le seuil de similarité configuré pour ce scénario."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT similarity_threshold FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    return float(row["similarity_threshold"]) if row and row["similarity_threshold"] is not None else DEFAULT_SIMILARITY_THRESHOLD


# ── La porte de pertinence, écrite UNE fois ──────────────────────────────────
# « Les articles pertinents d'un scénario » veut dire : jamais un doublon, jamais un
# article qu'un relecteur a exclu, et sinon inclus à la main OU au-dessus du seuil. Un
# article sans score compte pour 0, donc il reste dehors tant que le seuil n'est pas nul.
#
# Cette condition avait été recopiée à la main dans chaque module et les copies ont
# divergé : celle du RAG (`/ask/stream/filtered`) avait perdu l'exclusion des doublons ET
# celle des articles exclus, si bien que l'assistant pouvait citer un article qu'un
# relecteur venait d'écarter, pendant que le compteur affiché sous la réponse, lui,
# comptait le bon sous-ensemble. Une fonction, un seul endroit à corriger.
def relevant_gate_sql(doc: str = "d", link: str = "ars", thr: str = ":thr") -> str:
    """Le prédicat SQL du sous-ensemble pertinent, à mettre dans un WHERE.

    `doc` et `link` sont les alias de literature_document et article_scenarios ; `thr` le
    paramètre lié qui porte le seuil. Pur : aucune connexion, testable hors base."""
    status = f"COALESCE({link}.screening_status, {doc}.screening_status)"
    return (f"{doc}.is_duplicate IS NOT TRUE"
            f" AND {status} IS DISTINCT FROM 'excluded'"
            f" AND ({status} = 'included' OR COALESCE({link}.similarity_score, 0) >= {thr})")


# ── Invalidation des artefacts calculés sur le corpus pertinent ───────────────
# Clustering, réseau de similarité, carte des concepts et actions recommandées sont
# tous des FONCTIONS du sous-ensemble pertinent : ils périment dès que ce sous-ensemble
# bouge, c'est-à-dire quand le seuil change ou quand le corpus gagne des articles. Le
# brief et les variables, eux, s'invalident seuls (leur empreinte porte le seuil ET les
# identifiants des articles).
#
# UNE seule liste, parce que deux requêtes à tenir en phase ont déjà divergé : le seuil
# et la living review nettoyaient les trois visualisations mais oubliaient les actions
# recommandées, qui restaient servies indéfiniment alors qu'elles décrivaient le corpus
# précédent.
CORPUS_DERIVED_CACHE_RESET = """
    clustering_json = NULL, clustering_generated_at = NULL,
    knowledge_graph_json = NULL, kg_generated_at = NULL,
    concept_graph_json = NULL, concept_graph_generated_at = NULL,
    recommended_actions_json = NULL, recommended_actions_lang = NULL,
    actions_generated_at = NULL
"""
