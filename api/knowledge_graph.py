"""Semantic similarity knowledge graph of a scenario corpus.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text

from .core import app, engine, logger
from .scenario_store import _get_scenario_threshold, _get_user_scenario_or_404
from .clustering import _load_viz_cache, _save_viz_cache

# ─── KNOWLEDGE GRAPH (réseau de similarité sémantique) ───────────────────────

# Mots vides (EN + FR + remplissage scientifique) pour étiqueter les communautés
# thématiques à partir des titres d'articles.
_KG_STOPWORDS: set[str] = {
    # anglais courant
    "the", "and", "for", "with", "from", "this", "that", "study", "studies",
    "using", "based", "between", "among", "during", "after", "before", "into",
    "their", "these", "those", "which", "while", "about", "versus", "over",
    "analysis", "review", "systematic", "meta", "trial", "trials", "randomized",
    "randomised", "controlled", "results", "methods", "patients", "patient",
    "outcomes", "outcome", "associated", "association", "effect", "effects",
    "evaluation", "assessment", "comparison", "clinical", "data", "report",
    "case", "cases", "cohort", "prospective", "retrospective", "evidence",
    # français courant
    "les", "des", "une", "dans", "pour", "avec", "sur", "par", "aux", "leur",
    "étude", "étude", "analyse", "revue", "résultats", "méthode", "méthodes",
    "patients", "patient", "effet", "effets", "entre", "chez", "selon", "lors",
}


def _kg_cluster_label(titles: list[str], top_k: int = 3) -> str:
    """Étiquette thématique d'une communauté : termes les plus fréquents des titres."""
    from collections import Counter
    cnt: Counter = Counter()
    for t in titles:
        for tok in re.findall(r"[a-zàâäéèêëïîôöùûüç]{4,}", (t or "").lower()):
            if tok not in _KG_STOPWORDS:
                cnt[tok] += 1
    return ", ".join(w for w, _ in cnt.most_common(top_k))


def _build_knowledge_graph(
    scenario_id: str,
    rows: list,
    min_similarity: float,
    n_total: int,
) -> dict[str, Any]:
    """
    Construit un graphe de connaissance à partir d'articles + embeddings.

    Nœuds = articles ; taille = centralité (degré) ; couleur = communauté thématique.
    Arêtes = paires d'articles dont la similarité cosinus des embeddings ≥ min_similarity.
    Communautés = détection greedy ; chacune reçoit une étiquette de mots-clés (titres).
    `n_total` = nombre total d'articles éligibles (pour signaler un éventuel sous-ensemble).
    """
    if not rows:
        return {"nodes": [], "edges": [], "clusters": [], "n_total": n_total}

    import numpy as np

    nodes_data = []
    for r in rows:
        try:
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", r["emb_str"])
            emb = np.array([float(x) for x in nums], dtype=np.float32)
            if len(emb) > 0:
                nodes_data.append({
                    "id": r["id"],
                    "title": r["title"] or "",
                    "year": r["year"],
                    "journal": r["journal"],
                    "design": r["design"],
                    "quality": float(r["quality_score"] or 0),
                    "emb": emb,
                })
        except Exception:
            continue

    if not nodes_data:
        return {"nodes": [], "edges": [], "clusters": [], "n_total": n_total}

    embeddings = np.array([n["emb"] for n in nodes_data])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms
    sim_matrix = embeddings_norm @ embeddings_norm.T

    n = len(nodes_data)
    # Arêtes (vectorisé : on ne garde que le triangle supérieur au-dessus du seuil)
    edges = []
    iu, ju = np.triu_indices(n, k=1)
    mask = sim_matrix[iu, ju] >= min_similarity
    for i, j, w in zip(iu[mask], ju[mask], sim_matrix[iu, ju][mask]):
        edges.append({
            "source": nodes_data[int(i)]["id"],
            "target": nodes_data[int(j)]["id"],
            "weight": round(float(w), 3),
        })

    # Détection de communautés greedy (lien fort ≥ 0.5)
    cluster_ids = [-1] * n
    cluster_counter = 0
    for i in range(n):
        if cluster_ids[i] == -1:
            cluster_ids[i] = cluster_counter
            strong = np.where(sim_matrix[i] >= 0.5)[0]
            for j in strong:
                if cluster_ids[j] == -1:
                    cluster_ids[j] = cluster_counter
            cluster_counter += 1

    # Degré (centralité) par nœud
    degree: dict[int, int] = {nd["id"]: 0 for nd in nodes_data}
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    nodes = []
    for idx, nd in enumerate(nodes_data):
        title = nd["title"]
        nodes.append({
            "id": nd["id"],
            "title": title[:80] + ("..." if len(title) > 80 else ""),
            "year": nd["year"],
            "journal": nd["journal"],
            "design": nd["design"],
            "quality": nd["quality"],
            "cluster": cluster_ids[idx],
            "degree": degree[nd["id"]],
        })

    # Résumé des communautés + étiquette thématique
    from collections import defaultdict
    members_map: dict[int, list] = defaultdict(list)
    titles_map: dict[int, list] = defaultdict(list)
    for idx, nd in enumerate(nodes_data):
        cid = cluster_ids[idx]
        members_map[cid].append(nodes[idx])
        titles_map[cid].append(nd["title"])

    clusters = []
    for cid, members in sorted(members_map.items(), key=lambda x: -len(x[1])):
        label = _kg_cluster_label(titles_map[cid])
        clusters.append({
            "id": cid,
            "size": len(members),
            "label": label,
            "years": sorted(set(m["year"] for m in members if m["year"])),
            "designs": list(set(m["design"] for m in members if m["design"] and m["design"] != "unknown")),
            "top_articles": [m["title"] for m in sorted(members, key=lambda x: -x["quality"])[:3]],
        })

    return {
        "scenario_id": scenario_id,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_clusters": len(clusters),
        "n_total": n_total,
        "min_similarity": min_similarity,
        "nodes": nodes,
        "edges": edges,
        "clusters": clusters,
    }


# SQL partagé : sélectionne un embedding par article, priorise les meilleurs articles
_KG_NODE_SQL = """
    SELECT * FROM (
        SELECT DISTINCT ON (d.id)
            d.id, d.title, d.year, d.journal, d.study_design, d.quality_score,
            c.embedding::text AS emb_str,
            COALESCE((d.pico_json->>'study_design'), d.study_design, 'unknown') AS design
        FROM literature_document d
        {join}
        WHERE {where}
          AND d.is_duplicate IS NOT TRUE
          AND c.embedding IS NOT NULL
          AND d.abstract IS NOT NULL
        ORDER BY d.id, (c.chunk_type = 'title_abstract') DESC, c.id
    ) sub
    ORDER BY quality_score DESC NULLS LAST, year DESC NULLS LAST
    LIMIT :max_nodes
"""


@app.get("/user-scenarios/{scenario_id}/knowledge-graph")
def _compute_user_kg(scenario_id: str, max_nodes: int = 400, min_similarity: float = 0.35) -> dict[str, Any]:
    """Calcul du knowledge graph d'un scénario utilisateur (un seul endroit, réutilisé
    par l'endpoint ET le précalcul)."""
    # Le clustering porte sur le SOUS-ENSEMBLE PERTINENT (≥ seuil sémantique OU inclus
    # manuellement ; jamais les exclus), PAS sur tout le corpus — même définition que
    # corpus_above et l'Assistant RAG. Sinon les communautés étaient diluées par des
    # centaines d'articles hors-sujet ramenés par la fédération.
    _thr = _get_scenario_threshold(scenario_id)
    _relevant = (" AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'"
                 " AND (COALESCE(ars.screening_status, d.screening_status) = 'included'"
                 "      OR COALESCE(ars.similarity_score, 0) >= :thr)")
    sql = _KG_NODE_SQL.format(
        join=("JOIN article_scenarios ars ON ars.document_id = d.id"
              " JOIN document_chunk c ON c.document_id = d.id"),
        where="ars.scenario_id = :sid" + _relevant,
    )
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"sid": scenario_id, "max_nodes": max_nodes, "thr": _thr}).mappings().all()
        n_total = conn.execute(text("""
            SELECT COUNT(*) FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid
              AND d.is_duplicate IS NOT TRUE AND d.abstract IS NOT NULL
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
              AND (COALESCE(ars.screening_status, d.screening_status) = 'included'
                   OR COALESCE(ars.similarity_score, 0) >= :thr)
              AND EXISTS (SELECT 1 FROM document_chunk c
                          WHERE c.document_id = d.id AND c.embedding IS NOT NULL)
        """), {"sid": scenario_id, "thr": _thr}).scalar() or 0
    return _build_knowledge_graph(scenario_id, rows, min_similarity, int(n_total))


def _precompute_user_kg(scenario_id: str) -> None:
    """Précalcule + met en cache (DB) le knowledge graph aux paramètres par défaut."""
    try:
        _save_viz_cache(scenario_id, "kg", _compute_user_kg(scenario_id))
    except Exception as _e:
        logger.warning(f"Précalcul KG {scenario_id}: {_e}")


def get_user_scenario_knowledge_graph(
    scenario_id: str,
    max_nodes: int = 400,
    min_similarity: float = 0.35,
) -> dict[str, Any]:
    """Graphe de connaissance d'un scénario utilisateur (réseau de similarité sémantique)."""
    _get_user_scenario_or_404(scenario_id)
    # Cache DB durable aux paramètres par défaut (précalculé par le pipeline/populate).
    _default = (max_nodes == 400 and abs(min_similarity - 0.35) < 1e-6)
    if _default:
        _db = _load_viz_cache(scenario_id, "kg")
        if _db:
            return _db
    kg = _compute_user_kg(scenario_id, max_nodes, min_similarity)
    if _default:
        _save_viz_cache(scenario_id, "kg", kg)
    return kg
