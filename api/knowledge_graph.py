"""Semantic similarity knowledge graph of a scenario corpus.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
import re
import threading
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
    # manuellement ; jamais les exclus), PAS sur tout le corpus - même définition que
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
    """Précalcule + met en cache (DB) le knowledge graph aux paramètres par défaut, et la
    carte des concepts à partir de ce qui est déjà en base (l'annotation LLM des articles
    sans concepts est faite par le pipeline complet, qui suit)."""
    try:
        _save_viz_cache(scenario_id, "kg", _compute_user_kg(scenario_id))
    except Exception as _e:
        logger.warning(f"Précalcul KG {scenario_id}: {_e}")
    _precompute_concept_graph(scenario_id, extract=False)


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


# ─── CONCEPT GRAPH (carte des concepts du corpus) ─────────────────────────────
#
# Le graphe de similarité ci-dessus relie des ARTICLES : il répète l'onglet clustering
# sous une forme moins lisible (une boule de nœuds). Le graphe de concepts relie ce dont
# les articles PARLENT : pathogènes, vecteurs, hôtes, populations, expositions,
# interventions, issues (extraits du PICO et normalisés par le LLM, une fois par article,
# dans les deux langues), lieux (pays ISO2), devis d'étude et cadres (métadonnées).
# Un nœud = un concept typé, sa taille = son nombre d'articles ; une arête = deux
# concepts de types différents cités par un même article, son poids = le nombre de ces
# articles. Chaque nœud et chaque arête renvoie à ses articles. Le graphe signale aussi
# les triplets les plus documentés (sujet, exposition/intervention, issue), les lacunes
# (paires attendues sans aucun article) et les concepts portés par les articles de la
# dernière année du corpus.

CONCEPT_TYPES = ("pathogen", "vector", "host", "population", "exposure", "intervention",
                 "outcome", "method", "place", "design", "setting", "topic")
# Types que le LLM attribue (les autres viennent des champs structurés).
_LLM_CONCEPT_TYPES = ("pathogen", "vector", "host", "population", "exposure", "intervention",
                      "outcome", "method", "place")
CONCEPTS_VERSION = 1
# Plafond d'articles normalisés par le LLM en une passe (15 par appel → ≈ 100 appels).
# Aucun plafond par défaut : les concepts sont normalisés pour TOUS les articles
# pertinents, une seule fois par article (cache `concepts_json`), donc le coût est ponctuel
# et incrémental. CONCEPT_MAX_ARTICLES > 0 en pose un pour une passe (secours budgétaire).
CONCEPT_MAX_ARTICLES = int(os.getenv("CONCEPT_MAX_ARTICLES", "0") or 0)
_CONCEPT_BATCH = 15
_CONCEPT_WORKERS = 6

# Vocabulaire des devis d'étude (repli sans LLM et normalisation des champs libres).
# L'ordre compte : « systematic review » avant « review », « randomi » avant « cohort ».
_DESIGN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("systematic_review", ("systematic review", "meta-analy", "meta analy", "revue systématique",
                           "méta-analyse", "scoping review", "umbrella review")),
    ("rct", ("randomi", "rct", "essai contrôlé", "controlled trial")),
    ("cohort", ("cohort", "cohorte", "longitudinal", "prospective", "retrospective")),
    ("case_control", ("case-control", "case control", "cas-témoin", "cas témoin")),
    ("cross_sectional", ("cross-sectional", "cross sectional", "transversal", "survey", "enquête",
                         "prevalence study", "seroprevalence", "séroprévalence")),
    ("case_report", ("case report", "case series", "cas clinique", "série de cas")),
    ("modelling", ("model", "modél", "simulation", "forecast", "prédiction", "prediction",
                   "machine learning", "deep learning")),
    ("surveillance", ("surveillance", "outbreak", "épidémie", "epidemiological report", "notification")),
    ("laboratory", ("in vitro", "laborator", "experimental", "genomic", "phylogen", "sequencing",
                    "entomolog", "vector competence")),
    ("qualitative", ("qualitative", "interview", "focus group")),
    ("review", ("review", "revue", "narrative", "overview", "editorial", "commentary", "guideline",
                "recommandation", "recommendation")),
)
_DESIGN_LABELS = {
    "systematic_review": {"en": "systematic review / meta-analysis", "fr": "revue systématique / méta-analyse"},
    "rct": {"en": "randomised trial", "fr": "essai randomisé"},
    "cohort": {"en": "cohort study", "fr": "étude de cohorte"},
    "case_control": {"en": "case-control study", "fr": "étude cas-témoins"},
    "cross_sectional": {"en": "cross-sectional study", "fr": "étude transversale"},
    "case_report": {"en": "case report / series", "fr": "cas clinique / série de cas"},
    "modelling": {"en": "modelling study", "fr": "étude de modélisation"},
    "surveillance": {"en": "surveillance / outbreak report", "fr": "surveillance / rapport d'épidémie"},
    "laboratory": {"en": "laboratory / experimental study", "fr": "étude de laboratoire / expérimentale"},
    "qualitative": {"en": "qualitative study", "fr": "étude qualitative"},
    "review": {"en": "review / guidance", "fr": "revue / recommandation"},
}
_SETTING_LABELS = {
    "hospital": {"en": "hospital", "fr": "hôpital"},
    "prehospital": {"en": "prehospital", "fr": "préhospitalier"},
    "community": {"en": "community", "fr": "communauté"},
    "primary_care": {"en": "primary care", "fr": "soins primaires"},
    "icu": {"en": "intensive care", "fr": "soins intensifs"},
    "laboratory": {"en": "laboratory", "fr": "laboratoire"},
}
# Mots-clés MeSH « check tags » sans valeur pour une carte de concepts.
_TOPIC_STOP = {
    "humans", "human", "male", "female", "animals", "animal", "adult", "adults", "aged",
    "middle aged", "young adult", "aged, 80 and over", "child", "children", "child, preschool",
    "infant", "infant, newborn", "adolescent", "adolescents", "article", "review", "humain", "homme",
    "femme", "adulte", "enfant",
}


def _normalise_design(raw: str | None) -> str | None:
    """Ramène un devis en texte libre (PICO, métadonnées, colonne) à la petite liste ci-dessus."""
    s = (raw or "").strip().lower()
    if not s or s in ("unknown", "non précisé", "not specified", "n/a", "na", "none", "null", "other"):
        return None
    for key, needles in _DESIGN_RULES:
        if any(n in s for n in needles):
            return key
    return None


def _concept_key(label: str) -> str:
    """Clé de fusion d'un libellé : minuscules, espaces normalisés, ponctuation finale ôtée."""
    s = re.sub(r"\s+", " ", (label or "").strip().strip(".;,:")).lower()
    return s


def _iso2(code: Any) -> str | None:
    s = str(code or "").strip().upper()
    return s if re.fullmatch(r"[A-Z]{2}", s) and s not in ("NA", "XX", "ZZ") else None


def _article_concepts(row: dict) -> list[tuple[str, str, dict]]:
    """Concepts typés d'un article : (type, clé, libellés {en, fr}). Combine les concepts
    normalisés par le LLM (`concepts_json`), les champs structurés (pays, devis, cadre) et
    les mots-clés. Pur."""
    out: dict[tuple[str, str], dict] = {}

    def add(t: str, en: str | None, fr: str | None = None):
        if t not in CONCEPT_TYPES:
            return
        key = (en or "").strip() if t == "place" else _concept_key(en or "")   # a place IS its ISO2 code
        if not key or len(key) < 2 or len(key) > 60:
            return
        out.setdefault((t, key), {"en": (en or "").strip(), "fr": (fr or en or "").strip()})

    cj = row.get("concepts_json")
    if isinstance(cj, str):
        try:
            cj = json.loads(cj)
        except Exception:
            cj = None
    if isinstance(cj, dict):
        for c in cj.get("concepts") or []:
            if not isinstance(c, dict):
                continue
            t = str(c.get("t") or "").strip().lower()
            if t == "place":
                code = _iso2(c.get("en"))
                if code:
                    add("place", code, code)
                continue
            add(t, c.get("en"), c.get("fr"))

    # Champs structurés : lieu (ISO2), devis, cadre - indépendants du LLM.
    mj = row.get("metadata_json")
    if isinstance(mj, str):
        try:
            mj = json.loads(mj)
        except Exception:
            mj = None
    mj = mj if isinstance(mj, dict) else {}
    pj = row.get("pico_json")
    if isinstance(pj, str):
        try:
            pj = json.loads(pj)
        except Exception:
            pj = None
    pj = pj if isinstance(pj, dict) else {}

    for code in (row.get("country"), mj.get("country")):
        c2 = _iso2(code)
        if c2:
            add("place", c2, c2)
    design = (_normalise_design(pj.get("study_design")) or _normalise_design(mj.get("study_type"))
              or _normalise_design(row.get("study_design")))
    if design:
        add("design", _DESIGN_LABELS[design]["en"], _DESIGN_LABELS[design]["fr"])
    setting = str(mj.get("setting") or "").strip().lower().replace(" ", "_")
    if setting in _SETTING_LABELS:
        add("setting", _SETTING_LABELS[setting]["en"], _SETTING_LABELS[setting]["fr"])

    kw = row.get("keywords")
    if isinstance(kw, list):
        parts = [str(k) for k in kw]
    else:
        parts = re.split(r"[;,|]", str(kw or ""))
    n_topics = 0
    for p in parts:
        term = re.sub(r"\s+", " ", p.strip().strip("*").lower())
        if not term or term in _TOPIC_STOP or len(term) < 3 or len(term.split()) > 4:
            continue
        add("topic", term, term)
        n_topics += 1
        if n_topics >= 8:
            break
    return [(t, k, lab) for (t, k), lab in out.items()]


def _build_concept_graph(rows: list[dict], *, max_nodes: int = 60, min_edge: int | None = None,
                         n_total: int | None = None) -> dict[str, Any]:
    """Construit la carte des concepts à partir des lignes d'articles (pur, testé hors base).

    `rows` : dicts {id, title, year, quality, doi, pmid, country, study_design, pico_json,
    metadata_json, keywords, concepts_json, similarity}, déjà limités au sous-ensemble
    pertinent du scénario, triés par pertinence décroissante."""
    from collections import Counter, defaultdict

    n_articles = len(rows)
    n_total = n_articles if n_total is None else int(n_total)
    small = n_articles < 30
    if min_edge is None:
        min_edge = 1 if small else 2
    min_count = 1 if small else 2

    per_article: dict[int, set[tuple[str, str]]] = {}
    labels: dict[tuple[str, str], dict] = {}
    count: Counter = Counter()
    n_with_concepts = 0
    years = [int(r["year"]) for r in rows if r.get("year")]
    latest_year = max(years) if years else None
    for r in rows:
        cs = _article_concepts(r)
        if any(t in _LLM_CONCEPT_TYPES and t != "place" for t, _, _ in cs):
            n_with_concepts += 1
        keys = set()
        for t, k, lab in cs:
            keys.add((t, k))
            labels.setdefault((t, k), lab)
        per_article[int(r["id"])] = keys
        count.update(keys)

    # Sélection des nœuds : chaque type présent garde ses 3 meilleurs, le reste au rang global.
    ranked = [k for k, c in count.most_common() if c >= min_count]
    chosen: list[tuple[str, str]] = []
    per_type_taken: Counter = Counter()
    for k in ranked:
        if per_type_taken[k[0]] < 3:
            chosen.append(k)
            per_type_taken[k[0]] += 1
    for k in ranked:
        if len(chosen) >= max_nodes:
            break
        if k not in chosen:
            chosen.append(k)
    chosen = chosen[:max(max_nodes, len([k for k in chosen if per_type_taken[k[0]]]))]
    chosen_set = set(chosen)
    node_id = {k: i for i, k in enumerate(chosen)}

    # Articles par nœud (ordre de pertinence = ordre des lignes) et arêtes inter-types.
    node_articles: dict[int, list[int]] = defaultdict(list)
    node_new: Counter = Counter()
    edge_w: Counter = Counter()
    edge_articles: dict[tuple[int, int], list[int]] = defaultdict(list)
    triple_c: Counter = Counter()
    pair_c: Counter = Counter()      # co-occurrences toutes paires (pour les lacunes)
    subj_types = {"pathogen", "host", "population", "vector"}
    rel_types = {"exposure", "intervention", "vector"}
    for r in rows:
        aid = int(r["id"])
        ks = [k for k in per_article.get(aid, ()) if k in chosen_set]
        ids = sorted(node_id[k] for k in ks)
        is_new = bool(latest_year and r.get("year") and int(r["year"]) == latest_year)
        for i in ids:
            node_articles[i].append(aid)
            if is_new:
                node_new[i] += 1
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                pair_c[(i, j)] += 1
                if chosen[i][0] != chosen[j][0]:
                    edge_w[(i, j)] += 1
                    edge_articles[(i, j)].append(aid)
        subj = [node_id[k] for k in ks if k[0] in subj_types]
        rel = [node_id[k] for k in ks if k[0] in rel_types]
        obj = [node_id[k] for k in ks if k[0] == "outcome"]
        for s in subj:
            for x in rel:
                if x == s:
                    continue
                for o in obj:
                    triple_c[(s, x, o)] += 1

    nodes = []
    for k in chosen:
        i = node_id[k]
        arts = node_articles.get(i, [])
        nodes.append({
            "id": i, "type": k[0], "label": labels.get(k, {"en": k[1], "fr": k[1]}),
            "count": len(arts), "new_count": int(node_new.get(i, 0)), "articles": arts[:40],
        })
    edges = []
    for (i, j), w in edge_w.most_common():
        if w < min_edge:
            break
        edges.append({"source": i, "target": j, "weight": int(w), "articles": edge_articles[(i, j)][:20]})
        if len(edges) >= 200:
            break

    triples = [{"nodes": list(t), "count": int(c)} for t, c in triple_c.most_common(8)
               if c >= (1 if small else 2)]

    # Lacunes : sujets × issues et expositions/interventions × issues les plus documentés,
    # sans aucun article commun - classées par « attente » (produit des tailles).
    by_type: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        by_type[n["type"]].append(n)
    for t in by_type:
        by_type[t].sort(key=lambda n: -n["count"])
    gaps = []
    outcomes = by_type.get("outcome", [])[:6]
    lefts = [n for t in ("pathogen", "population", "host") for n in by_type.get(t, [])[:4]] + \
            [n for t in ("exposure", "intervention", "vector") for n in by_type.get(t, [])[:4]]
    for a in lefts:
        for o in outcomes:
            i, j = sorted((a["id"], o["id"]))
            if pair_c.get((i, j), 0) == 0 and a["count"] >= 2 and o["count"] >= 2:
                gaps.append({"nodes": [a["id"], o["id"]], "expected": a["count"] * o["count"]})
    gaps.sort(key=lambda g: -g["expected"])
    gaps = gaps[:8]

    referenced: set[int] = set()
    for n in nodes:
        referenced.update(n["articles"])
    for e in edges:
        referenced.update(e["articles"])
    articles = {}
    for r in rows:
        aid = int(r["id"])
        if aid in referenced:
            articles[str(aid)] = {
                "t": (r.get("title") or "")[:160], "y": r.get("year"),
                "q": round(float(r.get("quality") or 0), 2),
                "doi": r.get("doi"), "pmid": r.get("pmid"),
            }

    type_counts = Counter(n["type"] for n in nodes)
    return {
        "kind": "concepts", "version": CONCEPTS_VERSION,
        "n_articles": n_articles, "n_total": n_total,
        "n_with_concepts": n_with_concepts,
        "n_missing_concepts": sum(1 for r in rows if not r.get("concepts_json")),
        "source": "llm" if n_with_concepts else "structured",
        "latest_year": latest_year,
        "types": [{"type": t, "count": int(c)} for t, c in type_counts.most_common()],
        "nodes": nodes, "edges": edges, "triples": triples, "gaps": gaps,
        "articles": articles,
    }


_CONCEPT_ROWS_SQL = """
    SELECT d.id, d.title, d.year, d.quality_score AS quality, d.doi, d.pmid, d.country,
           d.study_design, d.pico_json, d.metadata_json, d.keywords, d.concepts_json, d.abstract,
           COALESCE(ars.similarity_score, 0) AS similarity
    FROM literature_document d
    JOIN article_scenarios ars ON ars.document_id = d.id
    WHERE ars.scenario_id = :sid
      AND d.is_duplicate IS NOT TRUE
      AND d.abstract IS NOT NULL
      AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
      AND (COALESCE(ars.screening_status, d.screening_status) = 'included'
           OR COALESCE(ars.similarity_score, 0) >= :thr)
    ORDER BY (COALESCE(ars.screening_status, d.screening_status) = 'included') DESC,
             ars.similarity_score DESC NULLS LAST, d.year DESC NULLS LAST, d.id
    LIMIT :cap
"""
# Aucun plafond par défaut : la carte compte les concepts de TOUS les articles pertinents
# (lecture de colonnes déjà extraites, sans LLM). > 0 pose un plafond d'exploitation.
CONCEPT_GRAPH_MAX_ARTICLES = int(os.getenv("CONCEPT_GRAPH_MAX_ARTICLES", "0") or 0)


def _concept_rows(scenario_id: str) -> tuple[list[dict], int]:
    thr = _get_scenario_threshold(scenario_id)
    _cap = CONCEPT_GRAPH_MAX_ARTICLES if CONCEPT_GRAPH_MAX_ARTICLES > 0 else None
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(text(_CONCEPT_ROWS_SQL),
                                              {"sid": scenario_id, "thr": thr,
                                               "cap": _cap}).mappings().all()]
        n_total = conn.execute(text(
            "SELECT COUNT(*) FROM literature_document d JOIN article_scenarios ars ON ars.document_id = d.id "
            "WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE AND d.abstract IS NOT NULL "
            "AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded' "
            "AND (COALESCE(ars.screening_status, d.screening_status) = 'included' "
            "     OR COALESCE(ars.similarity_score, 0) >= :thr)"
        ), {"sid": scenario_id, "thr": thr}).scalar() or 0
    return rows, int(n_total)


def _compute_concept_graph(scenario_id: str) -> dict[str, Any]:
    """Carte des concepts d'un scénario à partir de ce qui est déjà en base (rapide, sans LLM)."""
    rows, n_total = _concept_rows(scenario_id)
    payload = _build_concept_graph(rows, n_total=n_total)
    payload["scenario_id"] = scenario_id
    return payload


_CONCEPT_SYSTEM = (
    "You index epidemiological and clinical literature for a knowledge graph. For EACH article "
    "given, return 3 to 8 short canonical concepts (1 to 4 words, singular, lowercase except proper "
    "nouns and acronyms, in English) typed among: pathogen, vector, host, population, exposure, "
    "intervention, outcome, method, place. Use the most common scientific name (e.g. 'Aedes "
    "albopictus', never 'Ae. albopictus'; 'chikungunya virus', not 'CHIKV infection agent'). "
    "'place' is the ISO 3166-1 alpha-2 code of the country studied (e.g. 'IT'), omit it when the "
    "article has no specific country. Give the French label too (for a place, repeat the code). "
    "Return ONLY JSON: {\"articles\": [{\"id\": <id>, \"concepts\": [{\"t\": <type>, \"en\": <label>, "
    "\"fr\": <libellé>}]}]}, one entry per input article, same ids."
)


def _llm_concepts_for_batch(client, batch: list[dict]) -> dict[int, list[dict]]:
    """Un appel LLM pour un lot d'articles → {id: [concepts]}. Robuste : lot perdu = {}."""
    items = []
    for r in batch:
        pj = r.get("pico_json") if isinstance(r.get("pico_json"), dict) else {}
        items.append({
            "id": int(r["id"]),
            "title": (r.get("title") or "")[:300],
            "P": str(pj.get("P") or "")[:300], "I": str(pj.get("I") or "")[:300],
            "O": str(pj.get("O") or "")[:300], "design": str(pj.get("study_design") or "")[:80],
            "abstract": "" if pj else (r.get("abstract") or "")[:900],
        })
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": _CONCEPT_SYSTEM},
                      {"role": "user", "content": json.dumps({"articles": items}, ensure_ascii=False)}],
            temperature=0, seed=42, max_tokens=4000,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.warning(f"concept extraction batch: {e}")
        return {}
    out: dict[int, list[dict]] = {}
    for a in (data.get("articles") or []) if isinstance(data, dict) else []:
        try:
            aid = int(a.get("id"))
        except Exception:
            continue
        clean = []
        for c in a.get("concepts") or []:
            if not isinstance(c, dict):
                continue
            t = str(c.get("t") or "").strip().lower()
            en = str(c.get("en") or "").strip()
            fr = str(c.get("fr") or en).strip()
            if t in _LLM_CONCEPT_TYPES and en:
                clean.append({"t": t, "en": en[:80], "fr": fr[:80]})
        out[aid] = clean[:8]
    return out


def _extract_concepts_for_scenario(scenario_id: str, max_articles: int | None = None) -> int:
    """Normalise par le LLM les concepts des articles pertinents qui n'en ont pas encore
    (`concepts_json`), une fois pour toutes - les autres scénarios les réutilisent.
    Renvoie le nombre d'articles annotés. Sans clé OpenAI : 0, sans erreur."""
    if not os.getenv("OPENAI_API_KEY"):
        return 0
    from concurrent.futures import ThreadPoolExecutor
    from llm_usage import MeteredOpenAI as _OAI
    rows, _ = _concept_rows(scenario_id)
    todo = [r for r in rows if not r.get("concepts_json")
            and (isinstance(r.get("pico_json"), dict) or (r.get("abstract") and len(r["abstract"]) > 80))]
    _cap = int(max_articles or CONCEPT_MAX_ARTICLES or 0)
    if _cap > 0:
        todo = todo[:_cap]
    if not todo:
        return 0
    client = _OAI(timeout=120.0)
    batches = [todo[i:i + _CONCEPT_BATCH] for i in range(0, len(todo), _CONCEPT_BATCH)]
    done = 0

    def _work(batch):
        res = _llm_concepts_for_batch(client, batch)
        n = 0
        if res:
            try:
                with engine.begin() as conn:
                    for aid, concepts in res.items():
                        conn.execute(text(
                            "UPDATE literature_document SET concepts_json = CAST(:c AS jsonb) WHERE id = :id"
                        ), {"c": json.dumps({"v": CONCEPTS_VERSION, "concepts": concepts}, ensure_ascii=False),
                            "id": aid})
                        n += 1
            except Exception as e:
                logger.warning(f"concepts_json write: {e}")
        return n

    with ThreadPoolExecutor(max_workers=_CONCEPT_WORKERS) as ex:
        for n in ex.map(_work, batches):
            done += n
    logger.info(f"Concepts {scenario_id}: {done}/{len(todo)} articles annotés ({len(batches)} appels).")
    return done


_concept_jobs_lock = threading.Lock()
_concept_jobs_running: set[str] = set()


def _precompute_concept_graph(scenario_id: str, extract: bool = True) -> dict[str, Any] | None:
    """Carte des concepts → cache DB. `extract` : normaliser d'abord par le LLM les articles
    sans concepts (pipeline) ; False = ce qui est en base seulement (post-recherche, rapide)."""
    try:
        if extract:
            _extract_concepts_for_scenario(scenario_id)
        payload = _compute_concept_graph(scenario_id)
        _save_viz_cache(scenario_id, "concepts", payload)
        return payload
    except Exception as _e:
        logger.warning(f"Précalcul concept graph {scenario_id}: {_e}")
        return None


def _start_concept_enrichment(scenario_id: str) -> bool:
    """Lance en arrière-plan (une fois par scénario à la fois) l'annotation LLM puis la
    reconstruction du cache. Renvoie True si un job a démarré ou tourne déjà."""
    if not os.getenv("OPENAI_API_KEY"):
        return False
    with _concept_jobs_lock:
        if scenario_id in _concept_jobs_running:
            return True
        _concept_jobs_running.add(scenario_id)

    def _run():
        try:
            _precompute_concept_graph(scenario_id, extract=True)
        finally:
            with _concept_jobs_lock:
                _concept_jobs_running.discard(scenario_id)

    threading.Thread(target=_run, daemon=True, name=f"concepts-{scenario_id}").start()
    return True


def get_user_scenario_concept_graph_payload(scenario_id: str, refresh: bool = False) -> dict[str, Any]:
    """Cache DB d'abord (précalculé par le pipeline et après chaque recherche) ; sinon calcul
    rapide sur ce qui est en base, et annotation LLM des articles manquants en arrière-plan
    (`enriching` = true → l'interface re-lit le graphe quand c'est fini)."""
    _get_user_scenario_or_404(scenario_id)
    if not refresh:
        cached = _load_viz_cache(scenario_id, "concepts")
        if cached and cached.get("kind") == "concepts" and cached.get("version") == CONCEPTS_VERSION:
            cached["enriching"] = scenario_id in _concept_jobs_running
            if cached["enriching"] is False and cached.get("n_missing_concepts", 0) > 0 \
                    and cached.get("n_articles", 0) > 0 and os.getenv("OPENAI_API_KEY"):
                cached["enriching"] = _start_concept_enrichment(scenario_id)
            return cached
    payload = _compute_concept_graph(scenario_id)
    _save_viz_cache(scenario_id, "concepts", payload)
    payload["enriching"] = False
    if payload.get("n_missing_concepts", 0) > 0 and payload.get("n_articles", 0) > 0:
        payload["enriching"] = _start_concept_enrichment(scenario_id)
    return payload


@app.get("/user-scenarios/{scenario_id}/concept-graph")
def get_user_scenario_concept_graph(scenario_id: str, refresh: bool = False) -> dict[str, Any]:
    """Carte des concepts du corpus pertinent (voir le commentaire de section)."""
    return get_user_scenario_concept_graph_payload(scenario_id, refresh)
