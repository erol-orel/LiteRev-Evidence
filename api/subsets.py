"""Narrowing a scenario's corpus by cluster or by concept, written where the app reads.

A reviewer who says "keep only these two clusters" is doing screening: deciding the rest
is out of scope. The app already has exactly one mechanism for that, `screening_status =
'excluded'` on the scenario's OWN link row, and everything reads through it: the relevance
gate, the PRISMA counts, every extraction, the exports, the threshold curve. So this
module writes THERE instead of inventing a second notion of "the corpus" that the brief
and the PRISMA would know nothing about.

Two consequences, and both are the point:
  - the narrowing appears in PRISMA among the excluded, with the reason that produced it,
    so a methods section can state what was dropped and why;
  - it composes with the threshold for free, because the threshold curve and the gate both
    already skip excluded articles;
  - it is undone by clearing the same rows, and ONLY the ones this module wrote (a
    reviewer's own per-article exclusions are never touched).

The trap it refuses to fall into: the clustering is a PROJECTION capped at
CLUSTER_MAX_DOCS, and the concept map only covers articles whose concepts were extracted
and which have an abstract. On a corpus of 8,000 relevant articles, "keep clusters 3 and
7" would otherwise throw out the 5,000 the clustering never looked at, silently. An
article that no dimension of the selection can judge is therefore KEPT, counted on its own
line, and excluded only when the caller asks for it in so many words.
"""
from __future__ import annotations

import re
from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .scenario_store import (
    CORPUS_DERIVED_CACHE_RESET,
    _get_scenario_threshold,
    _get_user_scenario_or_404,
    relevant_gate_sql,
)

# Ce préfixe est la SIGNATURE des exclusions posées ici : l'annulation ne défait que
# celles-là. Sans lui, « annuler le sous-corpus » effacerait aussi les exclusions qu'un
# relecteur a posées une par une, ce qui est exactement le travail qu'on ne peut pas
# refaire.
SUBSET_REASON_PREFIX = "scope:"


# ─── La décision, pure et testable hors base ────────────────────────────────
def plan_subset(
    relevant: list[int],
    *,
    cluster_of: dict[int, int] | None = None,
    clusters_wanted: set[int] | None = None,
    concept_match: set[int] | None = None,
    concept_known: set[int] | None = None,
    design_of: dict[int, str] | None = None,
    designs_wanted: set[str] | None = None,
    level_of: dict[int, str] | None = None,
    levels_wanted: set[str] | None = None,
    combine: str = "all",
    unassigned: str = "keep",
) -> dict[str, Any]:
    """Qui reste, qui sort, et qui n'a pas pu être jugé. PUR.

    Une dimension vaut `None` quand la sélection ne s'en sert pas. Pour chaque article et
    chaque dimension employée, le verdict est `in`, `out`, ou `unknown` (l'article est
    hors de la projection : jamais clusterisé, ou sans concepts extraits).

    `combine='all'` garde l'article si TOUTES les dimensions qui savent le juger disent
    `in` ; `combine='any'` s'il y en a au moins une. Dans les deux cas une dimension qui ne
    sait pas ne vote pas : elle ne peut ni sauver ni condamner. Un article qu'AUCUNE
    dimension ne sait juger est `undecided`, et `unassigned` décide de son sort ('keep' par
    défaut, parce qu'une projection plafonnée n'est pas un avis).
    """
    if combine not in ("all", "any"):
        raise ValueError("combine doit valoir 'all' ou 'any'")
    if unassigned not in ("keep", "exclude"):
        raise ValueError("unassigned doit valoir 'keep' ou 'exclude'")

    verdicts: list[tuple[str, Any]] = []
    if cluster_of is not None:
        wanted = clusters_wanted or set()

        def _cluster(i: int) -> str:
            c = cluster_of.get(i)
            if c is None:
                return "unknown"
            return "in" if c in wanted else "out"

        verdicts.append(("clusters", _cluster))
    if concept_match is not None:
        known = concept_known if concept_known is not None else set(concept_match)

        def _concept(i: int) -> str:
            if i not in known:
                return "unknown"
            return "in" if i in concept_match else "out"

        verdicts.append(("concepts", _concept))
    if design_of is not None:
        wanted_designs = designs_wanted or set()

        def _design(i: int) -> str:
            # « Devis non précisé » est une VRAIE réponse, pas une absence : l'article a
            # été lu, et rien n'y indiquait de devis. Le relecteur peut donc choisir de le
            # garder ou non, au lieu que la dimension s'abstienne pour lui.
            key = design_of.get(i)
            if key is None:
                return "unknown"
            return "in" if key in wanted_designs else "out"

        verdicts.append(("designs", _design))
    if level_of is not None:
        wanted_levels = levels_wanted or set()

        def _level(i: int) -> str:
            level = level_of.get(i)
            if level is None:
                return "unknown"
            return "in" if level in wanted_levels else "out"

        verdicts.append(("levels", _level))

    if not verdicts:
        raise ValueError("aucune dimension de sélection : donnez des clusters, des concepts, des devis d'étude ou des niveaux de preuve")

    keep: list[int] = []
    drop: list[int] = []
    undecided: list[int] = []
    tally = {name: {"in": 0, "out": 0, "unknown": 0} for name, _ in verdicts}

    for i in relevant:
        said: list[str] = []
        for name, fn in verdicts:
            v = fn(i)
            tally[name][v] += 1
            if v != "unknown":
                said.append(v)
        if not said:
            undecided.append(i)
            continue
        ok = all(v == "in" for v in said) if combine == "all" else any(v == "in" for v in said)
        (keep if ok else drop).append(i)

    if unassigned == "exclude":
        drop.extend(undecided)
    else:
        keep.extend(undecided)

    return {
        "relevant": len(relevant),
        "keep_ids": keep,
        "exclude_ids": drop,
        "undecided_ids": undecided,
        "keep": len(keep),
        "exclude": len(drop),
        "undecided": len(undecided),
        "by_dimension": tally,
        "combine": combine,
        "unassigned": unassigned,
    }


def describe_selection(clusters: list[int] | None, cluster_names: dict[int, str] | None,
                       concepts: list[str] | None, combine: str, unassigned: str,
                       threshold: float | None = None,
                       designs: list[str] | None = None,
                       levels: list[str] | None = None) -> str:
    """La phrase stockée dans `screening_reason`, donc celle que lira la PRISMA et qui
    devra tenir dans une section Méthodes. Elle nomme ce qui a été GARDÉ, pas un
    identifiant de calcul : les numéros de cluster changent au recalcul suivant.

    Elle porte AUSSI le seuil en vigueur au moment où on l'applique, parce que le
    découpage ne juge que les articles pertinents à ce seuil-là. Abaisser le seuil ensuite
    fait entrer des articles que la sélection n'a jamais vus, et sans cette mention rien
    ne le rappelle : ni la PRISMA, ni la courbe du seuil, ni la personne qui relit."""
    parts: list[str] = []
    if clusters:
        names = [f"{(cluster_names or {}).get(c) or f'cluster {c}'}" for c in clusters]
        parts.append("clusters " + ", ".join(names))
    if concepts:
        parts.append("concepts " + ", ".join(concepts))
    # Les devis et les niveaux sont nommés en toutes lettres, comme les clusters : une
    # section Méthodes doit pouvoir dire « restreint aux essais randomisés » sans renvoyer
    # à un code interne.
    if designs:
        parts.append("devis " + ", ".join(designs))
    if levels:
        parts.append("niveaux de preuve " + ", ".join(levels))
    joined = (" et " if combine == "all" else " ou ").join(parts) if len(parts) > 1 else (parts[0] if parts else "")
    tail = "" if unassigned == "keep" else " ; articles hors projection exclus aussi"
    at = f" (seuil {float(threshold):.4g})" if threshold is not None else ""
    return f"{SUBSET_REASON_PREFIX} hors de {joined}{tail}{at}"[:480]


# Le seuil relu dans le motif. Un motif reste du texte libre (un relecteur peut écrire le
# sien), donc l'absence de correspondance n'est pas une erreur : on ne sait simplement pas
# à quel seuil ce découpage a été posé, et on le dit plutôt que d'inventer.
_SCOPE_THRESHOLD_RE = re.compile(r"\(seuil\s+([0-9]*\.?[0-9]+)\)\s*$")


def scope_threshold_of(reason: str | None) -> float | None:
    """Le seuil auquel ce découpage a été appliqué, relu dans son motif. Pur."""
    m = _SCOPE_THRESHOLD_RE.search(reason or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def scope_state(scenario_id: str) -> dict[str, Any]:
    """Les découpages en vigueur : combien d'articles ils tiennent à l'écart, et le seuil
    le plus BAS auquel l'un d'eux a été posé. Ce seuil est la frontière au-dessous de
    laquelle plus rien n'a été jugé par la sélection."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT screening_reason AS reason, COUNT(*) AS n, MAX(screened_at) AS at
            FROM article_scenarios
            WHERE scenario_id = :sid AND screening_status = 'excluded'
              AND screening_reason LIKE :pfx
            GROUP BY screening_reason
            ORDER BY n DESC
        """), {"sid": scenario_id, "pfx": SUBSET_REASON_PREFIX + "%"}).mappings().all()
    steps = [{"reason": r["reason"], "articles": int(r["n"]),
              "applied_at": r["at"].isoformat() if r["at"] else None,
              "applied_at_threshold": scope_threshold_of(r["reason"])} for r in rows]
    thresholds = [s["applied_at_threshold"] for s in steps if s["applied_at_threshold"] is not None]
    return {
        "narrowed": bool(steps),
        "excluded_by_scope": sum(s["articles"] for s in steps),
        "judged_above_threshold": min(thresholds) if thresholds else None,
        "steps": steps,
    }


# ─── Lire les trois ingrédients dans la base ────────────────────────────────
def _relevant_ids(scenario_id: str, threshold: float) -> list[int]:
    """Le sous-ensemble pertinent ACTUEL, par la porte commune. C'est sur lui seul que le
    découpage opère : le clustering et la carte n'ont jamais rien vu d'autre, donc ils
    n'ont rien à dire des articles sous le seuil."""
    gate = relevant_gate_sql(doc="d", link="ars", thr=":thr")
    with engine.connect() as conn:
        return [int(r[0]) for r in conn.execute(text(f"""
            SELECT d.id
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {gate}
            ORDER BY d.id
        """), {"sid": scenario_id, "thr": threshold}).all()]


def _cluster_membership(scenario_id: str) -> tuple[dict[int, int], dict[int, str], dict]:
    """(article -> cluster, cluster -> nom, méta du cache). Lève 404 si rien n'est en
    cache : proposer un découpage par clusters sans clustering calculé ne peut produire
    qu'un corpus vidé de tout ce que la projection n'a pas vu."""
    from .clustering import CLUSTER_MAX_DOCS, _load_viz_cache

    cache = _load_viz_cache(scenario_id, "clustering")
    if not cache or not cache.get("clusters"):
        raise HTTPException(status_code=404,
                            detail="Aucun clustering en cache pour ce scénario : ouvrez "
                                   "l'onglet Clusters pour le calculer, puis réessayez.")
    of: dict[int, int] = {}
    names: dict[int, str] = {}
    for c in cache["clusters"]:
        if not isinstance(c, dict) or c.get("cluster_id") is None:
            continue
        cid = int(c["cluster_id"])
        names[cid] = str(c.get("cluster_name") or f"cluster {cid}")
        for p in c.get("points") or []:
            if isinstance(p, dict) and p.get("id") is not None:
                of[int(p["id"])] = cid
    meta = {"n_docs_clustered": int(cache.get("n_docs") or len(of)),
            "n_docs_eligible": int(cache.get("n_docs_total") or 0),
            "cluster_max_docs": CLUSTER_MAX_DOCS,
            "lang": cache.get("lang")}
    return of, names, meta


def _design_membership(scenario_id: str, threshold: float) -> tuple[dict[int, str], dict[int, str], dict]:
    """({id: devis}, {id: niveau de preuve}, méta), sur tout le sous-ensemble pertinent.

    Les deux viennent de `api/study_design`, la même table que les graphiques et que la
    notation des affirmations. Un article sans devis identifiable n'est pas absent de ces
    cartes : il porte « Devis non précisé » et « Non évaluée ». C'est délibéré, et ça
    change ce que le relecteur peut faire. Pour les clusters, `unknown` veut dire « hors
    de la projection, la carte ne dit rien de lui » ; ici l'article A été lu, et le fait
    qu'aucun devis n'y figure est une réponse. Le relecteur peut donc décider de garder ou
    d'écarter les devis non précisés, au lieu que la dimension s'abstienne pour lui."""
    from .study_design import classify, grade_level, label, raw_design_sql
    _raw_design_d = raw_design_sql("d")

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT d.id, {_raw_design_d} AS raw
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {relevant_gate_sql(doc="d", link="ars")}
        """), {"sid": scenario_id, "thr": threshold}).mappings().all()

    design_of: dict[int, str] = {}
    level_of: dict[int, str] = {}
    counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    for r in rows:
        article_id = int(r["id"])
        design_label = label(classify(r["raw"]))
        level = grade_level(r["raw"])
        design_of[article_id] = design_label
        level_of[article_id] = level
        counts[design_label] = counts.get(design_label, 0) + 1
        level_counts[level] = level_counts.get(level, 0) + 1
    meta = {
        "judged": len(design_of),
        "designs": [{"value": k, "n": v} for k, v in
                    sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))],
        "levels": [{"value": k, "n": v} for k, v in
                   sorted(level_counts.items(), key=lambda kv: (-kv[1], kv[0]))],
    }
    return design_of, level_of, meta


def _concept_membership(scenario_id: str, concepts: list[str], mode: str) -> tuple[set[int], set[int], dict]:
    """(articles correspondants, articles JUGEABLES, méta).

    Jugeables = ceux dont les concepts ont été extraits. Les autres sont `unknown` : la
    carte ne dit pas qu'ils sont hors sujet, elle ne dit rien d'eux."""
    from .knowledge_graph import _build_concept_graph, _concept_rows

    wanted: list[tuple[str, str]] = []
    for part in concepts:
        part = (part or "").strip()
        if not part:
            continue
        if ":" not in part:
            raise HTTPException(status_code=400,
                                detail=f"« {part} » doit s'écrire type:label (ex. pathogen:dengue virus).")
        t, lab = part.split(":", 1)
        wanted.append((t.strip().lower(), lab.strip()))
    if not wanted:
        raise HTTPException(status_code=400, detail="Aucun concept demandé.")

    rows, n_total = _concept_rows(scenario_id)
    known = {int(r["id"]) for r in rows if r.get("concepts_json")}
    graph = _build_concept_graph(rows, n_total=n_total, full_articles=True)
    by_key = {(str(n["type"]).lower(), str((n.get("label") or {}).get("en") or "").strip().lower()): n
              for n in graph.get("nodes") or []}
    sets: list[set[int]] = []
    labels: list[str] = []
    missing: list[str] = []
    for t, lab in wanted:
        node = by_key.get((t, lab.lower()))
        if node is None:
            missing.append(f"{t}:{lab}")
            continue
        sets.append({int(i) for i in (node.get("articles") or [])})
        labels.append(str((node.get("label") or {}).get("en") or lab))
    if missing:
        raise HTTPException(status_code=404,
                            detail=f"Concept(s) absent(s) de la carte : {', '.join(missing)}. "
                                   f"Les libellés sont ceux de la carte (anglais canonique).")
    match = set.union(*sets) if mode == "any" else set.intersection(*sets)
    meta = {"labels": labels, "concept_mode": mode,
            "n_with_concepts": len(known), "n_rows": len(rows)}
    return match, known, meta


# ─── Le plan complet, tel que les trois endpoints le partagent ──────────────
def _build_plan(scenario_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    clusters = payload.get("clusters")
    concepts = payload.get("concepts")
    designs = payload.get("designs")
    levels = payload.get("levels")
    combine = str(payload.get("combine") or "all")
    unassigned = str(payload.get("unassigned") or "keep")
    concept_mode = str(payload.get("concept_mode") or "any")
    if concept_mode not in ("any", "all"):
        raise HTTPException(status_code=422, detail="concept_mode doit valoir 'any' ou 'all'")
    if not clusters and not concepts and not designs and not levels:
        raise HTTPException(status_code=422,
                            detail="Donnez au moins `clusters`, `concepts`, `designs` ou `levels` : "
                                   "sans dimension de sélection il n'y a rien à découper.")

    threshold = _get_scenario_threshold(scenario_id)
    relevant = _relevant_ids(scenario_id, threshold)
    meta: dict[str, Any] = {"threshold": threshold}

    cluster_of = cluster_names = None
    wanted_clusters: set[int] | None = None
    if clusters:
        try:
            wanted_clusters = {int(c) for c in clusters}
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="`clusters` doit être une liste d'entiers.")
        cluster_of, cluster_names, cmeta = _cluster_membership(scenario_id)
        unknown_ids = sorted(wanted_clusters - set(cluster_names))
        if unknown_ids:
            raise HTTPException(status_code=404,
                                detail=f"Cluster(s) inconnu(s) : {unknown_ids} "
                                       f"(disponibles : {sorted(cluster_names)}).")
        meta["clustering"] = cmeta

    concept_match = concept_known = None
    concept_labels: list[str] = []
    if concepts:
        if isinstance(concepts, str):
            concepts = [p for p in concepts.split("|") if p.strip()]
        concept_match, concept_known, kmeta = _concept_membership(scenario_id, list(concepts), concept_mode)
        concept_labels = kmeta["labels"]
        meta["concepts"] = kmeta

    design_of = level_of = None
    wanted_designs = wanted_levels = None
    if designs or levels:
        design_map, level_map, dmeta = _design_membership(scenario_id, threshold)
        meta["study_designs"] = dmeta
        if designs:
            wanted_designs = {str(x) for x in designs}
            known = {d["value"] for d in dmeta["designs"]}
            unknown_labels = sorted(wanted_designs - known)
            if unknown_labels:
                raise HTTPException(status_code=404,
                                    detail=f"Devis inconnu(s) dans ce corpus : {unknown_labels} "
                                           f"(disponibles : {sorted(known)}).")
            design_of = design_map
        if levels:
            wanted_levels = {str(x) for x in levels}
            known_levels = {l["value"] for l in dmeta["levels"]}
            unknown_levels = sorted(wanted_levels - known_levels)
            if unknown_levels:
                raise HTTPException(status_code=404,
                                    detail=f"Niveau(x) inconnu(s) dans ce corpus : {unknown_levels} "
                                           f"(disponibles : {sorted(known_levels)}).")
            level_of = level_map

    try:
        plan = plan_subset(relevant, cluster_of=cluster_of, clusters_wanted=wanted_clusters,
                           concept_match=concept_match, concept_known=concept_known,
                           design_of=design_of, designs_wanted=wanted_designs,
                           level_of=level_of, levels_wanted=wanted_levels,
                           combine=combine, unassigned=unassigned)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    plan["meta"] = meta
    plan["reason"] = describe_selection(sorted(wanted_clusters) if wanted_clusters else None,
                                        cluster_names, concept_labels or None, combine, unassigned,
                                        threshold=threshold,
                                        designs=sorted(wanted_designs) if wanted_designs else None,
                                        levels=sorted(wanted_levels) if wanted_levels else None)
    return plan


def _reset_corpus_caches(conn, scenario_id: str) -> None:
    """Les artefacts calculés sur le corpus pertinent (clustering, graphe, carte des
    concepts, actions recommandées) décrivent un corpus qui vient de changer : les servir
    ensuite présenterait le corpus PRÉCÉDENT comme l'actuel, ce qu'a déjà fait le curseur
    de seuil. Même liste que lui, pour qu'un artefact ajouté demain ne soit pas oublié
    d'un côté."""
    conn.execute(text("""
        INSERT INTO scenario_settings (scenario_id, updated_at) VALUES (:sid, NOW())
        ON CONFLICT (scenario_id) DO NOTHING
    """), {"sid": scenario_id})
    conn.execute(text(f"""
        UPDATE scenario_settings SET {CORPUS_DERIVED_CACHE_RESET} WHERE scenario_id = :sid
    """), {"sid": scenario_id})


def _public(plan: dict[str, Any], sample: int = 20) -> dict[str, Any]:
    """La réponse servie : des COMPTES, plus un échantillon d'identifiants pour vérifier.
    Renvoyer huit mille identifiants pour afficher un nombre est un aller-retour inutile,
    et les cacher tous empêcherait de contrôler ce qu'on s'apprête à exclure."""
    out = {k: v for k, v in plan.items() if not k.endswith("_ids")}
    out["exclude_sample"] = plan["exclude_ids"][:sample]
    out["undecided_sample"] = plan["undecided_ids"][:sample]
    return out


# ─── Endpoints ──────────────────────────────────────────────────────────────
@app.post("/user-scenarios/{scenario_id}/subset/preview")
def preview_scenario_subset(scenario_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Ce que le découpage ferait, sans rien écrire.

    `keep` / `exclude` / `undecided` portent sur le sous-ensemble PERTINENT actuel.
    `undecided` est la ligne à lire avant d'appliquer : ce sont les articles que la
    sélection ne sait pas juger (hors du clustering plafonné, ou sans concepts extraits).
    Ils sont gardés, sauf `unassigned: "exclude"`."""
    _get_user_scenario_or_404(scenario_id)
    return _public(_build_plan(scenario_id, payload))


@app.post("/user-scenarios/{scenario_id}/subset/apply")
def apply_scenario_subset(scenario_id: str, payload: dict[str, Any],
                          _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Applique le découpage : les articles écartés passent à `excluded` SUR CE SCÉNARIO.

    Rien d'autre ne bouge. La décision est écrite sur `article_scenarios`, jamais sur la
    ligne globale du document : restreindre la portée d'un scénario ne doit pas retirer
    l'article des autres. Elle porte le préfixe `scope:` et la description de la sélection,
    de sorte que la PRISMA affiche le motif et que l'annulation retrouve exactement ces
    lignes.

    Les découpages se CUMULENT : appliquer une seconde sélection restreint ce qui reste.
    Pour élargir, annulez d'abord (`/subset/undo`).

    Les artefacts calculés sur le corpus (clustering, graphe, carte des concepts, actions
    recommandées) sont invalidés, parce qu'ils décrivent un corpus qui vient de changer.
    Le clustering affiché disparaît donc : c'est voulu, il portait sur l'ancien corpus."""
    _get_user_scenario_or_404(scenario_id)
    plan = _build_plan(scenario_id, payload)
    ids = plan["exclude_ids"]
    if not ids:
        return {**_public(plan), "applied": 0, "status": "noop",
                "message": "La sélection ne retire aucun article : rien n'a été écrit."}
    if plan["keep"] == 0:
        raise HTTPException(status_code=422,
                            detail="Cette sélection ne laisserait aucun article. Un corpus vide "
                                   "n'est pas un découpage : élargissez la sélection.")
    reason = str(payload.get("reason") or plan["reason"])[:480]
    if not reason.startswith(SUBSET_REASON_PREFIX):
        reason = f"{SUBSET_REASON_PREFIX} {reason}"
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE article_scenarios
            SET screening_status = 'excluded', screening_reason = :reason, screened_at = NOW()
            WHERE scenario_id = :sid AND document_id = ANY(:ids)
        """), {"sid": scenario_id, "ids": ids, "reason": reason})
        _reset_corpus_caches(conn, scenario_id)
    logger.info(f"subset apply {scenario_id}: {len(ids)} articles excluded ({reason})")
    return {**_public(plan), "applied": len(ids), "status": "applied", "reason": reason,
            "caveat": ("Ce découpage porte sur les articles pertinents au seuil "
                       f"{plan['meta']['threshold']}. Abaisser le seuil ensuite fera entrer "
                       "des articles que cette sélection n'a jamais jugés.")}


@app.get("/user-scenarios/{scenario_id}/subset")
def get_scenario_subset(scenario_id: str) -> dict[str, Any]:
    """Quel découpage est en vigueur, et combien d'articles il retient à l'écart.

    Une restriction de portée qui ne se voit nulle part est une restriction qu'on oublie,
    puis qu'on lit dans un brief sans savoir qu'elle est là.

    `judged_above_threshold` est la frontière : la sélection n'a jugé que les articles
    pertinents à ce seuil. En descendant plus bas, on fait entrer des articles qu'elle
    n'a jamais vus."""
    _get_user_scenario_or_404(scenario_id)
    return {"scenario_id": scenario_id, **scope_state(scenario_id)}


@app.post("/user-scenarios/{scenario_id}/subset/undo")
def undo_scenario_subset(scenario_id: str, reason: str | None = None,
                         _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Annule les découpages : remet à `pending` les articles que CE mécanisme a exclus.

    `reason` annule une seule étape (celle dont le motif correspond exactement) ; sans
    elle, toutes. Les exclusions posées à la main par un relecteur ne sont jamais touchées:
    elles ne portent pas le préfixe, et ce travail-là ne se refait pas."""
    _get_user_scenario_or_404(scenario_id)
    params: dict[str, Any] = {"sid": scenario_id, "pfx": SUBSET_REASON_PREFIX + "%"}
    extra = ""
    if reason:
        extra = " AND screening_reason = :reason"
        params["reason"] = reason
    with engine.begin() as conn:
        n = conn.execute(text(f"""
            UPDATE article_scenarios
            SET screening_status = 'pending', screening_reason = NULL, screened_at = NOW()
            WHERE scenario_id = :sid AND screening_status = 'excluded'
              AND screening_reason LIKE :pfx{extra}
        """), params).rowcount or 0
        _reset_corpus_caches(conn, scenario_id)
    logger.info(f"subset undo {scenario_id}: {n} articles restored")
    return {"scenario_id": scenario_id, "restored": int(n),
            "status": "restored" if n else "noop"}
