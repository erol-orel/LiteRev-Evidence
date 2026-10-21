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

from fastapi import Depends, HTTPException, Query, Response
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .scenario_store import (
    mark_corpus_moved,
    _get_scenario_threshold,
    _get_user_scenario_or_404,
    relevant_gate_sql,
)

# Ce préfixe est la SIGNATURE des exclusions posées ici : l'annulation ne défait que
# celles-là. Sans lui, « annuler le sous-corpus » effacerait aussi les exclusions qu'un
# relecteur a posées une par une, ce qui est exactement le travail qu'on ne peut pas
# refaire.
SUBSET_REASON_PREFIX = "scope:"

#: L'identifiant que HDBSCAN donne aux articles qu'il n'a rangés dans aucun thème.
NOISE_CLUSTER = -1


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
    country_of: dict[int, str] | None = None,
    countries_wanted: set[str] | None = None,
    # Les conditions GÉNÉRIQUES, une par variable du registre que les cinq dimensions
    # nommées ci-dessus ne couvrent pas : `(nom, fonction de verdict)`. Les cinq gardent
    # leur chemin propre parce que leur sémantique est particulière et éprouvée (le bruit
    # du clustering, l'Europe sans pays unique, « devis non précisé » qui EST une réponse),
    # et une unification qui les réécrirait changerait leur comportement le jour où elle
    # est faite. Ce qui s'unifie, c'est le nombre de dimensions, pas leurs règles.
    conditions: list[tuple[str, Any]] | None = None,
    combine: str = "all",
    unassigned: str = "keep",
    # Les articles qu'un relecteur a INCLUS À LA MAIN. Ils ne sont jamais exclus par
    # une restriction de portée : la lecture d'un relecteur ne se défait pas en un clic.
    included_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Qui reste, qui sort, et qui n'a pas pu être jugé. PUR.

    Une dimension vaut `None` quand la sélection ne s'en sert pas. Pour chaque article et
    chaque dimension employée, le verdict est `in`, `out`, ou `unknown` (l'article est
    hors de la projection : jamais clusterisé, rangé dans aucun thème (groupe -1, sauf si
    ce groupe est coché), ou sans concepts extraits).

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
    unclustered = 0
    if cluster_of is not None:
        wanted = clusters_wanted or set()

        def _cluster(i: int) -> str:
            c = cluster_of.get(i)
            if c is None:
                return "unknown"
            # Le groupe -1 n'est pas un thème : c'est ce que HDBSCAN n'a PAS su grouper.
            # Le traiter comme un cluster qu'on n'a pas coché écrivait `excluded` sur
            # chacun de ses articles (« garder A et B » jetait tout le bruit, qui pèse
            # lourd sur un petit corpus). Il ne juge donc rien, comme un article au-delà
            # du plafond, sauf si le relecteur a coché ce groupe en toutes lettres.
            if c == NOISE_CLUSTER:
                return "in" if c in wanted else "unknown"
            return "in" if c in wanted else "out"

        unclustered = sum(1 for i in relevant if cluster_of.get(i) == NOISE_CLUSTER)
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
    if country_of is not None:
        wanted_countries = countries_wanted or set()

        def _country(i: int) -> str:
            # Le lieu de l'ÉTUDE, résolu à la lecture comme dans l'onglet Géographie. Un
            # article dont aucun lieu ne se résout ne vote pas : la carte ne dit pas qu'il
            # est ailleurs, elle ne dit rien de lui. Un article européen sans pays unique
            # (« Europe », deux pays à la fois) porte la clé EUROPE : il EST jugeable,
            # mais seulement par une sélection qui demande l'Europe.
            key = country_of.get(i)
            if key is None:
                return "unknown"
            return "in" if key in wanted_countries else "out"

        verdicts.append(("countries", _country))
    for name, fn in (conditions or ()):
        verdicts.append((str(name), fn))

    if not verdicts:
        raise ValueError("aucune dimension de sélection : donnez des clusters, des concepts, "
                         "des devis d'étude, des niveaux de preuve, des pays, ou une condition "
                         "sur une variable du registre")

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

    # Une INCLUSION POSÉE À LA MAIN n'est jamais recouverte en silence par une
    # restriction de portée. Un relecteur avait rescapé l'article en le lisant ; une
    # règle de périmètre appliquée en un clic ne défait pas cette lecture. Ils sont
    # gardés et COMPTÉS à part, pour que la restriction dise ce qu'elle n'a pas fait.
    kept_manual = [i for i in drop if included_ids and i in included_ids]
    if kept_manual:
        kept = set(kept_manual)
        drop = [i for i in drop if i not in kept]
        keep.extend(kept_manual)

    return {
        "relevant": len(relevant),
        "keep_ids": keep,
        "exclude_ids": drop,
        "undecided_ids": undecided,
        "kept_manual_ids": kept_manual,
        "kept_manual": len(kept_manual),
        "keep": len(keep),
        "exclude": len(drop),
        "undecided": len(undecided),
        # Combien d'articles pertinents la projection n'a pas groupés (le groupe -1) :
        # l'aperçu doit pouvoir dire quelle part de « non jugés » vient de là.
        "unclustered": unclustered,
        "by_dimension": tally,
        "combine": combine,
        "unassigned": unassigned,
    }


def selection_phrase(clusters: list[int] | None, cluster_names: dict[int, str] | None,
                     concepts: list[str] | None, combine: str, *,
                     designs: list[str] | None = None,
                     levels: list[str] | None = None,
                     countries: list[str] | None = None,
                     conditions: list[str] | None = None) -> str:
    """CE QUE LA SÉLECTION GARDE, en une phrase, sans verbe. PURE.

    Extraite de `describe_selection` parce que les deux sens de la même sélection sont
    lus à deux endroits : le motif stocké parle de ce qui SORT (« hors de pays Chine »,
    posé sur les articles exclus, relu par la PRISMA), et l'export parle de ce qu'il
    CONTIENT. Réutiliser le premier pour le second faisait dire à un fichier de 465
    articles chinois qu'il était « hors de pays Chine », soit l'inverse de son contenu,
    et un fichier qui se décrit de travers est pire qu'un fichier sans description."""
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
    # Les pays sont nommés, eux aussi : « pays Italie, France » se lit dans une section
    # Méthodes, « IT, FR » non.
    if countries:
        parts.append("pays " + ", ".join(countries))
    # Les conditions génériques arrivent déjà rédigées (« année à partir de 2015 »,
    # « langue en, fr ») : elles nomment leur variable et leurs valeurs en clair, pour la
    # même raison que les clusters sont nommés plutôt que numérotés.
    parts.extend(conditions or [])
    if len(parts) > 1:
        return (" et " if combine == "all" else " ou ").join(parts)
    return parts[0] if parts else ""


def describe_selection(clusters: list[int] | None, cluster_names: dict[int, str] | None,
                       concepts: list[str] | None, combine: str, unassigned: str,
                       threshold: float | None = None,
                       designs: list[str] | None = None,
                       levels: list[str] | None = None,
                       countries: list[str] | None = None,
                       conditions: list[str] | None = None) -> str:
    """La phrase stockée dans `screening_reason`, donc celle que lira la PRISMA et qui
    devra tenir dans une section Méthodes. Elle nomme ce qui a été GARDÉ, pas un
    identifiant de calcul : les numéros de cluster changent au recalcul suivant.

    Elle porte AUSSI le seuil en vigueur au moment où on l'applique, parce que le
    découpage ne juge que les articles pertinents à ce seuil-là. Abaisser le seuil ensuite
    fait entrer des articles que la sélection n'a jamais vus, et sans cette mention rien
    ne le rappelle : ni la PRISMA, ni la courbe du seuil, ni la personne qui relit."""
    joined = selection_phrase(clusters, cluster_names, concepts, combine, designs=designs,
                              levels=levels, countries=countries, conditions=conditions)
    tail = "" if unassigned == "keep" else " ; articles hors projection exclus aussi"
    # Le seuil EN TÊTE : le motif est coupé à 480 caractères, et une longue sélection
    # (vingt concepts, dix clusters) emportait le « (seuil X) » final avec elle. La
    # frontière de la sélection disparaissait alors de l'état du découpage et de la
    # courbe du seuil, qui la relisent ici.
    at = f"(seuil {float(threshold):.4g}) " if threshold is not None else ""
    return f"{SUBSET_REASON_PREFIX} {at}hors de {joined}{tail}"[:480]


# Le seuil relu dans le motif. Un motif reste du texte libre (un relecteur peut écrire le
# sien), donc l'absence de correspondance n'est pas une erreur : on ne sait simplement pas
# à quel seuil ce découpage a été posé, et on le dit plutôt que d'inventer. Il est écrit
# en tête depuis que la coupe à 480 caractères pouvait l'emporter ; les motifs écrits
# avant, qui le portent à la fin, se relisent toujours.
_SCOPE_THRESHOLD_RE = re.compile(
    r"^" + re.escape(SUBSET_REASON_PREFIX) + r"\s*\(seuil\s+([0-9]*\.?[0-9]+)\)")
_SCOPE_THRESHOLD_OLD_RE = re.compile(r"\(seuil\s+([0-9]*\.?[0-9]+)\)\s*$")


def scope_threshold_of(reason: str | None) -> float | None:
    """Le seuil auquel ce découpage a été appliqué, relu dans son motif. Pur."""
    m = (_SCOPE_THRESHOLD_RE.search(reason or "")
         or _SCOPE_THRESHOLD_OLD_RE.search(reason or ""))
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


def _manually_included_ids(scenario_id: str) -> set[int]:
    """Les articles que CE scénario a inclus à la main. Une restriction de portée les
    garde et le DIT, au lieu de recouvrir une décision de relecteur en silence."""
    with engine.connect() as conn:
        return {int(r[0]) for r in conn.execute(text("""
            SELECT document_id FROM article_scenarios
            WHERE scenario_id = :sid AND screening_status = 'included'
        """), {"sid": scenario_id}).all()}


def _cluster_membership(scenario_id: str) -> tuple[dict[int, int], dict[int, str], dict]:
    """(article -> cluster, cluster -> nom, méta du cache). Lève 404 si rien n'est en
    cache : proposer un découpage par clusters sans clustering calculé ne peut produire
    qu'un corpus vidé de tout ce que la projection n'a pas vu."""
    from .clustering import CLUSTER_MAX_DOCS, _load_viz_cache

    cache = _load_viz_cache(scenario_id, "clustering")
    if not cache or not cache.get("clusters"):
        # Deux absences très différentes derrière le même cache vide, et l'ancien message
        # ne disait que la première. Bouger le seuil périme le clustering et en relance
        # un : l'écran continue d'afficher les groupes d'AVANT, le relecteur en coche un,
        # et on lui répondait « ouvrez l'onglet Clusters pour le calculer » alors qu'il y
        # était et que le calcul tournait. Dire laquelle des deux c'est, et dans le cas
        # qui passe, dire d'attendre.
        from .clustering import _clustering_jobs
        if (_clustering_jobs.get(scenario_id) or {}).get("status") == "running":
            raise HTTPException(status_code=409,
                                detail="Le clustering est en cours de recalcul pour ce "
                                       "scénario : les groupes affichés sont ceux d'avant. "
                                       "Attendez la fin du calcul, puis refaites la sélection "
                                       "sur les nouveaux groupes.")
        raise HTTPException(status_code=404,
                            detail="Aucun clustering en cache pour ce scénario : ouvrez "
                                   "l'onglet Clusters pour le calculer, puis réessayez.")
    # UN CLUSTERING PÉRIMÉ NE SÉLECTIONNE RIEN. Le numéro d'un groupe n'a de sens que
    # dans la projection qui l'a produit : le « groupe 0 » d'un corpus d'avant n'est pas
    # celui d'aujourd'hui, et découper dessus exclurait les mauvais articles. Le cache
    # était effacé à chaque mouvement du corpus, ce qui l'interdisait par accident ; il
    # est désormais gardé (ses résumés coûtent un appel par groupe), donc le refus est
    # écrit. C'est la même réponse qu'une absence, et elle est plus juste qu'elle : le
    # clustering est là, il décrit simplement autre chose.
    if cache.get("stale"):
        raise HTTPException(status_code=404,
                            detail="Le clustering en cache décrit le corpus d'avant "
                                   f"(construit le {str(cache.get('built_at'))[:10]}, le corpus a "
                                   f"bougé le {str(cache.get('corpus_moved_at'))[:10]}) : les numéros "
                                   "de groupe ne correspondent plus. Recalculez le clustering, "
                                   "puis refaites la sélection sur les nouveaux groupes.")
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


#: La clé que porte un article EUROPÉEN DONT LE PAYS N'EST PAS UNIQUE (« Europe »,
#: « Germany and the Netherlands »). La carte ne peut pas le poser sur un pays, mais la
#: question « garder l'Europe » le concerne, et le laisser « non jugé » le ferait entrer
#: dans un sous-corpus italien. Ce n'est pas un code ISO : aucun pays ne s'appelle ainsi.
EUROPE_KEY = "EUROPE"


def _country_membership(scenario_id: str, threshold: float) -> tuple[dict[int, str], dict]:
    """(article -> pays, méta). Le pays de l'ÉTUDE, pas celui des auteurs.

    Lu exactement comme l'onglet Géographie et comme le digest : le texte de lieu que
    l'extraction a relevé (« Lombardy, Italy »), passé au résolveur et à l'index NUTS.
    `literature_document.country` est presque toujours vide, le croiser en SQL donnerait
    zéro partout.

    Un article sans extraction, ou dont le lieu ne se résout pas, n'est PAS dans la carte
    renvoyée : il reste `unknown` pour la sélection, donc gardé. C'est la même règle que
    pour le clustering plafonné : une absence de réponse n'est pas un avis."""
    from .geography import COUNTRY_NAME, NUTS_ISO, get_nuts_index, resolve_location

    gate = relevant_gate_sql("d", "ars", ":thr")
    index = get_nuts_index()
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT d.id, d.country AS country,
                   d.extraction_json->'ref'->>'location' AS location
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {gate}
              AND jsonb_typeof(d.extraction_json) = 'object'
        """), {"sid": scenario_id, "thr": threshold}).mappings().all()

    of: dict[int, str] = {}
    counts: dict[str, int] = {}
    n_extracted = len(rows)
    for r in rows:
        got = resolve_location(r["location"], r["country"], index) or {}
        iso = got.get("country")
        key = iso or (EUROPE_KEY if got.get("in_europe") is True else None)
        if not key:
            continue
        of[int(r["id"])] = key
        counts[key] = counts.get(key, 0) + 1
    meta = {
        "n_extracted": n_extracted,
        "n_placed": len(of),
        # Ceux que la sélection ne saura pas juger : pas d'extraction, ou pas de lieu.
        "n_unplaced": n_extracted - len(of),
        "europe_key": EUROPE_KEY,
        "europe_iso": sorted(NUTS_ISO),
        "countries": [{"value": k, "name": (COUNTRY_NAME.get(k, k) if k != EUROPE_KEY
                                            else "Europe (no single country)"), "n": v}
                      for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))],
    }
    return of, meta


def _concept_membership(scenario_id: str, concepts: list[str], mode: str,
                        concept_filter: dict[str, Any] | None = None) -> tuple[set[int], set[int], dict]:
    """(articles correspondants, articles JUGEABLES, méta).

    Jugeables = ceux dont les concepts ont été extraits. Les autres sont `unknown` : la
    carte ne dit pas qu'ils sont hors sujet, elle ne dit rien d'eux.

    La sélection se résout sur l'index COMPLET des concepts (api/knowledge_graph.py),
    jamais sur la carte dessinée : ses soixante noeuds et ses quatorze concepts par type
    sont des coupes d'affichage, et un « garder la carte composée » qui en héritait
    excluait les articles de tous les concepts que l'écran ne montrait pas. Deux formes :
    `concepts` (liste explicite, union ou intersection selon `mode`), ou `concept_filter`
    ({exclude_types, min_count, exclude}), la carte telle que l'utilisateur l'a composée."""
    from .knowledge_graph import _concept_rows, parse_concept_refs, select_concepts

    cf = concept_filter if isinstance(concept_filter, dict) else None
    try:
        wanted = parse_concept_refs(list(concepts or []))
        removed = parse_concept_refs(list((cf or {}).get("exclude") or []))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not wanted and cf is None:
        raise HTTPException(status_code=400, detail="Aucun concept demandé.")
    try:
        floor = max(1, int((cf or {}).get("min_count") or 1))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="concept_filter.min_count doit être un entier.")

    rows, n_total = _concept_rows(scenario_id)
    known = {int(r["id"]) for r in rows if r.get("concepts_json")}
    sel = select_concepts(rows, concepts=wanted or None, mode=mode,
                          exclude_types={str(t) for t in ((cf or {}).get("exclude_types") or [])},
                          min_count=floor, exclude=removed)
    if sel["missing"]:
        raise HTTPException(status_code=404,
                            detail=f"Concept(s) absent(s) des articles pertinents : {', '.join(sel['missing'])}. "
                                   f"Les libellés sont ceux de la carte (anglais canonique).")
    # Le motif nomme les concepts gardés ; pour une carte composée, la règle qui la définit.
    labels = sel["labels"] if wanted else [f"de la carte composée : {sel['description']}"]
    meta = {"labels": labels, "concept_mode": mode if wanted else "any",
            "composed": not wanted, "n_concepts": sel["n_concepts"],
            "n_with_concepts": len(known), "n_rows": len(rows)}
    return sel["ids"], known, meta


# ─── Le plan complet, tel que les trois endpoints le partagent ──────────────
#: Une condition générique, telle que l'appelant l'écrit:
#:   {"variable": "year", "min": 2015}                      un intervalle, pour un nombre
#:   {"variable": "language", "values": ["en", "fr"]}       une liste de modalités
#:   {"variable": "human_exp.occupation", "values": [...]}  un groupe du codebook
#: Les variables que les cinq dimensions nommées couvrent déjà sont REDIRIGÉES vers elles,
#: pour que « pays Italie » écrit en condition et écrit en `countries` fassent exactement la
#: même chose, l'extension de l'Europe comprise.
_CONDITION_KEYS = ("variable", "values", "min", "max")


def _fold_conditions(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Le payload avec les conditions redirigées vers les cinq dimensions nommées quand
    elles les couvrent, et la liste de celles qui restent génériques. PUR."""
    from .variables_of_interest import ROUTED_TO_SUBSETS

    raw = payload.get("conditions")
    if raw is None:
        return payload, []
    if not isinstance(raw, list):
        raise HTTPException(status_code=422, detail="`conditions` doit être une liste.")
    folded = dict(payload)
    generic: list[dict[str, Any]] = []
    for i, c in enumerate(raw, 1):
        if not isinstance(c, dict) or not str(c.get("variable") or "").strip():
            raise HTTPException(status_code=422,
                                detail=f"condition {i} : il faut `variable`.")
        var = str(c["variable"]).strip()
        values = c.get("values")
        slot = ROUTED_TO_SUBSETS.get(var)
        if slot:
            if var == "europe":
                # « En Europe » EST la sélection par pays avec la clé Europe: une seule
                # résolution, donc un seul comportement à vérifier.
                values = [EUROPE_KEY]
            if not values:
                raise HTTPException(status_code=422,
                                    detail=f"condition {i} ({var}) : il faut `values`.")
            folded[slot] = list(folded.get(slot) or []) + list(values)
            continue
        generic.append({"variable": var, "values": values,
                        "min": c.get("min"), "max": c.get("max")})
    return folded, generic


def _generic_conditions(scenario_id: str, threshold: float,
                        wanted: list[dict[str, Any]]) -> tuple[list[tuple[str, Any]], list[str], dict]:
    """Les verdicts des conditions génériques, les phrases qui les nomment, et leur méta.

    Chaque variable est lue par le registre (`variable_membership`), qui la lit par la MÊME
    expression que celle qui la compte sur la page: une condition ne peut donc pas porter
    sur une autre lecture que celle que le relecteur a vue."""
    from .variables_of_interest import condition_verdict, filterable_keys, variable_membership

    out: list[tuple[str, Any]] = []
    phrases: list[str] = []
    meta: dict[str, Any] = {}
    allowed = filterable_keys()
    for c in wanted:
        var = c["variable"]
        if "." not in var and var not in allowed:
            raise HTTPException(
                status_code=404,
                detail=f"La variable « {var} » ne se filtre pas. Filtrables : "
                       f"{', '.join(sorted(allowed))}, ou un groupe du codebook (sheet.group).")
        try:
            got = variable_membership(scenario_id, var, threshold)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        kind, of, labels = got["kind"], got["of"], got["labels"]
        lo = hi = None
        asked: set[str] = set()
        if kind == "number":
            if c.get("min") is None and c.get("max") is None:
                raise HTTPException(status_code=422,
                                    detail=f"« {var} » est un nombre : donnez `min` et/ou `max`.")
            try:
                lo = None if c.get("min") is None else float(c["min"])
                hi = None if c.get("max") is None else float(c["max"])
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail=f"« {var} » : `min` et `max` sont des nombres.")
            if lo is not None and hi is not None and lo > hi:
                raise HTTPException(status_code=422, detail=f"« {var} » : `min` dépasse `max`.")
            bounds = " ".join(filter(None, [f"à partir de {lo:g}" if lo is not None else "",
                                            f"jusqu'à {hi:g}" if hi is not None else ""]))
            phrases.append(f"{var} {bounds}")
        else:
            asked = {str(v) for v in (c.get("values") or []) if str(v).strip()}
            if not asked:
                raise HTTPException(status_code=422, detail=f"« {var} » : aucune valeur demandée.")
            missing = sorted(asked - set(got["present"]))
            if missing:
                raise HTTPException(
                    status_code=404,
                    detail=f"Valeur(s) absente(s) de ce corpus pour « {var} » : {missing} "
                           f"(présentes : {', '.join(got['present'][:25]) or 'aucune'}).")
            phrases.append(f"{var} " + ", ".join(labels.get(v, v) for v in sorted(asked)))
        out.append((var, condition_verdict(of, kind, asked, lo, hi)))
        meta[var] = {"kind": kind, "n_with_value": len(of), "n_values": len(got["present"])}
    return out, phrases, meta


def _build_plan(scenario_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload, generic = _fold_conditions(payload)
    clusters = payload.get("clusters")
    concepts = payload.get("concepts")
    concept_filter = payload.get("concept_filter")
    designs = payload.get("designs")
    levels = payload.get("levels")
    countries = payload.get("countries")
    combine = str(payload.get("combine") or "all")
    unassigned = str(payload.get("unassigned") or "keep")
    concept_mode = str(payload.get("concept_mode") or "any")
    if concept_mode not in ("any", "all"):
        raise HTTPException(status_code=422, detail="concept_mode doit valoir 'any' ou 'all'")
    if not clusters and not concepts and not concept_filter and not designs and not levels \
            and not countries and not generic:
        raise HTTPException(status_code=422,
                            detail="Donnez au moins une condition (`conditions`), ou `clusters`, "
                                   "`concepts`, `concept_filter`, `designs`, `levels` ou `countries` : "
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
    if concepts or concept_filter:
        if isinstance(concepts, str):
            concepts = [p for p in concepts.split("|") if p.strip()]
        concept_match, concept_known, kmeta = _concept_membership(
            scenario_id, list(concepts or []), concept_mode,
            concept_filter if not concepts else None)
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

    country_of = None
    wanted_countries: set[str] | None = None
    country_labels: list[str] = []
    if countries:
        if isinstance(countries, str):
            countries = [p for p in countries.split(",") if p.strip()]
        asked = {str(x).strip().upper() for x in countries if str(x).strip()}
        if not asked:
            raise HTTPException(status_code=422, detail="`countries` ne nomme aucun pays.")
        country_of, cnmeta = _country_membership(scenario_id, threshold)
        from .geography import COUNTRY_NAME, NUTS_ISO

        present = {c["value"] for c in cnmeta["countries"]}
        # EUROPE n'est pas un pays : elle s'étend à tous les pays NUTS présents dans ce
        # corpus, PLUS les articles européens sans pays unique. Demander « l'Europe »
        # garde donc exactement ce que l'onglet Géographie compte comme européen.
        wanted_countries = set()
        labels: list[str] = []
        for code in sorted(asked):
            if code == EUROPE_KEY:
                wanted_countries |= ({EUROPE_KEY} | (present & set(NUTS_ISO)))
                labels.append("Europe (NUTS)")
                continue
            if code not in present:
                raise HTTPException(
                    status_code=404,
                    detail=f"Aucun article placé en « {code} » dans ce corpus "
                           f"(disponibles : {', '.join(sorted(present)) or 'aucun'}).")
            wanted_countries.add(code)
            labels.append(COUNTRY_NAME.get(code, code))
        country_labels = labels
        meta["countries"] = cnmeta

    conditions, condition_phrases, condition_meta = _generic_conditions(
        scenario_id, threshold, generic)
    if condition_meta:
        meta["conditions"] = condition_meta

    try:
        plan = plan_subset(relevant, cluster_of=cluster_of, clusters_wanted=wanted_clusters,
                           concept_match=concept_match, concept_known=concept_known,
                           design_of=design_of, designs_wanted=wanted_designs,
                           level_of=level_of, levels_wanted=wanted_levels,
                           country_of=country_of, countries_wanted=wanted_countries,
                           conditions=conditions,
                           combine=combine, unassigned=unassigned,
                           included_ids=_manually_included_ids(scenario_id))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    plan["meta"] = meta
    plan["kept_phrase"] = selection_phrase(
        sorted(wanted_clusters) if wanted_clusters else None, cluster_names,
        concept_labels or None, combine,
        designs=sorted(wanted_designs) if wanted_designs else None,
        levels=sorted(wanted_levels) if wanted_levels else None,
        countries=country_labels or None, conditions=condition_phrases or None)
    plan["reason"] = describe_selection(sorted(wanted_clusters) if wanted_clusters else None,
                                        cluster_names, concept_labels or None, combine, unassigned,
                                        threshold=threshold,
                                        designs=sorted(wanted_designs) if wanted_designs else None,
                                        levels=sorted(wanted_levels) if wanted_levels else None,
                                        countries=country_labels or None,
                                        conditions=condition_phrases or None)
    return plan


def _reset_corpus_caches(conn, scenario_id: str) -> None:
    """Un découpage vient de changer le corpus pertinent : les vues qui en dérivent
    décrivent celui d'avant. MÊME chemin que le curseur de seuil et que les passes de
    scoring (`mark_corpus_moved`), pour qu'un résultat ajouté demain ne soit pas traité
    d'une façon ici et d'une autre là : ce qui se refait pour rien est effacé, ce qui se
    paie est gardé et daté."""
    mark_corpus_moved(conn, scenario_id)


def _public(plan: dict[str, Any], sample: int = 20) -> dict[str, Any]:
    """La réponse servie : des COMPTES, plus un échantillon d'identifiants pour vérifier.
    Renvoyer huit mille identifiants pour afficher un nombre est un aller-retour inutile,
    et les cacher tous empêcherait de contrôler ce qu'on s'apprête à exclure."""
    out = {k: v for k, v in plan.items() if not k.endswith("_ids")}
    out["exclude_sample"] = plan["exclude_ids"][:sample]
    out["undecided_sample"] = plan["undecided_ids"][:sample]
    out["kept_manual_sample"] = plan.get("kept_manual_ids", [])[:sample]
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


@app.post("/user-scenarios/{scenario_id}/subset/export")
def export_scenario_subset(scenario_id: str, payload: dict[str, Any],
                           format: str = Query("csv"),
                           include_abstract: bool = True) -> Response:
    """Les articles que cette sélection GARDE, dans un fichier. Rien n'est écrit.

    Un relecteur qui a composé « Europe ET étude de cohorte ET volailles » veut d'abord
    cette liste, et non l'appliquer au scénario : appliquer exclut le reste du corpus et
    se cumule, alors qu'un export ne touche à rien. Les deux chemins partent du MÊME plan
    (`_build_plan`), donc le fichier contient exactement ce que l'aperçu annonce, et le
    fichier porte sa propre définition dans `coverage` : une liste d'articles qui circule
    sans dire de quelle sélection elle vient n'est pas une donnée, c'est une rumeur."""
    from .exports import articles_by_ids, articles_export_response

    _get_user_scenario_or_404(scenario_id)
    plan = _build_plan(scenario_id, payload)
    ids = [int(i) for i in plan["keep_ids"]]
    articles = articles_by_ids(scenario_id, ids) if ids else []
    undecided = int(plan.get("undecided") or 0)
    coverage = (f"Articles gardés par la sélection : {plan['kept_phrase'] or 'tous'}. "
                f"{len(articles)} article(s) sur les {plan['relevant']} pertinents"
                + (f", dont {undecided} que la sélection ne sait pas juger (gardés)."
                   if undecided else ".")
                + " Aucun découpage n'a été appliqué au scénario.")
    return articles_export_response(
        scenario_id, format, articles, include_abstract,
        subset="selection", subset_label="selection",
        coverage=coverage,
        extra_meta={"selection": plan["kept_phrase"], "exclusion_reason": plan["reason"],
                    "kept": len(articles),
                    "relevant": plan["relevant"], "undecided": undecided,
                    "combine": str(payload.get("combine") or "all"),
                    "applied": False})


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
    reason = str(payload.get("reason") or plan["reason"])
    if not reason.startswith(SUBSET_REASON_PREFIX):
        reason = f"{SUBSET_REASON_PREFIX} {reason}"
    # Un motif écrit par l'appelant porte lui aussi la frontière, en tête comme les
    # autres : sans elle, ce découpage disparaissait de `judged_above_threshold`.
    if scope_threshold_of(reason) is None:
        reason = (f"{SUBSET_REASON_PREFIX} (seuil {float(plan['meta']['threshold']):.4g}) "
                  + reason[len(SUBSET_REASON_PREFIX):].lstrip())
    # Coupé APRÈS la mise en forme : le seuil est en tête, la coupe ne peut plus l'emporter.
    reason = reason[:480]
    with engine.begin() as conn:
        # L'état ANTÉRIEUR est conservé avant d'être recouvert, et seulement s'il ne
        # l'est pas déjà : deux découpages successifs ne doivent pas faire oublier ce
        # qu'il y avait avant le premier. `undo` écrivait littéralement 'pending' et
        # NULL, si bien qu'un article rescapé à la main puis exclu par un découpage
        # revenait « en attente, sans motif » : sous le seuil et sans inclusion, il
        # quittait le corpus pertinent, et le travail du relecteur avec lui.
        n = conn.execute(text("""
            UPDATE article_scenarios
            SET prior_screening_status = COALESCE(prior_screening_status, screening_status),
                prior_screening_reason = COALESCE(prior_screening_reason, screening_reason),
                screening_status = 'excluded',
                screening_reason = :reason,
                screened_at = NOW()
            WHERE scenario_id = :sid AND document_id = ANY(:ids)
        """), {"sid": scenario_id, "ids": ids, "reason": reason}).rowcount or 0
        _reset_corpus_caches(conn, scenario_id)
    logger.info(f"subset apply {scenario_id}: {n} articles excluded ({reason}); "
                f"{plan['kept_manual']} inclusion(s) manuelle(s) laissée(s) en place")
    return {**_public(plan), "applied": int(n), "status": "applied", "reason": reason,
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
        # On RESTAURE l'état d'avant le découpage, au lieu d'écrire 'pending' : une
        # inclusion manuelle recouverte par une restriction de portée revient
        # inclusion, avec son motif. À défaut d'état conservé (découpage appliqué par
        # une version antérieure), on retombe sur 'pending', l'ancien comportement.
        n = conn.execute(text(f"""
            UPDATE article_scenarios
            SET screening_status = COALESCE(prior_screening_status, 'pending'),
                screening_reason = prior_screening_reason,
                prior_screening_status = NULL,
                prior_screening_reason = NULL,
                screened_at = NOW()
            WHERE scenario_id = :sid AND screening_status = 'excluded'
              AND screening_reason LIKE :pfx{extra}
        """), params).rowcount or 0
        _reset_corpus_caches(conn, scenario_id)
    logger.info(f"subset undo {scenario_id}: {n} articles restored")
    return {"scenario_id": scenario_id, "restored": int(n),
            "status": "restored" if n else "noop"}
