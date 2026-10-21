"""Scenario lookups shared by every domain: existence, threshold, display name.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Optional

from fastapi import HTTPException
from sqlalchemy import text

from .core import engine, logger

# ── Helpers internes ──────────────────────────────────────────────────────────

def _get_user_scenario_or_404(scenario_id: str) -> dict[str, Any]:
    """Retourne la ligne user_scenarios ou lève 404."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, name, query, mode, filters, result_count, pinned, folder_id, created_at, updated_at,
                   search_strategy, populate_status, pipeline_status, pipeline_step,
                   pipeline_progress, pipeline_started_at, article_count, is_system,
                   sub_queries, combinator, kind, created_ip
            FROM user_scenarios WHERE id = :id
        """), {"id": scenario_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Scénario utilisateur '{scenario_id}' non trouvé")
    return dict(row)

#: Le seuil de DÉPART, celui qu'on applique tant que personne n'en a choisi un. Rien ne
#: l'écrit en base : il est lu ici au moment de servir, de sorte qu'un changement de
#: cette constante atteint tous les scénarios auxquels personne n'a touché.
#:
#: Ce n'est pas une valeur défendable en soi, et le code le mesure lui-même (ci-dessous) :
#: sur le corpus HPAI, les articles hors sujet ont une similarité médiane PLUS HAUTE que
#: les articles aviaires, si bien qu'aucune coupe fixe du cosinus ne tient d'un sujet à
#: l'autre. D'où `threshold_chosen`, que les rapports et les exports impriment : « 0.45,
#: par défaut, non choisi » ne se lit pas comme un choix de méthode.
DEFAULT_SIMILARITY_THRESHOLD = 0.45

#: La valeur que la DDL écrivait d'office (`similarity_threshold FLOAT DEFAULT 0.45`)
#: dans TOUTE ligne de réglages, créée pour n'importe quelle raison : un cache de
#: visualisation, un seuil de rerank, un découpage. Une ligne antérieure à
#: `similarity_threshold_set_at` qui porte cette valeur ne dit donc rien d'un choix. Figée
#: ici, et non lue sur DEFAULT_SIMILARITY_THRESHOLD : c'est l'histoire de la base, pas le
#: réglage d'aujourd'hui.
_LEGACY_DDL_THRESHOLD = 0.45

# Le rerank juge la PERTINENCE À LA QUESTION, le cosinus la proximité de vocabulaire, et
# sur un corpus thématique les deux ne disent pas du tout la même chose. Mesuré sur le
# scénario HPAI (602 pertinents) : les articles hors sujet (COVID, dengue, Ebola) ont une
# similarité MÉDIANE PLUS HAUTE que les articles aviaires, 0.398 contre 0.361, si bien que
# monter le curseur de similarité jette d'abord ce qu'on voulait garder. Le rerank, lui,
# les sépare : 0.588 contre 0.070. Un second seuil était donc le seul moyen de nettoyer un
# corpus sans le mutiler, et il ne remplace pas le premier, il s'y ajoute.
#
# Par défaut 0 : la porte se comporte alors EXACTEMENT comme avant, puisque tout score de
# rerank est positif. Rien ne change pour un scénario existant tant que personne n'y touche.
DEFAULT_RERANK_THRESHOLD = 0.0


def threshold_chosen_sql(ss: str = "ss") -> str:
    """Vrai quand une PERSONNE a choisi le seuil de similarité de cette ligne de réglages.

    `ss` est l'alias de scenario_settings. Choisi veut dire : posé par le PATCH du seuil
    (`similarity_threshold_set_at` renseigné), ou, pour une ligne d'avant cette colonne,
    une valeur stockée autre que celle que la DDL écrivait d'office. Pur : la règle est
    écrite ICI, une fois, et lue aussi bien par le SQL que par `scenario_threshold_info`."""
    return (f"({ss}.similarity_threshold IS NOT NULL"
            f" AND ({ss}.similarity_threshold_set_at IS NOT NULL"
            f" OR {ss}.similarity_threshold <> {_LEGACY_DDL_THRESHOLD}))")


def scenario_threshold_chosen_sql(sid: str = ":sid") -> str:
    """Le seuil de ce scénario a-t-il été choisi ? Lu DANS la requête, faux sans ligne."""
    return (f"COALESCE((SELECT {threshold_chosen_sql('ss')} FROM scenario_settings ss"
            f" WHERE ss.scenario_id = {sid}), FALSE)")


def scenario_threshold_info(scenario_id: str) -> dict[str, Any]:
    """Le seuil de similarité EN VIGUEUR et s'il a été choisi, en une lecture.

    Toute lecture du seuil passe par ici (ou par `scenario_threshold_sql`, qui applique la
    même règle dans une requête) : un seuil jamais choisi vaut DEFAULT_SIMILARITY_THRESHOLD,
    quoi que la ligne de réglages porte. Le repli silencieux sur un littéral 0.45, recopié
    dans quatre modules, ne se distinguait pas d'un choix."""
    with engine.connect() as conn:
        row = conn.execute(text(f"""
            SELECT {scenario_threshold_sql(':sid')} AS thr,
                   {scenario_threshold_chosen_sql(':sid')} AS chosen,
                   (SELECT similarity_threshold_set_at FROM scenario_settings
                     WHERE scenario_id = :sid) AS set_at
        """), {"sid": scenario_id}).mappings().first()
    thr = row["thr"] if row else None
    return {
        "threshold": float(thr) if thr is not None else DEFAULT_SIMILARITY_THRESHOLD,
        "chosen": bool(row["chosen"]) if row else False,
        "set_at": row["set_at"].isoformat() if row and row["set_at"] else None,
    }


def _get_scenario_threshold(scenario_id: str) -> float:
    """Retourne le seuil de similarité EN VIGUEUR pour ce scénario (cf. scenario_threshold_info)."""
    return scenario_threshold_info(scenario_id)["threshold"]


def _get_scenario_rerank_threshold(scenario_id: str) -> float:
    """Retourne le seuil de rerank configuré pour ce scénario (0 = aucun filtrage)."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT rerank_threshold FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    return float(row["rerank_threshold"]) if row and row["rerank_threshold"] is not None else DEFAULT_RERANK_THRESHOLD


def scenario_rerank_threshold_sql(sid: str = ":sid") -> str:
    """Le seuil de rerank, LU DANS la requête, comme celui de similarité.

    Passé en sous-requête corrélée plutôt qu'en paramètre lié pour que la porte reste UNE
    chaîne utilisable telle quelle par les quinze appelants : ajouter un `:rthr` aurait
    obligé chacun à le lier, et le premier oubli aurait fait diverger un compteur de son
    lot, ce que cette fonction existe précisément pour empêcher."""
    return (f"COALESCE((SELECT ss.rerank_threshold FROM scenario_settings ss"
            f" WHERE ss.scenario_id = {sid}), {DEFAULT_RERANK_THRESHOLD})")


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

#: La longueur sous laquelle un résumé n'en est pas un (« [No abstract available] »).
ABSTRACT_MIN_CHARS = 30


def has_abstract_sql(doc: str = "d") -> str:
    """Le prédicat « cette notice a un résumé », lu sur `doc` (literature_document).

    Évalué À LA LECTURE, jamais figé sur le lien : une notice dont le résumé est comblé
    plus tard (backfill Europe PMC, une autre source dans une recherche suivante)
    devient d'elle-même une notice avec résumé, éligible au score. Les notices sans
    résumé restent au corpus, signalées, et hors du sous-ensemble pertinent tant qu'un
    relecteur ne les inclut pas à la main (cf. relevant_gate_sql). Pur."""
    return f"({doc}.abstract IS NOT NULL AND length(TRIM({doc}.abstract)) >= {ABSTRACT_MIN_CHARS})"


def screening_status_sql(doc: str = "d", link: str = "ars") -> str:
    """Le statut de screening d'un article DANS CETTE REVUE.

    Il se lisait `COALESCE(ars.screening_status, d.screening_status)`, c'est-à-dire : à
    défaut de décision dans cette revue, la décision prise dans une AUTRE. Or
    `literature_document` est partagé par tous les scénarios qui contiennent l'article.
    Un relecteur excluait un article dans sa revue, et l'article quittait aussitôt le
    sous-ensemble pertinent de toutes les autres revues qui le contiennent : leur brief,
    leurs extractions, leurs exports et leurs compteurs, sans qu'un seul de leurs écrans
    ne le dise. Deux scénarios de production partagent une requête sur la grippe aviaire
    et se déplaçaient ainsi l'un l'autre.

    Les écritures sur la ligne globale ont été retirées, et les décisions déjà prises y
    ont été recopiées une fois sur les liens (`_backfill_ars_screening_from_document`),
    pour qu'aucune ne soit perdue. `doc` reste dans la signature : il documente de quelle
    table on NE lit plus, et garde les appelants symétriques de `relevant_gate_sql`."""
    return f"{link}.screening_status"


def reviewer_decision_sql(link: str = "ars") -> str:
    """Le prédicat « ce lien porte la décision d'un relecteur », lu sur `link`
    (article_scenarios) : inclus ou exclu, un vote de double aveugle, un arbitrage.

    Une décision est un travail humain. Une recherche relancée supprimait les liens que sa
    requête n'appariait plus, décision, motif et votes compris : l'inclusion à la main
    disparaissait du sous-ensemble pertinent (« au-dessus du seuil, plus ce qu'un relecteur
    a inclus »), et l'article revenait « en attente » si une recherche suivante le
    retrouvait. Aucune recherche ne supprime plus un tel lien (`_set_scenario_corpus`).
    Jamais NULL, pour qu'un `NOT` s'y applique tel quel (un statut NULL ferait sinon
    garder tous les liens jamais jugés). Pur."""
    return (f"(COALESCE({link}.screening_status IN ('included', 'excluded'), FALSE)"
            f" OR {link}.reviewer_1_status IS NOT NULL"
            f" OR {link}.reviewer_2_status IS NOT NULL"
            f" OR {link}.kappa_final_status IS NOT NULL)")


def added_by_hand_sql(link: str = "ars") -> str:
    """The predicate "this link was added by hand" (a file uploaded by a reviewer,
    api/additions.py), read on `link` (article_scenarios) through `scenario_additions`.
    Like a reviewer's decision, an addition is human work: no search deletes such a link,
    and the PRISMA card counts it on its own line. Never NULL. Pure."""
    return (f"EXISTS (SELECT 1 FROM scenario_additions sa"
            f" WHERE sa.scenario_id = {link}.scenario_id AND sa.document_id = {link}.document_id)")


def relevant_gate_sql(doc: str = "d", link: str = "ars", thr: str = ":thr") -> str:
    """Le prédicat SQL du sous-ensemble pertinent, à mettre dans un WHERE.

    `doc` et `link` sont les alias de literature_document et article_scenarios ; `thr` le
    paramètre lié qui porte le seuil de similarité. Pur : aucune connexion, testable hors
    base.

    Le seuil de rerank s'y ajoute, lu dans la requête (`scenario_rerank_threshold_sql`).
    Les deux scores ne sont PAS traités pareil, et l'asymétrie est voulue :

      - pas de score de similarité vaut 0, donc dehors dès que le seuil est non nul ;
      - pas de score de RERANK ne vaut rien du tout, donc l'article reste.

    Parce que le rerank est calculé après coup et par lots : sur le scénario HPAI, 45 des
    602 pertinents n'en avaient pas encore. Le compter pour 0 aurait fait disparaître ces
    45 articles à la seconde où quelqu'un bouge le curseur, sans que rien ne le dise. Un
    article non encore jugé n'est pas un article jugé mauvais.

    Une notice SANS résumé (has_abstract_sql) n'est jamais pertinente par son score, qui
    ne lit que son titre : elle ne l'est que si un relecteur l'inclut à la main."""
    status = screening_status_sql(doc, link)
    rthr = scenario_rerank_threshold_sql(f"{link}.scenario_id")
    return (f"{doc}.is_duplicate IS NOT TRUE"
            f" AND {status} IS DISTINCT FROM 'excluded'"
            f" AND ({status} = 'included' OR ("
            f"COALESCE({link}.similarity_score, 0) >= {thr}"
            f" AND ({link}.rerank_score IS NULL OR {link}.rerank_score >= {rthr})"
            f" AND {has_abstract_sql(doc)}))")


#: La première clause de la porte, celle que `relevant_gate_tail_sql` retire. Épinglée
#: ici parce que la queue est obtenue en coupant sur le premier « AND » : si la clause
#: des doublons cessait d'arriver en tête, la coupe emporterait autre chose.
_GATE_FIRST_CLAUSE = "is_duplicate IS NOT TRUE"


def relevance_order_sql(doc: str = "d", link: str = "ars", thr: str | None = None) -> str:
    """L'ORDRE DE PERTINENCE d'un corpus, écrit UNE fois.

    Il y en avait trois, pour la même question : l'onglet Corpus triait par
    « au-dessus du seuil, puis reranké, puis rerank, puis similarité, puis année » ;
    l'export relisait « inclus à la main, puis similarité, puis citations » sous une
    docstring promettant « le même ensemble et le même ordre que l'onglet Corpus » ; et
    l'export par identifiants en avait un troisième. Un relecteur qui compare son écran
    au fichier qu'il vient de télécharger ne retrouvait pas ses dix premiers articles.

    `thr` : quand un seuil est donné, les articles qui le passent viennent d'abord,
    comme sur l'écran. Sans seuil, l'ordre est le même, sans cette première coupe.

    `IS TRUE` : un article jamais jugé a un statut NULL, et `NULL = 'included'` vaut NULL,
    que PostgreSQL range EN TÊTE d'un tri décroissant. Les inclusions à la main passaient
    donc après tous les articles jamais jugés, c'est-à-dire presque tout le corpus, au lieu
    d'ouvrir la liste."""
    status = screening_status_sql(doc, link)
    head = (f"CASE WHEN COALESCE({link}.similarity_score, 0) >= {thr} THEN 0 ELSE 1 END ASC, "
            if thr else "")
    return (head
            + f"({status} = 'included') IS TRUE DESC, "
            + f"({link}.rerank_score IS NOT NULL) DESC, "
            + f"{link}.rerank_score DESC NULLS LAST, "
            + f"{link}.similarity_score DESC NULLS LAST, "
            + f"{doc}.year DESC NULLS LAST, "
            + f"{doc}.citation_count DESC NULLS LAST, "
            + f"{doc}.id")


def relevant_gate_tail_sql(doc: str = "d", link: str = "ars", thr: str = ":thr") -> str:
    """La porte SANS sa clause sur les doublons, pour les WHERE qui l'écrivent déjà.

    Treize des requêtes converties portaient la porte en deux morceaux : la clause des
    doublons en haut du WHERE, parmi les prédicats de contexte, et le reste plus bas.
    Leur donner la porte entière ajouterait un prédicat, et je ne veux pas qu'une
    unification change le SQL qui part en base le jour où elle est faite : ce qui doit
    changer, c'est où la condition est ÉCRITE, pas ce qu'elle dit.

    Le reste de leur WHERE exclut déjà les doublons, donc le lot est le même ; et la
    clause ajoutée demain à la porte les atteindra, elles aussi."""
    full = relevant_gate_sql(doc, link, thr)
    head, _, tail = full.partition(" AND ")
    assert _GATE_FIRST_CLAUSE in head, head
    return tail


# ── La nature d'une question, et ce qu'elle rend disponible ──────────────────
# Toute question n'appelle pas un modèle. Beaucoup se terminent par une synthèse :
# ce que la littérature établit, avec quelle certitude, et ce qui manque. Pour
# celles-là, la moitié prédictive de l'application n'est pas seulement du décor,
# elle coûte un passage de modèle par scénario qui fabrique des variables
# candidates que personne n'ajustera jamais.
#
# La coupe n'est pas symétrique : la revue est un sous-ensemble strict sur lequel la
# prédiction est bâtie. Une porte n'a donc jamais qu'à RETIRER la moitié prédictive,
# jamais à retirer quoi que ce soit à un scénario prédictif. D'où deux natures, une
# seule colonne, et NULL qui vaut « tout », c'est-à-dire exactement ce qui existait.
KIND_REVIEW = "review"
KIND_PREDICTIVE = "predictive"
KINDS = (KIND_REVIEW, KIND_PREDICTIVE)
DEFAULT_KIND = KIND_PREDICTIVE          # NULL en base : rien ne change pour l'existant

# Les deux seules capacités qui RETIRENT quelque chose. Tout le reste (corpus,
# seuil, screening, PICO, concepts, synthèse, graphes, assistant, questions,
# exports, revue vivante, enrichissement) appartient aux deux natures.
CAP_MODEL = "model_spec"                # variables prédictives, spec, entraînement, SEIR
CAPABILITIES = (CAP_MODEL,)

# Il y en avait une deuxième, `field_data`, qui retirait l'onglet des rapports de
# situation. Retirée : la littérature grise de ReliefWeb EST une source de
# littérature, qu'une revue humanitaire veut lire, et la capacité ne gardait rien
# côté serveur, si bien que marquer un scénario « revue » retirait une source de
# l'écran pendant que l'ingestion continuait. Une capacité qui cache sans garder
# n'est pas une porte, c'est un oubli.

_CAPABILITIES_BY_KIND = {
    KIND_REVIEW: frozenset(),
    KIND_PREDICTIVE: frozenset(CAPABILITIES),
}

# Délibérément HORS de CAP_MODEL : les paramètres épidémiologiques mis en commun
# (api/variables.py). Ils se lisent sans modèle, sans appel LLM, sur tout le corpus
# pertinent, et un R0 pondéré par la qualité avec la provenance de chaque étude EST
# un produit de revue. Les ranger avec la prévision les retirerait à une
# méta-analyse de paramètres, qui est précisément une revue.


def normalise_kind(value: Optional[str]) -> str:
    """La nature d'un scénario, NULL et inconnu valant l'ancien comportement."""
    text_value = (value or "").strip().lower()
    return text_value if text_value in KINDS else DEFAULT_KIND


def capabilities_for(kind: Optional[str]) -> frozenset:
    return _CAPABILITIES_BY_KIND[normalise_kind(kind)]


def kind_has(kind: Optional[str], capability: str) -> bool:
    """La seule question que le code pose : ce scénario fait-il cela ?"""
    if capability not in CAPABILITIES:
        raise ValueError(f"capacité inconnue : {capability!r}")
    return capability in capabilities_for(kind)


def scenario_kind(scenario_id: str, conn=None) -> str:
    """La nature enregistrée du scénario. Un scénario absent vaut l'ancien
    comportement : une porte ne doit pas être la première à signaler un 404."""
    sql = text("SELECT kind FROM user_scenarios WHERE id = :sid")
    if conn is not None:
        row = conn.execute(sql, {"sid": scenario_id}).mappings().first()
    else:
        with engine.connect() as c:
            row = c.execute(sql, {"sid": scenario_id}).mappings().first()
    return normalise_kind(row["kind"] if row else None)


def scenario_can(scenario_id: str, capability: str, conn=None) -> bool:
    return kind_has(scenario_kind(scenario_id, conn=conn), capability)


def capability_refusal(scenario_id: str, capability: str) -> dict[str, Any]:
    """La forme d'un refus d'applicabilité.

    L'application en a déjà une (cf. api/seir.py) : `applicable: false`, un
    `reason_code` stable que l'interface traduit, et une phrase lisible. On ne lui en
    ajoute pas une deuxième. Ce n'est pas une erreur : la question posée n'appelle pas
    cette moitié de l'outil, et le dire par un 4xx ferait croire à une panne."""
    return {
        "applicable": False,
        "scenario_id": scenario_id,
        "capability": capability,
        "kind": KIND_REVIEW,
        "reason_code": "review_scenario",
        "reason": ("Ce scénario est une revue de littérature : la moitié prédictive "
                   "(variables candidates, spécification de modèle, entraînement, "
                   "projection) n'y est pas activée. Changez sa nature pour l'ouvrir ; "
                   "rien de ce qui a déjà été produit n'est supprimé."),
    }


# ── Le périmètre d'un traitement par lot sur un scénario ─────────────────────
# Un enrichissement coûte un appel de modèle PAR ARTICLE. Sur un scénario de six mille
# cinq cents références dont quatre cent soixante-sept passent le seuil, le lancer sur
# tout le scénario coûte quatorze fois le lot utile, pour la même réponse. D'où un
# périmètre explicite, et un seul endroit qui l'écrit.
SCOPES = ("all", "relevant")


def pipeline_enrich_scope() -> str:
    """La portée des étapes d'enrichissement DU PIPELINE (texte intégral, PICO,
    métadonnées), réglée par l'environnement.

    `relevant` par défaut depuis le 12 octobre 2026 (décision d'Erol) : le pipeline
    enchaîné après une recherche enrichissait TOUT le scénario (HPAI_last : 17 510
    articles, dont 1 075 pertinents), et le texte intégral, le PICO et les métadonnées des
    articles hors du sous-ensemble ne servaient à aucune extraction. La règle « toute
    extraction lit tous les articles pertinents » tient toujours : un article que le
    seuil rend pertinent plus tard reçoit son PICO du travail de fond (relevant
    seulement, lui aussi), et le panneau d'enrichissement dit ce qui reste à faire, sur
    l'une ou l'autre portée.

    `PIPELINE_ENRICH_SCOPE=all` pour tout le scénario. Une valeur illisible garde la
    portée par défaut."""
    import os
    value = (os.getenv("PIPELINE_ENRICH_SCOPE") or "relevant").strip().lower()
    return value if value in SCOPES else "relevant"


def set_scenario_exclude_preprints(scenario_id: str, exclude: bool) -> None:
    """Enregistre l'exclusion des préprints dans les filtres du scénario, pour qu'une
    relance refasse la même recherche. La clé n'est écrite que quand elle est vraie : une
    recherche avec préprints garde des filtres identiques à ceux d'avant cette option, et
    donc sa clé dans le cache des sources."""
    with engine.begin() as conn:
        if exclude:
            conn.execute(text(
                "UPDATE user_scenarios SET filters = (CASE WHEN jsonb_typeof(filters) = 'object' "
                "THEN filters ELSE '{}'::jsonb END) || jsonb_build_object('exclude_preprints', true) "
                "WHERE id = :id"), {"id": scenario_id})
        else:
            conn.execute(text(
                "UPDATE user_scenarios SET filters = filters - 'exclude_preprints' "
                "WHERE id = :id AND jsonb_typeof(filters) = 'object'"), {"id": scenario_id})


def scenario_threshold_sql(sid: str = ":sid") -> str:
    """Le seuil du scénario, LU DANS la requête plutôt que passé en paramètre, pour
    que le lot et le compteur qui l'annonce voient le même instantané.

    Même règle que `scenario_threshold_info` : une valeur stockée sans avoir été choisie
    (`threshold_chosen_sql`) ne compte pas, c'est le seuil de départ qui s'applique."""
    return (f"COALESCE((SELECT CASE WHEN {threshold_chosen_sql('ss')} THEN ss.similarity_threshold END"
            f" FROM scenario_settings ss WHERE ss.scenario_id = {sid}),"
            f" {DEFAULT_SIMILARITY_THRESHOLD})")


def scenario_scope_sql(scope: str, doc: str = "ld", link: str = "asn",
                       sid: str = ":sid") -> str:
    """Le prédicat d'un lot sur un scénario, à mettre dans un WHERE après la jointure.

    `all` : tout le scénario, hors doublons et hors articles écartés par un relecteur.
    Exclure ces deux-là n'est pas une restriction du périmètre, c'est la même règle que
    partout : on ne paie pas un modèle pour un doublon ni pour un article déjà écarté.

    `relevant` : le sous-ensemble pertinent, par la porte partagée ci-dessus. Les trois
    lots d'enrichissement filtraient chacun à leur façon, l'un excluant les articles
    écartés et les deux autres non."""
    if scope not in SCOPES:
        raise ValueError(f"portée inconnue : {scope!r} (attendu : {', '.join(SCOPES)})")
    if scope == "relevant":
        return relevant_gate_sql(doc, link, scenario_threshold_sql(sid))
    status = screening_status_sql(doc, link)
    return f"{doc}.is_duplicate IS NOT TRUE AND {status} IS DISTINCT FROM 'excluded'"


# ── Les compteurs d'articles, comptés UNE fois ───────────────────────────────
# « Combien d'articles ? » recevait des réponses différentes sur le même écran : 433
# dans le bandeau (/detail), 449 dans le titre du corpus (/corpus), 441 scorés sur 433
# (/embedding-status). Trois requêtes, trois connexions, trois instants - et en
# READ COMMITTED, deux instructions d'une MÊME connexion voient déjà deux instantanés
# différents, si bien que « scorés » pouvait dépasser « total » pendant que le pipeline
# écrivait.
#
# La règle : tous les compteurs du corpus viennent d'UNE SEULE instruction SQL. Une
# instruction, un instantané, par construction - quel que soit le niveau d'isolation et
# quoi qu'écrive le pipeline pendant ce temps. Les chiffres peuvent changer d'un appel à
# l'autre pendant une recherche, mais ils bougent ENSEMBLE et restent cohérents entre eux.
#
# Le seuil et la date de création du scénario sont lus DANS la même instruction : les
# passer en paramètres aurait rouvert la porte à deux lectures à deux instants.
def scenario_counts_sql() -> str:
    """L'instruction unique qui compte le corpus d'un scénario. Pure : aucune
    connexion, testable hors base. Paramètres liés : `sid`, et `thr` (seuil forcé,
    NULL → le seuil en vigueur du scénario, cf. scenario_threshold_sql)."""
    status = screening_status_sql("d", "ars")
    fulltext = ("EXISTS (SELECT 1 FROM document_chunk c"
                " WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section')")
    chunkless = "NOT EXISTS (SELECT 1 FROM document_chunk c WHERE c.document_id = d.id)"
    in_range = "d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int"
    # Les notices sans résumé forment leur propre case : ni au-dessus, ni en dessous du
    # seuil, ni « non scorées » (leur score, s'il existe, ne lit que le titre et n'ouvre
    # pas le sous-ensemble pertinent). above + below + unscored + no_abstract = total.
    has_abs = has_abstract_sql("d")
    return f"""
        WITH s AS (
            SELECT COALESCE(CAST(:thr AS double precision),
                            {scenario_threshold_sql(':sid')}) AS thr,
                   (SELECT created_at FROM user_scenarios WHERE id = :sid) AS screated
        )
        SELECT
            MIN(s.thr)                                                        AS threshold,
            COUNT(*)                                                          AS total,
            COUNT(*) FILTER (WHERE ars.similarity_score >= s.thr AND {has_abs})
                                                                              AS above_threshold,
            COUNT(*) FILTER (WHERE ars.similarity_score IS NOT NULL
                               AND ars.similarity_score < s.thr AND {has_abs}) AS below_threshold,
            COUNT(*) FILTER (WHERE ars.similarity_score IS NULL AND {has_abs}) AS unscored,
            COUNT(*) FILTER (WHERE ars.similarity_score IS NOT NULL AND {has_abs})
                                                                              AS scored,
            COUNT(*) FILTER (WHERE NOT {has_abs})                             AS no_abstract,
            COUNT(*) FILTER (WHERE d.year IS NULL)                            AS year_unknown,
            COUNT(*) FILTER (WHERE ars.rerank_score IS NOT NULL)              AS reranked,
            COUNT(*) FILTER (WHERE {relevant_gate_sql(doc='d', link='ars', thr='s.thr')})
                                                                              AS relevant,
            COUNT(*) FILTER (WHERE {status} = 'included')                     AS included,
            COUNT(*) FILTER (WHERE {status} = 'excluded')                     AS excluded,
            COUNT(*) FILTER (WHERE {status} IS DISTINCT FROM 'included'
                               AND {status} IS DISTINCT FROM 'excluded')      AS pending,
            COUNT(*) FILTER (WHERE {fulltext})                                AS with_fulltext,
            COUNT(*) FILTER (WHERE {chunkless})                               AS chunkless,
            COUNT(*) FILTER (WHERE s.screated IS NOT NULL
                               AND d.created_at >= s.screated)                AS newly_fetched,
            COUNT(*) FILTER (WHERE s.screated IS NULL
                                OR d.created_at < s.screated)                 AS from_local,
            COUNT(DISTINCT d.year) FILTER (WHERE {in_range})                  AS years_covered,
            COUNT(DISTINCT d.journal)                                         AS journals_count,
            MIN(d.year) FILTER (WHERE {in_range})                             AS year_min,
            MAX(d.year) FILTER (WHERE {in_range})                             AS year_max
        FROM article_scenarios ars
        JOIN literature_document d ON d.id = ars.document_id
        CROSS JOIN s
        WHERE ars.scenario_id = :sid
          AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
    """


_COUNT_KEYS = ("total", "above_threshold", "below_threshold", "unscored", "scored",
               "no_abstract", "year_unknown",
               "reranked", "relevant", "included", "excluded", "pending",
               "with_fulltext", "chunkless", "newly_fetched", "from_local",
               "years_covered", "journals_count")


def scenario_counts(scenario_id: str, threshold: float | None = None,
                    conn: Any = None) -> dict[str, Any]:
    """Tous les compteurs d'articles d'un scénario, d'un seul instantané.

    Le SEUL endroit où le corpus d'un scénario est compté. Tout panneau qui affiche
    un nombre d'articles lit ce dictionnaire ; aucun n'écrit son propre COUNT, sinon
    deux chiffres du même écran se remettent à diverger.
    """
    params = {"sid": scenario_id, "thr": float(threshold) if threshold is not None else None}
    sql = text(scenario_counts_sql())
    if conn is not None:
        row = conn.execute(sql, params).mappings().first()
    else:
        with engine.connect() as _c:
            row = _c.execute(sql, params).mappings().first()
    out: dict[str, Any] = {k: int(row[k] or 0) for k in _COUNT_KEYS} if row else {k: 0 for k in _COUNT_KEYS}
    out["threshold"] = float(row["threshold"]) if row and row["threshold"] is not None else (
        float(threshold) if threshold is not None else DEFAULT_SIMILARITY_THRESHOLD)
    out["year_min"] = int(row["year_min"]) if row and row["year_min"] is not None else None
    out["year_max"] = int(row["year_max"]) if row and row["year_max"] is not None else None
    return out


# ── Les articles que le scoring n'a pas (encore) atteints ────────────────────
def unscored_sql(doc: str = "d", link: str = "ars") -> str:
    """Un article SANS score de similarité. La porte le compte pour 0 : il reste hors du
    sous-ensemble pertinent tant qu'il n'est pas scoré, sauf inclusion à la main.

    UNE expression, que lisent le rapport, les exports, le brief et le re-scoring
    automatique. C'est ici, et seulement ici, qu'on l'étend pour sortir de ce compte une
    catégorie d'articles qui ne peut pas devenir pertinente par son score (les notices
    sans résumé, par exemple) : `doc` est l'alias de literature_document, prêt pour cela.
    La partition de `scenario_counts` (au-dessus + en dessous + non scorés = total) ne la
    lit pas, puisqu'elle doit rester une partition.

    Les notices sans résumé en sortent : gardées au corpus, elles n'entrent jamais par leur
    score (relevant_gate_sql), seulement par une inclusion à la main ; les compter comme
    « en attente d'un score » relançait le scoring automatique pour rien et gonflait le
    compte que lisent le rapport, les exports et le brief."""
    return f"({link}.similarity_score IS NULL AND {has_abstract_sql(doc)})"


def scenario_unscored_count(scenario_id: str, conn: Any = None) -> int:
    """Combien d'articles du corpus attendent un score pour pouvoir entrer : ni doublon, ni
    décidés à la main (un exclu ne rentrera pas, un inclus est déjà dedans)."""
    status = screening_status_sql("d", "ars")
    sql = text(f"""
        SELECT COUNT(*) FROM article_scenarios ars
        JOIN literature_document d ON d.id = ars.document_id
        WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
          AND {status} IS DISTINCT FROM 'excluded' AND {status} IS DISTINCT FROM 'included'
          AND {unscored_sql('d', 'ars')}
    """)
    if conn is not None:
        return int(conn.execute(sql, {"sid": scenario_id}).scalar() or 0)
    with engine.connect() as _c:
        return int(_c.execute(sql, {"sid": scenario_id}).scalar() or 0)


def threshold_provenance(scenario_id: str) -> dict[str, Any]:
    """Ce qu'un produit de la revue (brief, rapport, export) dit de son seuil : s'il a été
    choisi, et combien d'articles restent hors du sous-ensemble faute de score.

    Jamais d'erreur : une ligne de provenance qui ne peut pas être lue devient `None`
    (« inconnu »), elle ne fait pas tomber le document qu'elle accompagne."""
    try:
        return {"threshold_chosen": scenario_threshold_info(scenario_id)["chosen"],
                "unscored": scenario_unscored_count(scenario_id)}
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"threshold_provenance {scenario_id}: {e}")
        return {"threshold_chosen": None, "unscored": None}


# ── Chercher DANS un corpus ──────────────────────────────────────────────────
# Un corpus de plusieurs milliers d'articles se parcourt par pages de cent : filtrer
# la page affichée ne cherche que dans ces cent-là. La recherche est donc faite en base,
# sur les champs qu'un relecteur a en tête quand il cherche un article qu'il a vu passer :
# titre, résumé, auteurs, revue, mots-clés, DOI, PMID.
#
# Les termes sont CUMULATIFS : « dengue vaccine » ne renvoie que les articles qui portent
# les deux mots, dans n'importe quel ordre et n'importe quel champ. Une suite entre
# guillemets est cherchée telle quelle. unaccent n'étant pas garanti installé, la
# comparaison se fait en minuscules sur un texte dont les accents latins sont repliés,
# des deux côtés : « Lévy » se trouve en tapant « levy », et inversement.
_ACCENT_FROM = "àáâãäåçèéêëìíîïñòóôõöùúûüýÿ"
_ACCENT_TO = "aaaaaaceeeeiiiinooooouuuuyy"


def _corpus_haystack_sql(doc: str = "d") -> str:
    """Le texte où l'on cherche : les sept champs, en minuscules, accents repliés."""
    fields = (f"{doc}.title", f"{doc}.abstract", f"{doc}.authors", f"{doc}.journal",
              f"{doc}.keywords", f"{doc}.doi", f"{doc}.pmid")
    folded = " || ' ' || ".join(f"COALESCE({f}, '')" for f in fields)
    return f"translate(lower({folded}), '{_ACCENT_FROM}', '{_ACCENT_TO}')"


def corpus_search_sql(param: str, doc: str = "d") -> str:
    """Le prédicat SQL d'UN terme de recherche dans le corpus, pour un WHERE.

    `param` est le nom du paramètre lié qui porte le motif (déjà en minuscules, accents
    repliés, encadré de %). Pur : aucune connexion, testable hors base.

    À ne pas poser tel quel dans une jointure avec la bibliothèque entière : voir
    `corpus_search_ids_sql`, que l'onglet Corpus emploie."""
    return f"{_corpus_haystack_sql(doc)} LIKE :{param}"


def corpus_search_ids_sql(params: list[str], scope_where: str) -> str:
    """Les identifiants des articles DU SCÉNARIO qui portent tous les termes, en UNE
    instruction. Pur.

    Le prédicat d'un terme replie sept champs par ligne, résumé compris. Postgres le
    chiffre comme une comparaison ordinaire et juge un LIKE très sélectif : posé dans le
    WHERE de la jointure, il pouvait donc filtrer d'abord les 494 213 articles de la
    bibliothèque, puis joindre le scénario. Sur HPAI_last (15 660 articles), « poultry
    cats » avec « Pertinents seulement » prenait 208 secondes en production, et le même
    texte sans ce filtre 8 secondes, répété dans quatre requêtes.

    Ici le texte replié n'est calculé que dans une CTE MATERIALIZED, une barrière que
    l'optimiseur ne traverse pas : elle ne contient que les liens du scénario qui passent
    déjà les filtres bon marché (`scope_where` : doublons, années, source, porte de
    pertinence), une fois par ligne et non une fois par terme. Le LIKE ne s'applique
    qu'après. L'appelant pose ensuite `d.id = ANY(...)` dans ses requêtes.

    `scope_where` lit les alias `d` (literature_document) et `ars` (article_scenarios)
    et le paramètre `:sid` ; `params` nomme les paramètres liés qui portent les termes."""
    likes = " AND ".join(f"hay LIKE :{p}" for p in params) or "TRUE"
    return f"""
        WITH scope AS MATERIALIZED (
            SELECT d.id, {_corpus_haystack_sql("d")} AS hay
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
            WHERE {scope_where}
        )
        SELECT id FROM scope WHERE {likes}
    """


def corpus_search_terms(query: str, max_terms: int = 8) -> list[str]:
    """Les termes d'une recherche dans le corpus, prêts à être liés en paramètres.

    Une suite entre guillemets reste un seul terme ; sinon on coupe aux espaces. Le
    résultat est en minuscules, accents repliés, encadré de `%`. Pur."""
    import re as _re
    text_ = (query or "").strip()
    if not text_:
        return []
    terms: list[str] = []
    for phrase, word in _re.findall(r'"([^"]+)"|(\S+)', text_):
        term = (phrase or word).strip()
        if not term:
            continue
        folded = term.lower().translate(str.maketrans(_ACCENT_FROM, _ACCENT_TO))
        # LIKE : % et _ seraient des jokers ; \ est l'échappement par défaut.
        escaped = folded.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
        terms.append(f"%{escaped}%")
        if len(terms) >= max_terms:
            break
    return terms


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
#
# La PROJECTION SEIR en fait partie. Elle est une fonction des paramètres extraits, qui
# sont eux-mêmes extraits du sous-ensemble pertinent : bouger le seuil changeait le lot
# d'articles que l'extraction lit, et la projection en cache continuait d'être servie
# telle quelle. Son invalidation ne regardait que `variables_generated_at`, donc elle ne
# voyait jamais un changement de seuil. Le commentaire du code prétendait l'inverse.
#
# Le SPEC (`variables_json`) n'y est PAS : il coûte un passage de modèle complet, et
# l'effacer sur un mouvement de curseur ferait perdre un travail que le relecteur a
# peut-être validé. Il porte sa propre empreinte (seuil + identifiants de TOUS les
# articles pertinents, depuis ctx-v9 : elle ne couvrait que les 25 premiers), et la
# génération suivante ne le réutilise tel quel que si elle n'a pas bougé ; c'est ce qu'on
# veut : ne pas détruire.
# CE QUI EST EFFACÉ, ET CE QUI EST DATÉ. La règle tient en une phrase : on efface ce qui
# se refait pour rien, on DATE ce qui se paie.
#
# Tout ce qui précède était effacé, et c'était détruire. Mesuré en production : le
# dernier article d'une revue des partenaires a reçu son score de similarité (15 033 sur
# 15 034 → 15 034), la passe de scoring a invalidé « le corpus a bougé », et le
# clustering, le graphe, la carte des concepts et les actions recommandées ont disparu
# ensemble. Une douzaine d'appels de modèle détruits par UN article, deux fois en quatre
# jours, sur deux revues ; l'onglet Visualisation restait vide et rien ne disait pourquoi.
#
# C'est la même règle que le rapport narratif a déjà obtenue : un résultat peut être
# vieux, il ne doit jamais disparaître. Elle s'applique ici à tout ce qui coûte un appel,
# et la classe de coût est déjà déclarée, résultat par résultat, dans `api/outputs.py`.
FREE_CACHE_RESET = """
    knowledge_graph_json = NULL, kg_generated_at = NULL,
    seir_projection_json = NULL, seir_projection_generated_at = NULL,
    evidence_brief_json = NULL, evidence_brief_at = NULL
"""
# Ces trois-là ne coûtent rien à refaire (pgvector et du SQL), et leur onglet les
# recalcule tout seul en les trouvant absents : les effacer, c'est les rafraîchir.
#
# Le CLUSTERING (un résumé écrit par groupe), la CARTE DES CONCEPTS (une annotation par
# article) et les ACTIONS RECOMMANDÉES (un appel) sont gardés et datés : `corpus_moved_at`
# ci-dessous porte la date du dernier mouvement, le panneau de complétude les dit « plus
# vieux que le corpus », et leur propre onglet le dit aussi, sans quoi on servirait
# l'ancien corpus en le présentant comme l'actuel - ce qui serait pire que de l'effacer.
#
# `evidence_brief_llm_json` (le rapport narratif) et `variables_json` (la spec) ne sont
# dans aucune des deux listes : ils portent leur propre empreinte et se datent eux-mêmes.


def mark_corpus_moved(conn, scenario_id: str) -> None:
    """LE SOUS-ENSEMBLE PERTINENT A BOUGÉ. Dans la transaction de l'appelant : la date du
    mouvement est écrite, ce qui se refait pour rien est effacé, ce qui se paie est gardé.

    Appelée par les six chemins qui déplacent la porte : le seuil, le scoring sémantique,
    le cross-encoder, la reconstruction du corpus, un découpage et une décision de
    relecture. Un seul article qui change de côté suffit, et c'est juste : une carte
    calculée sur l'ensemble d'avant décrit l'ensemble d'avant.

    `corpus_moved_at` est la seule chose qui reste de l'ancien effacement total. Elle est
    nécessaire parce qu'aucune des dates que le panneau de complétude regardait ne bouge
    quand un score est écrit : ni la date du lien, ni celle de la décision, ni celle du
    seuil. Sans elle, un résultat payé serait servi comme neuf après un mouvement que
    personne n'a vu passer.

    La copie EN MÉMOIRE du dernier clustering est oubliée : `/clustering` la sert quand la
    base n'a rien, si bien qu'un cache vidé renaissait de là (la même résurrection que
    l'ancienne copie /tmp, cf. _persist_clustering_result). Sans ligne de réglages, il n'y
    a rien à dater : l'INSERT en crée une, sinon le mouvement serait perdu."""
    conn.execute(text(f"""
        INSERT INTO scenario_settings (scenario_id, corpus_moved_at, updated_at)
        VALUES (:sid, NOW(), NOW())
        ON CONFLICT (scenario_id) DO UPDATE
        SET corpus_moved_at = NOW(), {FREE_CACHE_RESET}
    """), {"sid": scenario_id})
    try:
        from .clustering import _forget_clustering_result   # lazy: clustering se charge après
        _forget_clustering_result(scenario_id)
    except Exception as e:                                   # noqa: BLE001 - jamais bloquant
        logger.warning(f"mark_corpus_moved {scenario_id}: {e}")


def corpus_moved_at(conn, scenario_id: str):
    """La date du dernier mouvement du sous-ensemble pertinent, ou None. Ce qu'un onglet
    compare à la date de son propre résultat pour savoir s'il sert du vieux."""
    row = conn.execute(text(
        "SELECT corpus_moved_at FROM scenario_settings WHERE scenario_id = :sid"
    ), {"sid": scenario_id}).mappings().first()
    return row["corpus_moved_at"] if row else None


def is_stale(built_at, moved_at) -> bool:
    """Un résultat construit AVANT le dernier mouvement du corpus. PURE, et tolérante aux
    deux fuseaux : les dates de cette base ont été posées par une dizaine de migrations
    sur quatre ans, et deux d'entre elles ne s'accordent pas (cf. `_naive` dans
    api/outputs.py, et le HTTP 500 que leur comparaison a coûté)."""
    def _n(t):
        if not isinstance(t, _dt.datetime):
            return None
        return t.replace(tzinfo=None) if t.tzinfo is not None else t

    b, m = _n(built_at), _n(moved_at)
    return bool(b and m and b < m)
