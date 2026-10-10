"""Scenario lookups shared by every domain: existence, threshold, display name.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

from typing import Any, Optional

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
                   sub_queries, combinator, kind, created_ip
            FROM user_scenarios WHERE id = :id
        """), {"id": scenario_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Scénario utilisateur '{scenario_id}' non trouvé")
    return dict(row)

DEFAULT_SIMILARITY_THRESHOLD = 0.45

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


def _get_scenario_threshold(scenario_id: str) -> float:
    """Retourne le seuil de similarité configuré pour ce scénario."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT similarity_threshold FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    return float(row["similarity_threshold"]) if row and row["similarity_threshold"] is not None else DEFAULT_SIMILARITY_THRESHOLD


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
    article non encore jugé n'est pas un article jugé mauvais."""
    status = screening_status_sql(doc, link)
    rthr = scenario_rerank_threshold_sql(f"{link}.scenario_id")
    return (f"{doc}.is_duplicate IS NOT TRUE"
            f" AND {status} IS DISTINCT FROM 'excluded'"
            f" AND ({status} = 'included' OR ("
            f"COALESCE({link}.similarity_score, 0) >= {thr}"
            f" AND ({link}.rerank_score IS NULL OR {link}.rerank_score >= {rthr})))")


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
    comme sur l'écran. Sans seuil, l'ordre est le même, sans cette première coupe."""
    status = screening_status_sql(doc, link)
    head = (f"CASE WHEN COALESCE({link}.similarity_score, 0) >= {thr} THEN 0 ELSE 1 END ASC, "
            if thr else "")
    return (head
            + f"({status} = 'included') DESC, "
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
    """La portée des étapes d'enrichissement DU PIPELINE, réglée par l'environnement.

    `all` par défaut : le comportement d'aujourd'hui, inchangé. La carte (les faits
    par article, mis en cache sur la ligne) est la moitié sur laquelle repose la règle
    « toute extraction lit tous les articles pertinents » : la réduire au sous-ensemble
    pertinent du moment économise beaucoup, et laisse sans faits les articles qu'un
    seuil abaissé rendra pertinents plus tard. Le digest dit déjà combien d'articles
    pertinents portent un PICO, et le panneau d'enrichissement dit ce qu'il reste à
    faire, donc le manque se voit et se rattrape ; mais c'est un choix, pas un défaut.

    `PIPELINE_ENRICH_SCOPE=relevant` pour l'économie, comme les plafonds d'articles
    sont un repli opérationnel pour un jour où le budget doit être tenu."""
    import os
    value = (os.getenv("PIPELINE_ENRICH_SCOPE") or "all").strip().lower()
    return value if value in SCOPES else "all"


#: Les deux modes d'une recherche. `standard` : au plus LIVE_MAX_PER_SOURCE notices par
#: source, dans l'ordre de pertinence de la source. `exhaustive` : toutes les notices que
#: chaque base appariant le booléen renvoie (cf. _run_user_scenario_populate).
SEARCH_MODES = ("standard", "exhaustive")


def scenario_search_mode(scenario_id: str) -> str:
    """Le mode de recherche ENREGISTRÉ sur le scénario, `standard` par défaut.

    Jamais d'erreur : une base sans la colonne (antérieure à elle) est en mode standard,
    c'est-à-dire le comportement qu'elle a toujours eu."""
    try:
        with engine.connect() as conn:
            value = conn.execute(text("SELECT search_mode FROM user_scenarios WHERE id = :id"),
                                 {"id": scenario_id}).scalar()
    except Exception:                                    # noqa: BLE001
        return "standard"
    value = str(value or "").strip().lower()
    return value if value in SEARCH_MODES else "standard"


def set_scenario_search_mode(scenario_id: str, mode: str) -> None:
    """Enregistre le mode sur le scénario, pour qu'une relance refasse la même recherche."""
    mode = str(mode or "").strip().lower()
    if mode not in SEARCH_MODES:
        raise ValueError(f"Mode de recherche inconnu : {mode!r} (attendu : {', '.join(SEARCH_MODES)})")
    with engine.begin() as conn:
        conn.execute(text("UPDATE user_scenarios SET search_mode = :m WHERE id = :id"),
                     {"m": mode, "id": scenario_id})


def scenario_threshold_sql(sid: str = ":sid") -> str:
    """Le seuil du scénario, LU DANS la requête plutôt que passé en paramètre, pour
    que le lot et le compteur qui l'annonce voient le même instantané."""
    return (f"COALESCE((SELECT ss.similarity_threshold FROM scenario_settings ss"
            f" WHERE ss.scenario_id = {sid}), {DEFAULT_SIMILARITY_THRESHOLD})")


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
    NULL → le seuil enregistré du scénario, à défaut 0.45)."""
    status = screening_status_sql("d", "ars")
    fulltext = ("EXISTS (SELECT 1 FROM document_chunk c"
                " WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section')")
    chunkless = "NOT EXISTS (SELECT 1 FROM document_chunk c WHERE c.document_id = d.id)"
    in_range = "d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int"
    return f"""
        WITH s AS (
            SELECT COALESCE(CAST(:thr AS double precision),
                            (SELECT similarity_threshold FROM scenario_settings
                              WHERE scenario_id = :sid),
                            {DEFAULT_SIMILARITY_THRESHOLD}) AS thr,
                   (SELECT created_at FROM user_scenarios WHERE id = :sid) AS screated
        )
        SELECT
            MIN(s.thr)                                                        AS threshold,
            COUNT(*)                                                          AS total,
            COUNT(*) FILTER (WHERE ars.similarity_score >= s.thr)             AS above_threshold,
            COUNT(*) FILTER (WHERE ars.similarity_score IS NOT NULL
                               AND ars.similarity_score < s.thr)              AS below_threshold,
            COUNT(*) FILTER (WHERE ars.similarity_score IS NULL)              AS unscored,
            COUNT(*) FILTER (WHERE ars.similarity_score IS NOT NULL)          AS scored,
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


def corpus_search_sql(param: str, doc: str = "d") -> str:
    """Le prédicat SQL d'UN terme de recherche dans le corpus, pour un WHERE.

    `param` est le nom du paramètre lié qui porte le motif (déjà en minuscules, accents
    repliés, encadré de %). Pur : aucune connexion, testable hors base."""
    fields = (f"{doc}.title", f"{doc}.abstract", f"{doc}.authors", f"{doc}.journal",
              f"{doc}.keywords", f"{doc}.doi", f"{doc}.pmid")
    folded = " || ' ' || ".join(f"COALESCE({f}, '')" for f in fields)
    return (f"translate(lower({folded}), '{_ACCENT_FROM}', '{_ACCENT_TO}')"
            f" LIKE :{param}")


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
# peut-être validé. Il porte sa propre empreinte (seuil + identifiants des articles) et
# s'annonce périmé ; c'est ce qu'on veut : dire, pas détruire.
CORPUS_DERIVED_CACHE_RESET = """
    clustering_json = NULL, clustering_generated_at = NULL,
    knowledge_graph_json = NULL, kg_generated_at = NULL,
    concept_graph_json = NULL, concept_graph_generated_at = NULL,
    recommended_actions_json = NULL, recommended_actions_lang = NULL,
    actions_generated_at = NULL,
    seir_projection_json = NULL, seir_projection_generated_at = NULL
"""
