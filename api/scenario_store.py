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


# ── Le périmètre d'un traitement par lot sur un scénario ─────────────────────
# Un enrichissement coûte un appel de modèle PAR ARTICLE. Sur un scénario de six mille
# cinq cents références dont quatre cent soixante-sept passent le seuil, le lancer sur
# tout le scénario coûte quatorze fois le lot utile, pour la même réponse. D'où un
# périmètre explicite, et un seul endroit qui l'écrit.
SCOPES = ("all", "relevant")


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
    status = f"COALESCE({link}.screening_status, {doc}.screening_status)"
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
    status = "COALESCE(ars.screening_status, d.screening_status)"
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
CORPUS_DERIVED_CACHE_RESET = """
    clustering_json = NULL, clustering_generated_at = NULL,
    knowledge_graph_json = NULL, kg_generated_at = NULL,
    concept_graph_json = NULL, concept_graph_generated_at = NULL,
    recommended_actions_json = NULL, recommended_actions_lang = NULL,
    actions_generated_at = NULL
"""
