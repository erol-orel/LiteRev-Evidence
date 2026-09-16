"""Boolean and lexical search, facets, corpus membership, search strategy.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import Depends
from pydantic import BaseModel, Field
from sqlalchemy import text, bindparam

import lexical_search as _lex

from .core import app, engine, logger, require_api_key
from .documents import _strategy_is_degraded
from .scenario_store import _get_user_scenario_or_404

# ─────────────────────────────────────────────────────────────────────────────
# Search helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_where(filters: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if not filters:
        return "", {}

    # Normaliser project_context : gesica/geoai4ei/eva -> literev (migration)
    if filters.get("project_context") in ("gesica", "geoai4ei", "eva"):
        filters = {**filters, "project_context": "literev"}

    clauses: list[str] = []
    params: dict[str, Any] = {}

    field_map = {
        "source": "d.source",
        "source_type": "d.source_type",
        "disease_or_condition": "d.disease_or_condition",
        "scenario_type": "d.scenario_type",
        "geographic_scope": "d.geographic_scope",
        "evidence_category": "d.evidence_category",
        "project_context": "d.project_context",
    }

    for key, column in field_map.items():
        value = filters.get(key)
        if value not in (None, "", []):
            if key == "scenario_type":
                # Migration 1 (Way B) : filtrer par APPARTENANCE au scénario
                # (article_scenarios) et non par la colonne d'ingestion
                # d.scenario_type. Dernier prédicat encore en Way A ; on l'aligne
                # sur tous les autres compteurs/vues (corpus, stats, PRISMA, RAG…)
                # déjà basculés, sinon /search montre MOINS d'articles que le
                # corpus réel du scénario. Sous-requête corrélée sur d.id
                # (d = literature_document dans tous les appelants de _build_where).
                clauses.append(
                    "EXISTS (SELECT 1 FROM article_scenarios ars "
                    "WHERE ars.document_id = d.id AND ars.scenario_id = :scenario_type)"
                )
            else:
                clauses.append(f"{column} = :{key}")
            params[key] = value

    year_min = filters.get("year_min")
    year_max = filters.get("year_max")
    if year_min not in (None, ""):
        clauses.append("d.year >= :year_min")
        params["year_min"] = int(year_min)
    if year_max not in (None, ""):
        clauses.append("d.year <= :year_max")
        params["year_max"] = int(year_max)

    if not clauses:
        return "", {}

    return " AND " + " AND ".join(clauses), params

def _strip_field_tags(query: str) -> str:
    """Enlève les tags de champ PubMed ([MeSH Terms], [Title/Abstract], [tiab]…).
    Ils n'ont pas d'équivalent dans la base locale (on apparie titre+résumé+texte, soit
    l'équivalent de [Title/Abstract]) et, laissés en place, polluaient le parsing en
    devenant des termes REQUIS parasites (« titleabstract », « meshterms »)."""
    return re.sub(r"\[[^\]]*\]", " ", query or "")


# Conservés dans un terme : lettres (accentuées comprises — « cathéter » doit rester
# « cathéter », pas « cathter »), chiffres, '_', '-', espaces et '*'. Le reste est du
# bruit de ponctuation.
_TERM_JUNK_RE = re.compile(r"[^\w\-* ]")


def _clean_boolean_term(raw: str) -> str:
    """Normalise un terme ou une phrase : minuscules, ponctuation retirée.

    Une '*' FINALE est la troncature PubMed (forecast* = forecast, forecasting,
    forecasts…) : elle est GARDÉE en fin de terme, et chaque compilateur en fait ce
    qu'il sait faire (préfixe en plein texte, sous-chaîne en LIKE, retirée pour arXiv
    et S2). Toute autre '*' est du bruit."""
    t = _TERM_JUNK_RE.sub("", raw.lower())
    trunc = t.rstrip().endswith("*")
    t = t.replace("*", "").strip()
    return f"{t}*" if (t and trunc) else t


def _tokenize_boolean(query: str) -> list[tuple[str, str | None]]:
    """Découpe une requête booléenne (tags de champ retirés) en jetons :
    ('(' | ')' | 'AND' | 'OR' | 'NOT', None) ou ('TERM', phrase_normalisée)."""
    toks: list[tuple[str, str | None]] = []
    for m in re.finditer(r'"[^"]*"|[()]|[^\s()]+', _strip_field_tags(query)):
        raw = m.group(0)
        if raw in ("(", ")"):
            toks.append((raw, None))
            continue
        if raw.startswith('"') and raw.endswith('"'):
            phrase = _clean_boolean_term(raw[1:-1])
            if phrase:
                toks.append(("TERM", phrase))
            continue
        up = raw.upper()
        if up in ("AND", "OR", "NOT"):
            toks.append((up, None))
        elif raw == "-":
            toks.append(("NOT", None))
        else:
            term = _clean_boolean_term(raw)
            if term:
                toks.append(("TERM", term))
    return toks


def _parse_boolean_ast(tokens: list[tuple[str, str | None]]):
    """Analyse récursive-descendante → AST qui RESPECTE le groupement :
    ('and'|'or', [enfants]) | ('not', enfant) | ('term', phrase) | None.
    Précédence : OR délimite, AND (explicite OU implicite entre atomes adjacents) lie
    plus fort, NOT préfixe un atome, les parenthèses regroupent. Robuste aux
    parenthèses déséquilibrées (arrêt propre) — corrige l'ancien parseur plat qui
    transformait « (A OU B) ET (C OU D) » en « A ET C ET (B OU D) »."""
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else (None, None)

    def parse_atom():
        nonlocal pos
        t, val = peek()
        if t == "(":
            pos += 1
            node = parse_or()
            if peek()[0] == ")":
                pos += 1
            return node
        if t == "TERM":
            pos += 1
            return ("term", val)
        # ')' ou opérateur orphelin : on ne consomme pas '(' ni un TERM ici
        if t == ")":
            return None
        pos += 1            # jeton parasite (AND/OR/NOT sans opérande) → ignoré
        return None

    def parse_not():
        nonlocal pos
        if peek()[0] == "NOT":
            pos += 1
            child = parse_atom()
            return ("not", child) if child is not None else None
        return parse_atom()

    def parse_and():
        nonlocal pos
        nodes = [parse_not()]
        while True:
            t = peek()[0]
            if t == "AND":
                pos += 1
                nodes.append(parse_not())
            elif t in ("TERM", "(", "NOT"):      # AND implicite entre atomes adjacents
                nodes.append(parse_not())
            else:
                break
        nodes = [n for n in nodes if n is not None]
        if not nodes:
            return None
        return nodes[0] if len(nodes) == 1 else ("and", nodes)

    def parse_or():
        nonlocal pos
        nodes = [parse_and()]
        while peek()[0] == "OR":
            pos += 1
            nodes.append(parse_and())
        nodes = [n for n in nodes if n is not None]
        if not nodes:
            return None
        return nodes[0] if len(nodes) == 1 else ("or", nodes)

    return parse_or()


# ── Ce qui a DÉJÀ été essayé pour accélérer ce chemin (ne pas refaire) ───────────
# EXPLAIN ANALYZE en production (346 152 documents, 1 245 182 chunks, 1 terme, LIMIT 1000) :
#
#   Seq Scan on literature_document d                        8 496 ms
#     Filter: (title LIKE … OR abstract LIKE … OR (hashed SubPlan 2))
#     SubPlan 2
#       Bitmap Heap Scan on document_chunk                   4 934 ms   ← 50 % de faux
#         Rows Removed by Index Recheck: 8105                             positifs relus
#         Bitmap Index Scan on ix_docchunk_content_trgm         74 ms   ← l'index MARCHE
#
# Les trois index GIN trigrammes existent et sont VALIDES. La lecture tentante est :
# « l'EXISTS dans le OU empêche le BitmapOr, donc les index sur title/abstract ne
# servent jamais ; compilons l'AST en UNION/INTERSECT/EXCEPT sur document_id ».
#
# Cela a été implémenté et MESURÉ (même résultats, à l'ensemble d'id près) :
#
#   corpus              filtre de ligne (actuel)      ensembliste
#    20 000 docs                  556 ms                 1 876 ms
#    50 000 docs                  657 ms                 5 036 ms
#   120 000 docs                1 010 ms                11 865 ms
#
# La version ensembliste est 3 à 12× PLUS LENTE, et l'écart CROÎT avec le corpus. Neuf
# termes donnent 27 sous-requêtes ; les termes fréquents (incidence, prevalence,
# forecasting…) renvoient chacun des dizaines de milliers d'id, et les UNION/INTERSECT
# successifs sur ces ensembles coûtent bien plus qu'un seul balayage séquentiel qui
# court-circuite dès le premier prédicat vrai.
#
# Surtout : le coût dominant (4,9 s des 8,5 s) est la RELECTURE de tas du bitmap — les
# trigrammes rendent 16 338 chunks candidats dont la moitié sont de faux positifs, et
# vérifier un LIKE oblige à relire le texte complet. TOUTE approche fondée sur LIKE paie
# ce prix, y compris la version ensembliste. Le seul vrai levier serait la recherche
# plein texte (tsvector + GIN) : pas de relecture, index bien plus petit — mais la
# sémantique d'appariement change (racinisation, frontières de mots, plus de
# sous-chaînes), donc c'est une décision produit, pas une optimisation transparente.
def _boolean_ast_to_sql(ast, params: dict, idx: list | None = None) -> str | None:
    """Compile l'AST booléen en fragment SQL : chaque feuille apparie le titre, le
    résumé ET le texte du chunk (équivalent [Title/Abstract]) ; AND/OR/NOT préservés."""
    if idx is None:
        idx = [0]
    if ast is None:
        return None
    typ = ast[0]
    if typ == "term":
        key = f"bq_{idx[0]}"
        idx[0] += 1
        params[key] = f"%{ast[1].rstrip('*')}%"       # troncature : déjà une sous-chaîne
        # Appartenance PAR DOCUMENT (pas par chunk) : un terme correspond si le
        # titre, le résumé OU N'IMPORTE QUEL chunk du document le contient (EXISTS
        # corrélé sur d.id). Indispensable pour NOT : compiler `NOT terme` en
        # `(NOT (… OR c.content LIKE …))` évalué PAR LIGNE de chunk laissait
        # entrer un article exclu dès qu'un AUTRE de ses chunks ne contenait pas le
        # terme (SELECT DISTINCT d.id) — fuite d'articles exclus dans le corpus.
        # Corrige aussi le AND inter-chunks (deux termes dans deux chunks distincts).
        return (f"(LOWER(COALESCE(d.title,'')) LIKE :{key}"
                f" OR LOWER(COALESCE(d.abstract,'')) LIKE :{key}"
                f" OR EXISTS (SELECT 1 FROM document_chunk c2"
                f" WHERE c2.document_id = d.id AND LOWER(COALESCE(c2.content,'')) LIKE :{key}))")
    if typ == "not":
        inner = _boolean_ast_to_sql(ast[1], params, idx)
        return f"(NOT {inner})" if inner else None
    if typ in ("and", "or"):
        parts = [p for p in (_boolean_ast_to_sql(ch, params, idx) for ch in ast[1]) if p]
        if not parts:
            return None
        joiner = " AND " if typ == "and" else " OR "
        return "(" + joiner.join(parts) + ")"
    return None


def _build_boolean_match_sql_from_query(query: str, params: dict) -> str:
    """Requête booléenne → fragment SQL de correspondance (groupement RESPECTÉ).

    Un AST VIDE (requête sans terme : ponctuation seule, opérateurs/tags de champ
    seuls, ou repli dégradé) renvoie 'FALSE' — AUCUNE correspondance — et NON 'TRUE'.
    'TRUE' faisait exploser le corpus à la base ENTIÈRE (appartenance = tous les
    documents) sur une requête accidentellement sans terme."""
    return _boolean_ast_to_sql(_parse_boolean_ast(_tokenize_boolean(query)), params) or "FALSE"


def _boolean_to_arxiv(ast) -> str | None:
    """AST booléen → syntaxe de recherche arXiv : chaque terme devient all:"phrase",
    AND/OR conservés, parenthèses pour le groupement. Renvoie None si l'AST contient un
    NOT (l'opérateur arXiv ANDNOT est BINAIRE, pas unaire → ambigu) → l'appelant retombe
    alors sur `all:<mots-clés>`. PUR/testable."""
    if ast is None:
        return None
    typ = ast[0]
    if typ == "term":
        return f'all:"{ast[1].rstrip("*")}"'
    if typ == "not":
        return None
    if typ in ("and", "or"):
        parts = [_boolean_to_arxiv(c) for c in ast[1]]
        if any(p is None for p in parts):
            return None
        return "(" + (" AND " if typ == "and" else " OR ").join(parts) + ")"
    return None


def _boolean_to_s2(ast) -> str | None:
    """AST booléen → syntaxe de l'endpoint Semantic Scholar /paper/search/bulk :
    espace = AND, ` | ` = OR, guillemets = phrase, parenthèses = groupe. Renvoie None si
    l'AST contient un NOT (le `-` de S2 est un préfixe contextuel, ambigu isolé) → l'appelant
    retombe sur l'endpoint classique + mots-clés. PUR/testable."""
    if ast is None:
        return None
    typ = ast[0]
    if typ == "term":
        _t = ast[1].rstrip("*")
        return f'"{_t}"' if " " in _t else _t
    if typ == "not":
        return None
    if typ in ("and", "or"):
        parts = [_boolean_to_s2(c) for c in ast[1]]
        if any(p is None for p in parts):
            return None
        return "(" + (" " if typ == "and" else " | ").join(parts) + ")"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Search (Hybride & Vectorielle pgvector)
# ─────────────────────────────────────────────────────────────────────────────

def _search_local_doc_ids(
    query: str,
    mode: str,
    filters: dict,
    limit: int = 10_000,
    threshold: float = 0.45,
) -> list[str]:
    """Run the same local-DB search logic as /search and return matching doc IDs.

    Used by the pipeline to link already-ingested docs to a new scenario
    without re-querying external APIs.
    """
    where_sql, where_params = _build_where(filters)

    openai_key = os.getenv("OPENAI_API_KEY")
    use_vector = mode in ("semantic", "hybrid") and bool(openai_key)

    query_embedding = None
    if use_vector:
        try:
            from llm_usage import MeteredOpenAI as OpenAI
            client = OpenAI(api_key=openai_key, timeout=90.0)
            query_embedding = client.embeddings.create(
                input=[query.replace("\n", " ").strip()],
                model="text-embedding-3-small",
            ).data[0].embedding
        except Exception as e:
            logger.error(f"_search_local_doc_ids embedding error: {e}")
            use_vector = False

    params: dict[str, Any] = {**where_params, "limit": limit}

    _fts = False
    _ast = None
    if mode == "boolean":
        _ast = _parse_boolean_ast(_tokenize_boolean(query))
        _fts = _lex.use_fts()
        if _fts:
            # Plein texte : UN tsquery pour toute l'expression, évalué dans l'index GIN
            # de document_search — sémantique PAR DOCUMENT, comme le chemin LIKE
            # (cf. lexical_search.py). Un AST vide → FALSE : aucune correspondance.
            any_match_sql = _lex.match_sql(_ast, params) or "FALSE"
        else:
            any_match_sql = _boolean_ast_to_sql(_ast, params) or "FALSE"
    else:
        raw_terms = [t.strip() for t in re.split(r"\s+", query.lower()) if t.strip()]
        query_terms = [re.sub(r"[^a-zA-Z0-9\-_]", "", t) for t in raw_terms if re.sub(r"[^a-zA-Z0-9\-_]", "", t)]
        like_clauses: list[str] = []
        for i, term in enumerate(query_terms):
            key = f"lsd_term_{i}"
            params[key] = f"%{term}%"
            like_clauses.append(
                f"(LOWER(COALESCE(d.title,'')) LIKE :{key}"
                f" OR LOWER(COALESCE(d.abstract,'')) LIKE :{key}"
                f" OR LOWER(COALESCE(c.content,'')) LIKE :{key})"
            )
        any_match_sql = " OR ".join(like_clauses) if like_clauses else "TRUE"

    if use_vector:
        params["q_emb"] = str(query_embedding)
        params["threshold"] = threshold
        sql = text(f"""
            SELECT DISTINCT d.id
            FROM document_chunk c
            JOIN literature_document d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
              AND (1 - (c.embedding <=> CAST(:q_emb AS vector))) > :threshold
              AND d.abstract IS NOT NULL AND length(TRIM(d.abstract)) >= 30
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              {where_sql}
            LIMIT :limit
        """)
    elif mode == "boolean" and _fts:
        # Plein texte : un seul balayage d'index GIN sur document_search, puis jointure
        # par clé primaire pour les filtres (résumé ≥ 30 caractères, doublons, facettes).
        # Mêmes filtres externes que le chemin LIKE ci-dessous → même définition du
        # corpus, seule la correspondance des termes change (cf. lexical_search.py).
        sql = text(f"""
            SELECT d.id
            FROM document_search s
            JOIN literature_document d ON d.id = s.document_id
            WHERE ({any_match_sql})
              AND d.abstract IS NOT NULL AND length(TRIM(d.abstract)) >= 30
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              {where_sql}
            LIMIT :limit
        """)
    elif mode == "boolean":
        # Chemin LIKE/trigramme : REPLI tant que document_search n'est pas rempli
        # (ou LEXICAL_SEARCH_ENGINE=like). Lent — 55 à 240 s par requête sur le corpus
        # de production — mais correct.
        # PERF : le match booléen est PAR DOCUMENT — `any_match_sql` n'apparie que
        # d.title / d.abstract + un EXISTS corrélé sur document_chunk ; il ne référence
        # PAS le chunk joint `c`. Piloter la requête depuis literature_document (~207k
        # lignes) au lieu de document_chunk (souvent 1M+ lignes, multipliées par doc
        # puis dédupliquées par DISTINCT) renvoie EXACTEMENT le même ensemble d'IDs en
        # balayant 5-10× moins de lignes — cause majeure de la lenteur de la recherche
        # locale et de la reconstruction du corpus (exécutée une fois PAR sous-requête).
        sql = text(f"""
            SELECT d.id
            FROM literature_document d
            WHERE ({any_match_sql})
              AND d.abstract IS NOT NULL AND length(TRIM(d.abstract)) >= 30
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              {where_sql}
            LIMIT :limit
        """)
    else:
        # Mode mots-clés : `any_match_sql` référence c.content → la jointure au chunk
        # est nécessaire (DISTINCT dédoublonne les docs à plusieurs chunks matchés).
        sql = text(f"""
            SELECT DISTINCT d.id
            FROM document_chunk c
            JOIN literature_document d ON d.id = c.document_id
            WHERE ({any_match_sql})
              AND d.abstract IS NOT NULL AND length(TRIM(d.abstract)) >= 30
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              {where_sql}
            LIMIT :limit
        """)

    import time as _time_ls
    _t0 = _time_ls.perf_counter()
    with engine.connect() as conn:
        ids = conn.execute(sql, params).scalars().all()
    if mode == "boolean":
        # Une ligne par recherche, avec le moteur utilisé : c'est ce que l'on cherche
        # dans le journal quand « la recherche locale est lente ».
        logger.info(f"lexical search [{'fts' if _fts else 'like'}] {len(ids)} docs in "
                    f"{(_time_ls.perf_counter() - _t0) * 1000:.0f} ms — {query[:100]!r}")
        if _fts:
            _ignored = _lex.stopword_terms(_lex.ast_terms(_ast))
            if _ignored:
                logger.warning(f"lexical search: terms made only of stop words were "
                               f"ignored by PostgreSQL: {_ignored} — {query[:100]!r}")
    return ids


# Limite de récupération par source live (PubMed, OpenAlex, …). Appliquée à l'identique
# à la recherche ET à la construction du corpus. Réglable via l'env LIVE_MAX_PER_SOURCE :
# la mettre très haut (p. ex. 100000) « retire » le plafond — la vraie borne devient alors
# le budget temps (POPULATE_FEDERATION_BUDGET). ⚠ multiplier ce plafond multiplie les
# appels API (risque de 429 PubMed/S2) ET le coût d'embedding OpenAI de CHAQUE recherche.
try:
    LIVE_MAX_PER_SOURCE = int(os.getenv("LIVE_MAX_PER_SOURCE", "2000"))
except (TypeError, ValueError):
    LIVE_MAX_PER_SOURCE = 2000


def _boolean_corpus_ids(boolean_query: str, filters: dict) -> list:
    """LA source de vérité de l'appartenance au corpus : les documents de la base
    locale qui correspondent à la requête booléenne. Recherche et corpus utilisent
    EXACTEMENT ce helper → le compteur de la recherche == la taille du corpus."""
    return _search_local_doc_ids(boolean_query, "boolean", filters, limit=500_000)


def _looks_boolean(text: str) -> bool:
    """Le texte utilise-t-il une SYNTAXE booléenne (→ utilisé tel quel) plutôt que du
    langage naturel (→ traduit) ? Signaux : opérateurs AND/OR/NOT en MAJUSCULES, tags
    de champ ([dp], [tiab], [mesh]…), guillemets doubles appariés, ou parenthèses.
    Pur/déterministe — le même heuristique est reflété côté client (App.tsx:looksBoolean)."""
    if not text:
        return False
    t = text.strip()
    if re.search(r"\[(dp|tiab|ti|ab|mesh|majr|au|tw|la|pt)\]", t, re.IGNORECASE):
        return True
    if re.search(r"\b(AND|OR|NOT)\b", t):          # opérateurs en MAJUSCULES uniquement
        return True
    if t.count('"') >= 2:                           # au moins une phrase entre guillemets
        return True
    if "(" in t and ")" in t:                       # expression groupée
        return True
    return False


def _normalize_sub_queries(sub_queries: Any) -> list[dict]:
    """Nettoie une liste de sous-requêtes : ne garde que les entrées {kind,text} au
    texte non vide. Le `kind` explicite (boolean|natural) est respecté (override
    utilisateur) ; sinon (auto|absent|invalide) il est DÉTECTÉ par _looks_boolean —
    l'utilisateur n'a donc plus à taguer chaque sous-requête. Renvoie [] si aucune."""
    out: list[dict] = []
    if not isinstance(sub_queries, list):
        return out
    for sq in sub_queries:
        if not isinstance(sq, dict):
            continue
        _text = (sq.get("text") or "").strip()
        if not _text:
            continue
        _raw_kind = sq.get("kind")
        if _raw_kind in ("boolean", "natural"):
            _kind = _raw_kind                                      # override explicite
        else:
            _kind = "boolean" if _looks_boolean(_text) else "natural"   # auto-détection
        _raw_op = sq.get("op")
        _op = _raw_op if _raw_op in ("and", "or") else None        # combinateur PAR facette
        out.append({"kind": _kind, "text": _text, "op": _op})
    return out


def _facet_ops(facets: list[dict], combinator: str) -> list[str]:
    """Opérateur EFFECTIF de chaque facette à partir de la 2e ('and'|'or') : l'op porté
    par la facette, sinon le `combinator` global ('intersection'→'and', sinon 'or').
    Une seule source de vérité pour le fold, la règle d'union des sources natives,
    et l'expression affichée."""
    default_op = "and" if combinator == "intersection" else "or"
    out: list[str] = []
    for f in facets[1:]:
        op = f.get("op") if isinstance(f, dict) else None
        out.append(op if op in ("and", "or") else default_op)
    return out


def _facets_intersect(facets: list[dict], combinator: str) -> bool:
    """True dès qu'UNE facette est INTERSECTÉE (ET) avec le résultat courant. Dans ce
    cas un document qui ne correspond qu'à la requête principale n'appartient PAS
    forcément au corpus — les enregistrements booléens-natifs des sources live (qui
    n'ont vu que la requête principale) ne doivent donc pas être unis d'office."""
    return "and" in _facet_ops(facets, combinator)


def _combined_query_text(query: str | None, sub_queries: Any, combinator: str | None) -> str:
    """Expression lisible de la recherche COMPLÈTE : requête principale + sous-requêtes
    avec leurs opérateurs, parenthésée selon le fold gauche→droite réellement appliqué
    (« (A) AND (B) », « ((A) OR (B)) AND (C) »). Mono-requête → la requête telle quelle.
    C'est ce texte qui doit apparaître partout où la recherche est montrée (nom par
    défaut, carte, en-tête, onglet Stratégie) — la colonne `query` ne porte que la
    facette principale, d'où un « ET » invisible auparavant."""
    clean = _normalize_sub_queries(sub_queries)
    if len(clean) < 2:
        return (query or "").strip()
    expr = clean[0]["text"]
    for facet, op in zip(clean[1:], _facet_ops(clean, combinator or "union")):
        expr = f"({expr}) {op.upper()} ({facet['text']})"
    return expr


def _widen_boolean_for_or_facets(boolean: str, pubmed_q: str, sub_queries: Any, combinator: str | None,
                                 translate=None, max_len: int = 1200, max_pubmed_len: int = 1900) -> tuple[str, str, int]:
    """Élargit les requêtes envoyées aux sources LIVE aux facettes UNIES (OU) d'une
    recherche multi-facettes : « (booléen principal) OR (booléen de la facette) ».

    La fédération n'interrogeait que la requête PRINCIPALE : une facette « OU » ne
    trouvait que ce que la base locale contenait déjà — des articles qui ne
    correspondent qu'à elle n'étaient jamais ramenés de PubMed/Europe PMC/OpenAlex.
    Les facettes intersectées (ET) n'ont pas besoin d'être fédérées : leurs résultats
    sont un sous-ensemble de ceux de la requête principale, re-matché localement.
    Une facette naturelle est traduite via `translate` (par défaut
    _generate_search_strategy) ; sans traduction utilisable, son texte est utilisé
    tel quel. Les longueurs sont bornées (limites d'URL des API) : au-delà, on garde la
    requête principale seule. Renvoie (booléen, requête PubMed, nb de facettes ajoutées)."""
    clean = _normalize_sub_queries(sub_queries)
    if len(clean) < 2:
        return boolean, pubmed_q, 0
    translate = translate or _generate_search_strategy
    added = 0
    for facet, op in zip(clean[1:], _facet_ops(clean, combinator or "union")):
        if op != "or":
            continue
        fb, fp = facet["text"], None
        if facet["kind"] != "boolean":
            try:
                gen = translate(facet["text"])
                if isinstance(gen, dict) and gen.get("general") and not _strategy_is_degraded(gen, facet["text"]):
                    fb, fp = gen["general"], gen.get("pubmed")
            except Exception as _e:                      # noqa: BLE001 — repli texte brut
                logger.warning(f"facette OU « {facet['text'][:60]} » : traduction échouée ({_e}) ; texte brut")
        portable = _strip_field_tags(fb).strip() or fb
        if portable in boolean:
            continue
        new_bool = f"({boolean}) OR ({portable})"
        new_pub = f"({pubmed_q}) OR ({fp or portable})" if pubmed_q else (fp or portable)
        if len(new_bool) > max_len or len(new_pub) > max_pubmed_len:
            logger.warning(f"facette OU « {facet['text'][:60]} » ignorée pour la fédération live : requête trop longue")
            continue
        boolean, pubmed_q, added = new_bool, new_pub, added + 1
    return boolean, pubmed_q, added


def _fold_facet_sets(id_sets: list[set], facets: list[dict], combinator: str) -> set:
    """Combine les ensembles d'IDs des facettes de GAUCHE À DROITE : la facette 0
    (requête principale) est la base ; chaque facette suivante est UNIE (op='or') ou
    INTERSECTÉE (op='and') selon SON propre opérateur. Un op absent retombe sur le
    `combinator` global ('intersection'→'and', sinon 'or'). Quand tous les opérateurs
    sont uniformes, le résultat est IDENTIQUE à l'ancien tout-union / tout-intersection
    (le fold gauche→droite de ∪ ou ∩ = ∪/∩ de tous). Permet « principale OU #1 ET #2 »."""
    if not id_sets:
        return set()
    default_op = "and" if combinator == "intersection" else "or"
    out = set(id_sets[0])
    for i in range(1, len(id_sets)):
        op = (facets[i].get("op") if i < len(facets) else None) or default_op
        if op == "and":
            out &= id_sets[i]
        else:
            out |= id_sets[i]
    return out


def _multi_query_corpus_ids(sub_queries: list[dict], combinator: str, filters: dict) -> list:
    """Appartenance au corpus pour une recherche MULTI-sous-requêtes.

    Chaque sous-requête produit un ENSEMBLE d'IDs de documents de la base locale
    par correspondance LEXICALE (booléenne) — EXACTEMENT comme la recherche
    mono-requête (_boolean_corpus_ids) et comme le documente l'étape de populate :
      - kind="boolean" → la requête est utilisée telle quelle (AND/OR/NOT),
      - kind="natural" → elle est d'abord TRADUITE en booléen (_generate_search_strategy,
        déterministe : seed=42, avec expansion de synonymes), puis matchée en booléen.

    Le seuil sémantique N'INTERVIENT JAMAIS dans l'appartenance au corpus : il ne
    sert qu'EN AVAL (page scénario, _get_above_threshold_articles) à sélectionner le
    sous-ensemble PERTINENT parmi le corpus. Un corpus défini lexicalement reste
    reproductible et auditable (exigence revue systématique) ; le score sémantique
    classe/priorise ensuite, sans jamais retirer d'article du corpus.

    Les ensembles sont combinés par UNION (OU) ou INTERSECTION (ET) puis dédupliqués
    (ce sont des ensembles)."""
    clean = _normalize_sub_queries(sub_queries)
    id_sets: list[set] = []
    for sq in clean:
        if sq["kind"] == "boolean":
            _boolean = sq["text"]
        else:
            # Naturel → booléen, comme une requête naturelle mono-requête. Ainsi
            # l'appartenance reste LEXICALE (pas de seuil sémantique). Repli sur le
            # texte brut si la traduction échoue / clé OpenAI absente (mode dégradé).
            try:
                _gen = _generate_search_strategy(sq["text"])
                _boolean = (_gen.get("general") or sq["text"]) if isinstance(_gen, dict) else sq["text"]
            except Exception as _e:
                logger.warning(f"_multi_query_corpus_ids: traduction naturel→booléen échouée ({_e}) ; repli lexical brut")
                _boolean = sq["text"]
        id_sets.append(set(_search_local_doc_ids(_boolean, "boolean", filters, limit=500_000)))
    if not id_sets:
        return []
    # Fold gauche→droite selon l'op PAR facette (défaut = `combinator` global).
    return list(_fold_facet_sets(id_sets, clean, combinator))


def _set_scenario_corpus(scenario_id: str, ids: list, allow_empty: bool = False) -> int:
    """Fixe le corpus d'un scénario à EXACTEMENT `ids` (appartenance booléenne).
    Supprime les liens qui n'en font plus partie et insère les manquants. Si `ids`
    est vide on ne touche à rien (évite de vider le corpus sur un échec transitoire),
    SAUF si allow_empty=True — cas d'une intersection multi-requêtes légitimement
    vide (deux facettes sans document commun), où le corpus DOIT être vidé."""
    if not ids:
        if allow_empty:
            with engine.begin() as _c:
                _c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :sid"),
                           {"sid": scenario_id})
        return 0
    with engine.begin() as _c:
        _c.execute(
            text("DELETE FROM article_scenarios WHERE scenario_id = :sid "
                 "AND document_id NOT IN :ids").bindparams(bindparam("ids", expanding=True)),
            {"sid": scenario_id, "ids": list(ids)},
        )
        # Insertion en masse (un seul aller-retour) plutôt qu'une requête par
        # document : la (ré)construction du corpus local doit être quasi immédiate.
        _c.execute(text("""
            INSERT INTO article_scenarios (document_id, scenario_id, similarity_score)
            SELECT unnest(CAST(:ids AS bigint[])), :s, NULL
            ON CONFLICT (document_id, scenario_id) DO NOTHING
        """), {"ids": list(ids), "s": scenario_id})
    return len(ids)


def _dedup_scenario_links(scenario_id: str) -> int:
    """Supprime les liens `article_scenarios` DOUBLONS d'un scénario : un seul lien
    par article distinct, pour que le « corpus » affiché soit stable entre écrans et
    dans le temps.

    Les index uniques GLOBAUX couvrent le DOI et le titre normalisé, mais PAS le
    PMID : un même article PubMed SANS DOI ingéré via deux chemins (live =
    external_id « pmid:123 », populate = « 123 ») crée deux lignes qui échappent au
    pré-SELECT (external_id différent) ET à l'index DOI (NULL). D'où un article
    compté deux fois, puis un total qui « fond » plus tard quand la maintenance
    marque enfin `is_duplicate`. On fige le compte MAINTENANT, de façon déterministe
    et bornée au scénario : pour chaque clé (DOI › external_id normalisé sans préfixe
    pmid/pmcid › titre normalisé ≥ 20 › id), on ne garde que le lien de plus petit
    `document_id` (le canonique — MÊME règle que `scripts/_softdedup.py`, donc aucun
    « glissement » quand la dédup globale tournera : elle marquera exactement les id
    supérieurs déjà retirés ici). Aucune fusion abusive : seules des clés EXACTES
    (même DOI, même external_id normalisé, ou même titre long) sont réunies ; un
    external_id vide retombe sur le titre puis sur l'id (jamais fusionné). Renvoie le
    nombre de liens supprimés. Tolérant aux pannes (journalise et renvoie 0)."""
    try:
        with engine.begin() as _c:
            n = _c.execute(text("""
                WITH keyed AS (
                    SELECT a.document_id,
                           COALESCE(
                             NULLIF(lower(btrim(d.doi)), ''),
                             NULLIF('ext:' || lower(btrim(
                                 regexp_replace(d.external_id, '^(pmid|pmcid):', '', 'i')
                             )), 'ext:'),
                             CASE WHEN d.title_norm IS NOT NULL
                                   AND length(d.title_norm) >= 20
                                  THEN 'tn:' || d.title_norm END,
                             'id:' || a.document_id::text
                           ) AS k
                    FROM article_scenarios a
                    JOIN literature_document d ON d.id = a.document_id
                    WHERE a.scenario_id = :sid
                ),
                ranked AS (
                    SELECT document_id,
                           MIN(document_id) OVER (PARTITION BY k) AS keep_id
                    FROM keyed
                )
                DELETE FROM article_scenarios a
                USING ranked r
                WHERE a.scenario_id = :sid
                  AND a.document_id = r.document_id
                  AND r.document_id <> r.keep_id
            """), {"sid": scenario_id}).rowcount
        if n:
            logger.info(f"Dédup intra-scénario {scenario_id}: {n} lien(s) doublon(s) "
                        f"supprimé(s) (même DOI/PMID/titre).")
        return n or 0
    except Exception as _e:
        logger.warning(f"Dédup intra-scénario {scenario_id}: {_e}")
        return 0


def _prisma_identification_figures(records_by_source: dict, unique_records: int,
                                   duplicate_rows_removed: int, corpus_total: int,
                                   method: str = "populate",
                                   federation_incomplete: bool = False,
                                   removed_no_abstract: int = 0,
                                   removed_not_matching: int = 0) -> dict[str, Any]:
    """Chiffres PRISMA 2020 de l'étape « identification », calculés à partir de ce qu'une
    recherche a RÉELLEMENT ramené — et non du corpus déjà dédupliqué.

    Pourquoi : le PRISMA affichait « doublons retirés : 0 » par construction. La
    déduplication a lieu à l'ingestion (index uniques DOI / titre normalisé : un article
    renvoyé par OpenAlex ET PubMed devient UNE ligne, la seconde arrivée est absorbée
    sans trace) puis au liage (_dedup_scenario_links, dont le compte n'était que
    journalisé). Le flag `is_duplicate` que comptait le PRISMA n'est posé par aucun
    runtime. Ici, on compte à la source, une fois par source qui a renvoyé l'article.

    records_by_source      : enregistrements ramenés PAR SOURCE (« db_cache » = base locale)
    unique_records         : documents distincts derrière ces enregistrements
    duplicate_rows_removed : liens retirés par _dedup_scenario_links (même DOI/PMID/titre
                             sous deux lignes distinctes de la base)
    corpus_total           : liens restants après nettoyage (sans résumé, doublons)

    PRISMA 2020 : identifiés → doublons retirés → (retirés pour d'autres raisons) →
    passés au screening. « Autres raisons » ici : pas de résumé, ou enregistrement d'une
    source par mots-clés qui ne correspond pas à la requête booléenne en local."""
    from datetime import datetime as _dt, timezone as _tz
    records = {str(k): int(v) for k, v in (records_by_source or {}).items() if int(v or 0) > 0}
    identified = sum(records.values())
    across = max(0, identified - max(0, int(unique_records or 0)))
    rows = max(0, int(duplicate_rows_removed or 0))
    duplicates = min(identified, across + rows)
    unique_after = identified - duplicates
    corpus = max(0, int(corpus_total or 0))
    # Retraits AVANT screening, ventilés : sans résumé (règle qualité), hors requête
    # (enregistrement d'une source par mots-clés qui ne correspond pas au booléen en
    # local), et le reste (résiduel — p. ex. un document marqué doublon global). Les
    # deux premiers sont bornés à ce qui reste à expliquer, dans cet ordre, pour que
    # identifiés − doublons − retraits = passés au screening tienne toujours.
    to_explain = max(0, unique_after - corpus)
    no_abstract = min(to_explain, max(0, int(removed_no_abstract or 0)))
    not_matching = min(to_explain - no_abstract, max(0, int(removed_not_matching or 0)))
    other = to_explain - no_abstract - not_matching
    return {
        "method": method,
        "computed_at": _dt.now(_tz.utc).isoformat(),
        "federation_incomplete": bool(federation_incomplete),
        "records_by_source": records,
        "records_identified": identified,
        "duplicate_records_across_sources": across,
        "duplicate_rows_in_database": rows,
        "duplicates_removed": duplicates,
        "unique_records": unique_after,
        "removed_no_abstract": no_abstract,
        "removed_not_matching": not_matching,
        "removed_other_reasons": other,
        "removed_before_screening": to_explain,
        "records_screened": corpus,
    }


def _reconcile_prisma_identification(figures: dict, corpus_now: int) -> dict[str, Any]:
    """The identification figures of the last search, brought to the corpus AS IT STANDS.

    The figures are computed when a search closes. The corpus can change afterwards: a
    rebuild from the local database, duplicates marked later by the maintenance and —
    before the corpus was frozen at assembly — pages of a slow source arriving after
    the accounting. The panel used to show the live corpus as "records screened" next
    to duplicates and removals computed for another total, so identified − duplicates
    − removals no longer equalled screened (3,623 − 731 − 401 ≠ 3,602). The difference
    is now a line of its own, in the direction it happened:

      added_after_search   — documents in the corpus that this search did not count
      removed_after_search — documents the search screened that have left the corpus

    and identified − duplicates − removals + added − removed_after = screened holds
    for every scenario, whatever happened since the search. The per-source counts and
    the search-time figures are kept as they were (they describe the search)."""
    identified = int(figures.get("records_identified") or 0)
    duplicates = int(figures.get("duplicates_removed") or 0)
    unique = int(figures.get("unique_records") or 0)
    no_abstract = int(figures.get("removed_no_abstract") or 0)
    not_matching = int(figures.get("removed_not_matching") or 0)
    other = int(figures.get("removed_other_reasons") or 0)
    screened_then = max(0, int(figures.get("records_screened") or 0))
    corpus_now = max(0, int(corpus_now or 0))
    out = dict(figures)
    out.update({
        "records_identified": identified,
        "duplicates_removed": duplicates,
        "unique_records": unique,
        "removed_no_abstract": no_abstract,
        "removed_not_matching": not_matching,
        "removed_other_reasons": other,
        "removed_before_screening": no_abstract + not_matching + other,
        "records_screened_at_search": screened_then,
        "added_after_search": max(0, corpus_now - screened_then),
        "removed_after_search": max(0, screened_then - corpus_now),
        "records_screened": corpus_now,
    })
    return out


def _store_prisma_identification(scenario_id: str, figures: dict) -> None:
    """Persiste les chiffres (colonne JSONB user_scenarios.prisma_identification).
    best-effort : une colonne absente ou une panne ne doit jamais faire échouer un
    populate — le PRISMA retombe alors sur le calcul historique (cf. get_user_scenario_prisma)."""
    try:
        with engine.begin() as _c:
            _c.execute(text("UPDATE user_scenarios SET prisma_identification = CAST(:f AS jsonb) "
                            "WHERE id = :sid"), {"f": json.dumps(figures), "sid": scenario_id})
    except Exception as _e:                              # noqa: BLE001
        logger.warning(f"prisma_identification {scenario_id}: {_e}")


def _load_prisma_identification(scenario_id: str) -> dict | None:
    """Chiffres stockés par le dernier populate / rebuild, ou None (scénario antérieur à
    cette comptabilité, ou colonne absente)."""
    try:
        with engine.connect() as _c:
            _raw = _c.execute(text("SELECT prisma_identification FROM user_scenarios WHERE id = :sid"),
                              {"sid": scenario_id}).scalar()
    except Exception:                                    # noqa: BLE001 - colonne absente
        return None
    if isinstance(_raw, str):
        try:
            _raw = json.loads(_raw)
        except Exception:                                # noqa: BLE001
            return None
    return _raw if isinstance(_raw, dict) and _raw.get("records_by_source") is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Déduplication GLOBALE du corpus : comptage « à la lecture » + marquage on-demand
# ─────────────────────────────────────────────────────────────────────────────
# MÊME clé d'identité d'article que _dedup_scenario_links (scope scénario), mais en
# colonnes NUES (sans alias) pour les requêtes globales : DOI › external_id normalisé
# (préfixe pmid/pmcid retiré, casse ignorée) › titre normalisé ≥ 20 › id. On PARTITIONNE
# toujours par project_context (un « literev » et un « gesica » de même titre ne sont PAS
# des doublons — cf. l'index unique partiel uq_litdoc_title_norm sur (project_context,…)).
_DUP_KEY_SQL = """COALESCE(
    NULLIF(lower(btrim(doi)), ''),
    NULLIF('ext:' || lower(btrim(regexp_replace(external_id, '^(pmid|pmcid):', '', 'i'))), 'ext:'),
    CASE WHEN title_norm IS NOT NULL AND length(title_norm) >= 20 THEN 'tn:' || title_norm END,
    'id:' || id::text
)"""

# ids (+ project_context) des lignes NON canoniques = les doublons réels, calculés À LA
# LECTURE : aucune écriture, aucune dépendance au flag is_duplicate (qu'aucun runtime ne
# pose). Sert au « badge doublons », au statut dédup et à l'aperçu de maintenance → les
# compteurs affichés reflètent la réalité même si le script de dédup n'a jamais tourné.
_DUP_IDS_SQL = f"""
    SELECT id, project_context FROM (
        SELECT id, project_context,
               ROW_NUMBER() OVER (PARTITION BY project_context, {_DUP_KEY_SQL} ORDER BY id) AS rn
        FROM literature_document
    ) _g WHERE rn > 1
"""

# Doublons « à purger » = détectés par la clé de contenu (_DUP_IDS_SQL) UNION ceux DÉJÀ
# marqués is_duplicate (script de dédup manuel / historique). On ne remplace donc jamais
# le contrat existant fondé sur le flag ; on l'ÉLARGIT à la détection par contenu.
_DUP_ANY_IDS_SQL = f"""
    SELECT id, project_context FROM literature_document WHERE is_duplicate IS TRUE
    UNION
    SELECT id, project_context FROM ({_DUP_IDS_SQL}) _c
"""

# Marque is_duplicate=TRUE + canonical_id (plus petit id du groupe) sur les doublons réels.
# Idempotent (garde IS DISTINCT FROM). Utilisé UNIQUEMENT par la maintenance corpus
# on-demand (jamais en tâche de fond silencieuse) : geste explicite, prévisualisé et
# sauvegardé, donc toute correction de compteur qui en découle est transparente.
_DUP_FLAG_UPDATE_SQL = f"""
    WITH ranked AS (
        SELECT id, MIN(id) OVER (PARTITION BY project_context, {_DUP_KEY_SQL}) AS keep_id
        FROM literature_document
    )
    UPDATE literature_document d
    SET is_duplicate = TRUE, canonical_id = r.keep_id
    FROM ranked r
    WHERE d.id = r.id AND r.id <> r.keep_id
      AND (d.is_duplicate IS DISTINCT FROM TRUE OR d.canonical_id IS DISTINCT FROM r.keep_id)
"""


def _count_corpus_duplicates(conn, project_context: str | None = None) -> int:
    """Nombre de documents DOUBLONS calculé à la lecture = détectés par la clé de contenu
    (même identité que la dédup) UNION ceux déjà marqués is_duplicate. Honnête sans exiger
    qu'un script ait posé le flag. `project_context=None` → tous projets confondus."""
    if project_context:
        return int(conn.execute(
            text(f"SELECT COUNT(*) FROM ({_DUP_ANY_IDS_SQL}) x WHERE x.project_context = :ctx"),
            {"ctx": project_context},
        ).scalar() or 0)
    return int(conn.execute(
        text(f"SELECT COUNT(*) FROM ({_DUP_ANY_IDS_SQL}) x")
    ).scalar() or 0)


# ─────────────────────────────────────────────────────────────────────────────
# Live federated search
# ─────────────────────────────────────────────────────────────────────────────

def _plain_keywords(query: str, max_words: int = 8) -> str:
    """Convertit une requête booléenne en mots-clés simples pour les API qui
    n'acceptent PAS la syntaxe booléenne (OpenAlex `search` renvoie 400, les
    serveurs de prépublications n'ont pas de recherche plein-texte). On retire
    les opérateurs AND/OR/NOT, parenthèses, guillemets et jokers, en gardant
    les termes significatifs uniques."""
    import re as _re
    raw = _re.sub(r'["()\[\]*]', " ", query or "")
    words = []
    for w in _re.split(r"\s+", raw):
        wl = w.strip().lower()
        if not wl or wl in ("and", "or", "not"):
            continue
        if wl not in words:
            words.append(wl)
        if len(words) >= max_words:
            break
    return " ".join(words)


@app.get("/user-scenarios/{scenario_id}/search-strategy")
def get_search_strategy(scenario_id: str) -> dict[str, Any]:
    """Returns the stored search_strategy JSON for this scenario.
    If not yet generated, generates it now and stores it."""
    row = _get_user_scenario_or_404(scenario_id)
    query = row["query"]
    strategy = row.get("search_strategy")
    # Régénérer si absent OU si la valeur stockée est un repli dégradé (p. ex.
    # généré pendant une panne de quota OpenAI → requête brute échoée). On ne
    # persiste QUE les stratégies valides, pour ne pas figer un cache empoisonné.
    if _strategy_is_degraded(strategy, query):
        strategy = _generate_search_strategy(query)
        if not _strategy_is_degraded(strategy, query):
            try:
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE user_scenarios SET search_strategy = CAST(:strategy AS jsonb) WHERE id = :id
                    """), {"id": scenario_id, "strategy": json.dumps(strategy)})
            except Exception as _e:
                logger.warning(f"get_search_strategy store error: {_e}")
    return strategy if isinstance(strategy, dict) else {}


# Cache mémoire des traductions naturel→booléen. La fonction est DÉTERMINISTE
# (temperature=0, seed=42) → même requête, même stratégie : on évite de rappeler le
# LLM à chaque frappe débouncée de /search-facets puis à nouveau à la recherche. Seuls
# les résultats VALIDES sont mis en cache (jamais un repli dégradé → réessai quand la
# clé/OpenAI redevient disponible). Borné par un flush simple à la capacité.
_STRATEGY_CACHE: dict[str, dict] = {}
_STRATEGY_CACHE_MAX = 512


def _strategy_key(query: str) -> str:
    """Clé de cache NORMALISÉE : minuscules + espaces compactés + rognés. La même
    phrase en langage naturel (à la casse et aux espaces près) mappe donc toujours
    sur la même stratégie booléenne — condition d'un résultat déterministe."""
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def _generate_search_strategy(query: str) -> dict:
    """
    Uses GPT-4.1-mini to generate a structured boolean search strategy from a natural language query.
    Returns a dict with:
    - general: general boolean query string
    - pubmed: PubMed-specific with MeSH tags
    - explanation: brief explanation of term choices
    - synonyms: list of key synonym groups used
    Résultats déterministes mis en cache mémoire (voir _STRATEGY_CACHE) ; renvoie une COPIE.
    """
    _cache_key = _strategy_key(query)
    _cached = _STRATEGY_CACHE.get(_cache_key)
    if _cached is not None:
        return dict(_cached)
    # Cache PERSISTANT (DB) : même clé normalisée → MÊME stratégie à travers les
    # requêtes, les redémarrages et les workers. C'est ce qui garantit qu'une même
    # phrase produit toujours le même booléen (fin de « Main 57 vs sous-requête 56 »).
    try:
        with engine.connect() as _scc:
            _srow = _scc.execute(text(
                "SELECT strategy FROM search_strategy_cache WHERE query_key = :k"),
                {"k": _cache_key}).mappings().first()
        if _srow and isinstance(_srow["strategy"], dict):
            _STRATEGY_CACHE[_cache_key] = dict(_srow["strategy"])
            return dict(_srow["strategy"])
    except Exception as _ce:
        logger.warning(f"_generate_search_strategy DB cache read: {_ce}")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return {"general": query, "pubmed": query, "explanation": "", "synonyms": [], "degraded": True}
    try:
        from llm_usage import MeteredOpenAI as _OAI_ss
        _client = _OAI_ss(api_key=openai_key)
        response = _client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a systematic review librarian. The user may type EITHER a natural-language "
                    "description OR an already-formed boolean query. First decide which it is:\n"
                    "- If it is ALREADY a boolean query (it uses AND/OR/NOT operators or quoted phrases "
                    "with explicit structure), PRESERVE it as-is in 'general' (only fix obvious syntax), and "
                    "set 'explanation' to note that the query was already boolean and kept unchanged.\n"
                    "- Otherwise, TRANSLATE the natural-language query into a boolean query.\n"
                    "Return ONLY valid JSON with these fields:\n"
                    '{"general": "boolean query using AND/OR/NOT and quotes for phrases",\n'
                    '"pubmed": "PubMed-optimized query with MeSH terms [MeSH Terms] and field tags [Title/Abstract]",\n'
                    '"explanation": "1-2 sentences explaining the term choices and synonyms (or that the input was already boolean)",\n'
                    '"synonyms": [["term1", "synonym1a", "synonym1b"], ["term2", "synonym2a"]]}\n'
                    "Keep queries practical and not overly long. Use 2-4 concept groups max."
                )},
                {"role": "user", "content": f"Research query: {query}"}
            ],
            temperature=0,
            seed=42,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        _result = json.loads(response.choices[0].message.content)
        if isinstance(_result, dict) and not _result.get("degraded"):
            # Persiste ; le PREMIER writer gagne (ON CONFLICT DO NOTHING), puis on RELIT
            # la valeur gagnante → deux traductions LLM concurrentes de la même phrase
            # convergent vers UNE seule stratégie persistée et déterministe.
            try:
                with engine.begin() as _scw:
                    _scw.execute(text(
                        "INSERT INTO search_strategy_cache (query_key, strategy) "
                        "VALUES (:k, CAST(:s AS jsonb)) ON CONFLICT (query_key) DO NOTHING"),
                        {"k": _cache_key, "s": json.dumps(_result)})
                    _wrow = _scw.execute(text(
                        "SELECT strategy FROM search_strategy_cache WHERE query_key = :k"),
                        {"k": _cache_key}).mappings().first()
                if _wrow and isinstance(_wrow["strategy"], dict):
                    _result = _wrow["strategy"]      # la valeur PERSISTÉE fait foi
            except Exception as _we:
                logger.warning(f"_generate_search_strategy DB cache write: {_we}")
            if len(_STRATEGY_CACHE) >= _STRATEGY_CACHE_MAX:
                _STRATEGY_CACHE.clear()          # borne simple : flush à la capacité
            _STRATEGY_CACHE[_cache_key] = dict(_result)
        return _result
    except Exception as _e:
        logger.warning(f"_generate_search_strategy failed: {_e}")
        return {"general": query, "pubmed": query, "explanation": "", "synonyms": [], "degraded": True}


class SearchStrategyIn(BaseModel):
    query: str = Field(..., min_length=1)


@app.post("/search-strategy")
def post_search_strategy(payload: SearchStrategyIn) -> dict[str, Any]:
    """Traduit une requête en langage naturel en stratégie booléenne (LLM).

    Permet à la recherche d'AFFICHER la requête booléenne et de l'utiliser comme
    base du corpus : la même stratégie est ensuite persistée sur le scénario, de
    sorte que le compteur de recherche == la taille du corpus (même requête).
    """
    return _generate_search_strategy(payload.query)


class FacetPreviewIn(BaseModel):
    sub_queries: list[dict[str, Any]] | None = None
    combinator: str = Field(default="union")
    filters: dict[str, Any] = Field(default_factory=dict)


@app.post("/search-facets")
def post_search_facets(payload: FacetPreviewIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Prévisualise l'appartenance au corpus AVANT de lancer la recherche.

    Pour chaque facette (requête principale + sous-requêtes) : détecte le type
    (booléen/naturel), TRADUIT le naturel en booléen (renvoyé pour affichage), puis
    COMPTE les correspondances LEXICALES dans la bibliothèque locale via EXACTEMENT le
    helper qui définit le corpus (_search_local_doc_ids en mode booléen). Renvoie le
    compte par facette + les totaux par UNION (OU) et INTERSECTION (ET).

    Aucun score sémantique/Cohere ici : l'appartenance au corpus est purement lexicale.
    NB : ces comptes portent sur la BIBLIOTHÈQUE INDEXÉE — la recherche en direct
    ajoute ensuite des articles des sources externes avant de refixer le corpus."""
    clean = _normalize_sub_queries(payload.sub_queries)
    filters = payload.filters if isinstance(payload.filters, dict) else {}
    facets: list[dict[str, Any]] = []
    id_sets: list[set] = []
    for sq in clean:
        if sq["kind"] == "boolean":
            boolean = sq["text"]
        else:
            try:
                _gen = _generate_search_strategy(sq["text"])
                boolean = (_gen.get("general") or sq["text"]) if isinstance(_gen, dict) else sq["text"]
            except Exception as _e:
                logger.warning(f"post_search_facets: traduction naturel→booléen échouée ({_e})")
                boolean = sq["text"]
        ids = set(_search_local_doc_ids(boolean, "boolean", filters, limit=500_000))
        id_sets.append(ids)
        facets.append({"kind": sq["kind"], "text": sq["text"],
                       "boolean": boolean, "count": len(ids), "op": sq.get("op")})
    if id_sets:
        union = len(set().union(*id_sets))
        intersection = len(set.intersection(*id_sets))
        # `combined` reflète le fold gauche→droite avec l'op PAR facette (les totaux
        # union/intersection restent affichés comme repères).
        combined = len(_fold_facet_sets(id_sets, clean, payload.combinator))
    else:
        union = intersection = combined = 0
    return {"facets": facets, "union": union, "intersection": intersection,
            "combined": combined, "combinator": payload.combinator}
