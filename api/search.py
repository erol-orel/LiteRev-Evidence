"""Boolean and lexical search, facets, corpus membership, search strategy.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import text, bindparam

import lexical_search as _lex

from .core import _norm_lang, app, engine, logger, require_api_key
from .documents import PREPRINT_DOI_RE, _strategy_is_degraded, same_work_survivors
from .scenario_store import (_get_user_scenario_or_404, added_by_hand_sql, has_abstract_sql,
                             reviewer_decision_sql)
from llm_usage import model_for as _model

# ─────────────────────────────────────────────────────────────────────────────
# Préprints : ce qu'on appelle ainsi, en UN endroit
# ─────────────────────────────────────────────────────────────────────────────
# Un préprint arrive par toutes les bases, pas seulement par les serveurs de préprints :
# OpenAlex, Crossref, Semantic Scholar, CORE, OpenAIRE et même PubMed (le programme
# pilote du NIH y indexe bioRxiv, medRxiv, Research Square) en rendent. Le type que la
# source annonce n'est pas toujours enregistré (une notice déjà en base garde le type de
# sa première ingestion) : on reconnaît donc un préprint à ce qui reste sur la ligne.
#
#   - son DOI, quand il est celui d'un serveur de préprints (bioRxiv et medRxiv sous la
#     forme datée « 10.1101/2020.03.05.20031773 » ou l'ancienne à six chiffres, et non
#     « 10.1101/gr. » ou « 10.1101/cshperspect. », qui sont des revues de Cold Spring
#     Harbor) ;
#   - la source ou le type d'ingestion (« preprint », arXiv), l'URL ou le nom de revue
#     d'un serveur de préprints.
#
# Sauf trace d'une version PUBLIÉE sur la même ligne : un DOI de revue (arXiv donne celui
# de la version publiée quand elle existe), ou un nom de revue qui n'est pas celui d'un
# serveur de préprints. La déduplication par titre réunit souvent le préprint et l'article
# publié sous une seule ligne, qui garde le DOI de la première arrivée et reçoit la revue
# de la seconde : cette ligne est un article publié, et l'exclusion ne doit pas l'écarter.
# PREPRINT_DOI_RE : défini dans `documents`, où la dédup par travail le lit aussi.
PREPRINT_URL_RE = (
    r"(biorxiv\.org|medrxiv\.org|arxiv\.org/|researchsquare\.com|ssrn\.com|preprints\.org"
    r"|chemrxiv\.org|psyarxiv\.com|osf\.io/preprints|authorea\.com"
    r"|europepmc\.org/article/ppr/)"
)
#: Nom de revue d'un serveur de préprints, en tête du champ (PubMed écrit « bioRxiv : the
#: preprint server for biology », Crossref « SSRN Electronic Journal »).
PREPRINT_JOURNAL_RE = (
    r"^(biorxiv|medrxiv|arxiv|research ?square|ssrn|preprints?(\.org)?|chemrxiv|techrxiv"
    r"|psyarxiv|socarxiv|osf preprints|authorea|scielo preprints|jmir preprints)\y"
)
PREPRINT_SOURCES = ("preprint", "biorxiv", "medrxiv", "arxiv")


def preprint_sql(doc: str = "d") -> str:
    """Le prédicat SQL « ce document est un préprint » sur l'alias `doc` de
    literature_document. Un seul endroit : le filtre de recherche, la règle de corpus, le
    lien incrémental et les chiffres PRISMA l'emploient tous."""
    _doi = f"lower(COALESCE({doc}.doi, ''))"
    _journal = f"TRIM(COALESCE({doc}.journal, ''))"
    _sources = ", ".join(f"'{s}'" for s in PREPRINT_SOURCES)
    _says_preprint = (
        f"({_doi} ~ '{PREPRINT_DOI_RE}'"
        f" OR {doc}.source_type = 'preprint'"
        f" OR lower(COALESCE({doc}.source, '')) IN ({_sources})"
        f" OR lower(COALESCE({doc}.url, '')) ~ '{PREPRINT_URL_RE}'"
        f" OR {_journal} ~* '{PREPRINT_JOURNAL_RE}')"
    )
    # Le DOI d'une ligne venue d'un serveur ou d'une facette de préprints est celui du
    # préprint, même quand le serveur n'est pas dans la liste (Qeios, ScienceOpen) : il ne
    # vaut pas version publiée. Celui d'arXiv, si.
    _from_server = f"lower(COALESCE({doc}.source, '')) IN ('preprint', 'biorxiv', 'medrxiv')"
    _published = (
        f"(({_doi} ~ '^10\\.' AND NOT {_doi} ~ '{PREPRINT_DOI_RE}' AND NOT {_from_server})"
        f" OR ({_journal} <> '' AND NOT {_journal} ~* '{PREPRINT_JOURNAL_RE}'))"
    )
    return f"({_says_preprint} AND NOT {_published})"


def excludes_preprints(filters: dict[str, Any] | None) -> bool:
    """Les filtres d'une recherche excluent-ils les préprints ?"""
    _v = (filters or {}).get("exclude_preprints")
    return _v is True or str(_v).strip().lower() in ("1", "true", "yes", "on")


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

    # Une année INCONNUE n'est pas hors de la période : « d.year >= :year_min » est NULL
    # pour elle, et chaque recherche bornée en années perdait les notices dont la date
    # n'avait pas été lue (le MedlineDate de PubMed, « 1998 Spring »). Elles restent, et
    # le corpus les compte à part (scenario_counts : `year_unknown`).
    year_min = filters.get("year_min")
    year_max = filters.get("year_max")
    if year_min not in (None, ""):
        clauses.append("(d.year IS NULL OR d.year >= :year_min)")
        params["year_min"] = int(year_min)
    if year_max not in (None, ""):
        clauses.append("(d.year IS NULL OR d.year <= :year_max)")
        params["year_max"] = int(year_max)

    # Recherche « sans préprints » : la bibliothèque locale n'en apparie plus aucun.
    if excludes_preprints(filters):
        clauses.append(f"NOT {preprint_sql('d')}")

    if not clauses:
        return "", {}

    return " AND " + " AND ".join(clauses), params

#: Les tags de champ dont le terme N'EST PAS un mot du texte : type de publication, date,
#: langue, sous-ensemble, filtre, revue. Hors de PubMed, personne ne sait les évaluer, et
#: garder le terme comme mot faisait pire que l'ignorer. « NOT ("news"[Publication Type]
#: OR "letter"[Publication Type] OR "comment"[Publication Type] OR "editorial"[Publication
#: Type]) », la clause finale de la requête HPAI de production, devenait en local « aucun
#: article dont le titre, le résumé ou le texte intégral contient news, letter, comment ou
#: editorial » (« newsletter », « comments », « letters to »...), et à Europe PMC la même
#: exclusion en texte libre : un article sur la couverture médiatique du H5N1 sortait du
#: corpus pour le mot « news ». « "2021"[dp] : "3000"[dp] » laissait 2021 et 3000 comme
#: mots requis. Le terme part avec son tag ; PubMed, qui reçoit la requête taguée, reste
#: seul à les appliquer.
_NON_CONTENT_TAGS = (
    "pt", "ptyp", "publication type", "dp", "pdat", "edat", "mhda", "crdt",
    "date - publication", "date - entry", "date - create", "la", "lang", "language",
    "sb", "subset", "filter", "filt", "ta", "journal",
)
_NON_CONTENT_UNIT_RE = re.compile(
    r'(?:"[^"]*"|[^\s()\[\]"]+)\s*\[\s*(?:'
    + "|".join(re.escape(t) for t in _NON_CONTENT_TAGS) + r')\s*\]',
    re.IGNORECASE,
)
_EMPTY_GROUP_RE = re.compile(r"\(\s*(?:(?:AND|OR|NOT)\s*)*\)", re.IGNORECASE)
_DANGLING_OPERATOR_RES = (
    (re.compile(r"\(\s*(?:AND|OR)\b", re.IGNORECASE), "("),                 # « ( OR x »
    (re.compile(r"\b(?:AND|OR|NOT)\s*\)", re.IGNORECASE), ")"),             # « x AND ) »
    (re.compile(r"\b(?:AND|OR)\s+(?=(?:AND|OR)\b)", re.IGNORECASE), ""),    # « x AND OR y »
    (re.compile(r"\bNOT\s+(?=NOT\b)", re.IGNORECASE), ""),                  # « x NOT NOT (y) »
    (re.compile(r"\bNOT\s*(?=(?:AND|OR)\b|$)", re.IGNORECASE), ""),         # « NOT AND y », « NOT » final
    (re.compile(r"^\s*(?:AND|OR)\b", re.IGNORECASE), ""),                   # « OR x » en tête
    (re.compile(r"\b(?:AND|OR)\s*$", re.IGNORECASE), ""),                   # « x AND » en queue
)


def _tidy_boolean_operators(s: str) -> str:
    """Après le retrait d'unités entières : plus de groupe vide ni d'opérateur orphelin.

    « a NOT ( ) », « ( OR b ) », « a AND AND b », « NOT » en queue : chacun est soit une
    erreur de syntaxe chez une API qui reçoit la chaîne telle quelle (OpenAlex, DOAJ,
    CORE, ClinicalTrials.gov, OpenAIRE), soit, pour le parseur local, un jeton parasite
    dont la tolérance peut retourner le sens (« a NOT NOT (b) » lisait b comme requis).
    Les phrases entre guillemets sont masquées pendant le nettoyage : un « and or » dans
    une phrase n'est pas un opérateur."""
    phrases: list[str] = []

    def _mask(m):
        phrases.append(m.group(0))
        return f'"\x00{len(phrases) - 1}\x00"'

    s = re.sub(r'"[^"]*"', _mask, s)
    prev = None
    while prev != s:
        prev = s
        s = _EMPTY_GROUP_RE.sub(" ", s)
        for rx, rep in _DANGLING_OPERATOR_RES:
            s = rx.sub(rep, s)
    return re.sub(r'"\x00(\d+)\x00"', lambda m: phrases[int(m.group(1))], s)


def _strip_field_tags(query: str) -> str:
    """Enlève les tags de champ PubMed ([MeSH Terms], [Title/Abstract], [tiab]…).
    Ils n'ont pas d'équivalent dans la base locale (on apparie titre+résumé+texte, soit
    l'équivalent de [Title/Abstract]) et, laissés en place, polluaient le parsing en
    devenant des termes REQUIS parasites (« titleabstract », « meshterms »).

    Un terme dont le tag n'est pas un champ de texte (`_NON_CONTENT_TAGS` : type de
    publication, date, langue...) part AVEC son tag, et la requête est renettoyée pour
    rester bien formée. Sans unité de ce genre, la sortie est strictement celle d'avant."""
    q = query or ""
    if not _NON_CONTENT_UNIT_RE.search(q):
        return re.sub(r"\[[^\]]*\]", " ", q)
    q = _NON_CONTENT_UNIT_RE.sub(" ", q)
    q = re.sub(r"(?<!\S):(?!\S)", " ", q)                 # la borne « : » d'une plage de dates
    return _tidy_boolean_operators(re.sub(r"\[[^\]]*\]", " ", q))


class _BooleanTerm(str):
    """Un terme normalisé, ce que lisent le parseur et les compilateurs locaux, qui garde
    sa graphie d'origine (`raw`, en minuscules) pour les API qui l'apparient mieux telle
    quelle : « 2.3.4.4b » est un seul mot pour OpenAlex, quatre une fois découpé. Une
    sous-classe de `str`, pour que l'arbre garde sa forme et ses comparaisons."""
    raw: str = ""


#: Une ponctuation à ses deux bouts, retirée de la graphie d'origine.
_EDGE_SEPARATORS_RE = re.compile(rf"^(?:{_lex.SEPARATOR_PATTERN})|(?:{_lex.SEPARATOR_PATTERN})$")


def _unicode_punctuation_to_space(s: str) -> str:
    """Une ponctuation hors ASCII (guillemets courbes ou français, tirets moyens et longs)
    sépare deux mots, comme la ponctuation ASCII. Une requête collée depuis un traitement
    de texte arrive avec des guillemets courbes, que le tokeniseur ne prend pas pour des
    guillemets : ils restaient collés au mot (« “avian »), qui ne correspondait plus à
    rien. Les lettres accentuées et les chiffres de toute écriture restent."""
    return "".join(c if (c.isascii() or c.isalnum() or c.isspace()) else " " for c in s)


def _clean_boolean_term(raw: str) -> str:
    """Normalise un terme ou une phrase : minuscules, ponctuation changée en espace.

    La ponctuation à l'intérieur d'un terme SÉPARE deux mots, du côté du terme comme du
    côté du texte (lexical_search.SEPARATOR_PATTERN, la même règle des deux côtés et
    pour les deux moteurs). Elle était supprimée du terme seul : « 2.3.4.4b » devenait
    « 2344b », « influenza A(H5N1) » « influenza ah5n1 », « A/H5N1 » « ah5n1 », « PM2.5 »
    « pm25 », des formes qu'aucun document ne contient, envoyées telles quelles à
    OpenAlex, DOAJ, CORE et ClinicalTrials.gov. Les lettres accentuées, les chiffres,
    '_' et '-' restent (« cathéter » reste « cathéter »).

    Une '*' FINALE est la troncature PubMed (forecast* = forecast, forecasting,
    forecasts…) : elle est GARDÉE en fin de terme, et chaque compilateur en fait ce
    qu'il sait faire (préfixe en plein texte, sous-chaîne en LIKE, retirée pour arXiv
    et S2). Toute autre '*' est du bruit. La graphie d'origine voyage avec le terme
    (`_BooleanTerm.raw`), pour les API (cf. `_api_phrase`)."""
    low = _unicode_punctuation_to_space((raw or "").lower())
    t = _lex.separate(low)
    trunc = t.endswith("*")
    t = " ".join(t.replace("*", "").split())
    out = _BooleanTerm(f"{t}*" if (t and trunc) else t)
    orig = " ".join(_EDGE_SEPARATORS_RE.sub("", " ".join(low.replace("*", "").split())).split())
    out.raw = (f"{orig}*" if (orig and trunc) else orig) if t else ""
    return out


def _api_phrase(term, keep: str = ".") -> str:
    """La forme d'un terme pour une API qui reçoit le booléen dans une URL : la graphie
    d'origine où la ponctuation de `keep` reste quand elle est DANS un mot (le point de
    « 2.3.4.4b » ou de « pm2.5 », que les moteurs d'OpenAlex, DOAJ, CORE ou arXiv lisent
    comme un seul mot), et où toute autre ponctuation devient une espace (une parenthèse
    ou une barre oblique y sépare déjà deux mots, et dans leur syntaxe elles sont des
    opérateurs). Sans étoile. PUR/testable."""
    raw = getattr(term, "raw", "") or str(term or "")
    raw = raw.replace("*", "")
    kept = set(keep)
    out = re.sub(_lex.SEPARATOR_PATTERN,
                 lambda m: m.group(0) if (kept and set(m.group(0)) <= kept) else " ", raw)
    return " ".join(out.split())


def _api_needs_quotes(phrase: str, keep: str = ".") -> bool:
    """Une phrase, ou un mot qui garde une ponctuation, part entre guillemets."""
    return " " in phrase or any(c in phrase for c in keep)


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
    parenthèses déséquilibrées (arrêt propre) - corrige l'ancien parseur plat qui
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
# Surtout : le coût dominant (4,9 s des 8,5 s) est la RELECTURE de tas du bitmap - les
# trigrammes rendent 16 338 chunks candidats dont la moitié sont de faux positifs, et
# vérifier un LIKE oblige à relire le texte complet. TOUTE approche fondée sur LIKE paie
# ce prix, y compris la version ensembliste. Le seul vrai levier serait la recherche
# plein texte (tsvector + GIN) : pas de relecture, index bien plus petit - mais la
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
        needle = ast[1].rstrip('*')
        params[key] = f"%{needle}%"                   # troncature : déjà une sous-chaîne
        # Appartenance PAR DOCUMENT (pas par chunk) : un terme correspond si le
        # titre, le résumé OU N'IMPORTE QUEL chunk du document le contient (EXISTS
        # corrélé sur d.id). Indispensable pour NOT : compiler `NOT terme` en
        # `(NOT (… OR c.content LIKE …))` évalué PAR LIGNE de chunk laissait
        # entrer un article exclu dès qu'un AUTRE de ses chunks ne contenait pas le
        # terme (SELECT DISTINCT d.id) - fuite d'articles exclus dans le corpus.
        # Corrige aussi le AND inter-chunks (deux termes dans deux chunks distincts).
        if " " not in needle:
            return (f"(LOWER(COALESCE(d.title,'')) LIKE :{key}"
                    f" OR LOWER(COALESCE(d.abstract,'')) LIKE :{key}"
                    f" OR EXISTS (SELECT 1 FROM document_chunk c2"
                    f" WHERE c2.document_id = d.id AND LOWER(COALESCE(c2.content,'')) LIKE :{key}))")
        # Plusieurs mots (une phrase, ou un terme que sa ponctuation sépare : « a h5n1 »
        # pour A(H5N1) comme pour A/H5N1) : le texte passe par la MÊME règle des
        # séparateurs que le terme, sinon « a h5n1 » ne trouve pas « A(H5N1) ». Les mots
        # dans l'ordre sur le texte brut sont une condition nécessaire, que l'index
        # trigramme sait servir ; la règle des séparateurs ne s'applique qu'à ce qui la passe.
        pre = f"{key}_w"
        params[pre] = "%" + "%".join(needle.split()) + "%"

        def _match(expr: str) -> str:
            return (f"({expr} LIKE :{pre}"
                    f" AND regexp_replace({expr}, {_lex.SEPARATOR_SQL}, ' ', 'g') LIKE :{key})")
        in_title = _match("LOWER(COALESCE(d.title,''))")
        in_abstract = _match("LOWER(COALESCE(d.abstract,''))")
        in_chunk = _match("LOWER(COALESCE(c2.content,''))")
        return (f"({in_title} OR {in_abstract}"
                f" OR EXISTS (SELECT 1 FROM document_chunk c2"
                f" WHERE c2.document_id = d.id AND {in_chunk}))")
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
    seuls, ou repli dégradé) renvoie 'FALSE' - AUCUNE correspondance - et NON 'TRUE'.
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
        return f'all:"{_api_phrase(ast[1])}"'
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
        _t = _api_phrase(ast[1])
        return f'"{_t}"' if _api_needs_quotes(_t) else _t
    if typ == "not":
        return None
    if typ in ("and", "or"):
        parts = [_boolean_to_s2(c) for c in ast[1]]
        if any(p is None for p in parts):
            return None
        return "(" + (" " if typ == "and" else " | ").join(parts) + ")"
    return None


def _positive_boolean(ast, keep_not: bool = False):
    """L'arbre aplati, chaque bloc sans doublon, et sans ses exclusions sauf `keep_not`.

    Base de la requête RÉDUITE : un `NOT` n'a pas d'équivalent sûr sur arXiv ni sur
    Semantic Scholar, et le ré-appariement local applique de toute façon la requête
    entière, exclusions comprises. Les OU imbriqués (« ((A OU B) OU C) OU D ») sont fondus
    en un seul bloc pour que la troncature compte des termes, pas des niveaux ; et
    « Environmental Exposure » suivi de « Environmental Exposure* » ne fait qu'un terme
    une fois l'étoile retirée. Avec `keep_not`, les exclusions restent (la forme PROPRE du
    booléen entier, pour les API qui acceptent NOT). PUR/testable."""
    if not isinstance(ast, tuple):
        return None
    typ = ast[0]
    if typ == "term":
        return ast if str(ast[1]).rstrip("*").strip() else None
    if typ == "not":
        if not keep_not:
            return None
        inner = _positive_boolean(ast[1], keep_not=True)
        return ("not", inner) if inner is not None else None
    if typ in ("and", "or"):
        children: list = []
        seen: set = set()
        for child in ast[1]:
            sub = _positive_boolean(child, keep_not=keep_not)
            if sub is None:
                continue
            parts = sub[1] if (isinstance(sub, tuple) and sub[0] == typ) else [sub]
            for p in parts:
                key = (" ".join(str(p[1]).rstrip("*").lower().split()) if p[0] == "term"
                       else repr(p))
                if key in seen:
                    continue
                seen.add(key)
                children.append(p)
        if typ == "or":
            # Les mots seuls d'abord, puis les phrases courtes, et à nombre de mots égal
            # l'ordre de l'auteur, qui met d'ordinaire le concept principal en tête. Sous
            # une troncature serrée (OpenAIRE : quatre opérateurs), « occupational » et
            # « h5n1 » valent mieux que « influenza a virus h5n1 subtype », une vedette
            # MeSH que personne n'écrit ; et trier les mots seuls par longueur gardait
            # « cullers » avant « occupational ».
            children.sort(key=lambda p: ((0, len(str(p[1]).split())) if p[0] == "term" else (1, 0)))
        if not children:
            return None
        return children[0] if len(children) == 1 else (typ, children)
    return None


def _or_width(ast) -> int:
    """Le plus large bloc OU de l'arbre (1 s'il n'y en a pas)."""
    if not isinstance(ast, tuple) or ast[0] == "term":
        return 1
    if ast[0] == "not":
        return _or_width(ast[1])
    width = len(ast[1]) if ast[0] == "or" else 1
    return max([width] + [_or_width(c) for c in ast[1]])


def _truncate_or_groups(ast, keep: int):
    """Chaque bloc OU réduit à ses `keep` premiers termes ; les ET gardent tous les leurs.

    C'est la structure de la requête qui est conservée (le ET entre les concepts, qui dit
    de quoi parle la revue), et c'est la largeur des synonymes qui est sacrifiée."""
    if not isinstance(ast, tuple) or ast[0] == "term":
        return ast
    if ast[0] == "not":
        inner = _truncate_or_groups(ast[1], keep)
        return ("not", inner) if inner is not None else None
    children = [c for c in (_truncate_or_groups(c, keep) for c in ast[1]) if c is not None]
    if ast[0] == "or":
        children = children[:max(1, keep)]
    if not children:
        return None
    return children[0] if len(children) == 1 else (ast[0], children)


def _boolean_to_generic(ast, top: bool = True) -> str | None:
    """AST booléen → la syntaxe commune d'OpenAlex, DOAJ, CORE, ClinicalTrials.gov et
    OpenAIRE : AND/OR en toutes lettres, guillemets pour les phrases, parenthèses pour les
    groupes, pas d'étoile. Un terme garde le point qui est DANS un mot (« 2.3.4.4b »,
    « pm2.5 », entre guillemets), et sa ponctuation devient sinon une espace (cf.
    `_api_phrase`). None sur un NOT (la requête réduite n'en porte pas). PUR/testable."""
    if ast is None:
        return None
    typ = ast[0]
    if typ == "term":
        _t = _api_phrase(ast[1])
        return f'"{_t}"' if _api_needs_quotes(_t) else _t
    if typ == "not":
        inner = _boolean_to_generic(ast[1], top=False)
        return f"NOT {inner}" if inner else None
    if typ in ("and", "or"):
        parts = [_boolean_to_generic(c, top=False) for c in ast[1]]
        if any(p is None for p in parts):
            return None
        body = (" AND " if typ == "and" else " OR ").join(parts)
        return body if top else "(" + body + ")"
    return None


def _clean_boolean(portable: str) -> str:
    """Le booléen ENTIER, exclusions comprises, dans la syntaxe propre que les API
    acceptent dans une URL : AND/OR/NOT en capitales, phrases entre guillemets, un seul
    espace, pas d'étoile. "" si rien n'en sort.

    Mesuré sur le scénario HPAI_last de production : OpenAlex et DOAJ ont répondu 400 au
    booléen portable brut (« ( "Occupational Exposure" OR "occupational*" OR "farm
    worker*" ... »), pendant que CORE et ClinicalTrials.gov l'acceptaient ; la requête de
    contrôle, sans troncature, passait chez les quatre. L'étoile de troncature n'a pas de
    sens pour ces moteurs, qui racinisent ; elle part, et le reste est rendu depuis
    l'arbre, pas recopié. PUR/testable."""
    try:
        ast = _positive_boolean(_parse_boolean_ast(_tokenize_boolean(portable or "")), keep_not=True)
    except Exception:                                    # noqa: BLE001
        return ""
    return (_boolean_to_generic(ast) or "") if ast is not None else ""


#: OpenAIRE (Graph API v2) refuse toute recherche de plus de QUATRE opérateurs logiques :
#: « Too many logical operators found. Max allowed is 4 », mesuré en production avec la
#: sonde /sources/health : 48 opérateurs sur la requête recommandée, 52 sur la requête
#: HPAI réduite, 95 sur la requête HPAI entière, 400 à chaque fois. Depuis la migration
#: vers cette API, OpenAIRE ne répondait donc qu'aux requêtes sans booléen.
OPENAIRE_MAX_OPERATORS = 4

#: DOAJ sert la première page d'un booléen long et répond 502 à toutes les suivantes,
#: quelle que soit la taille de page. Mesuré depuis le serveur de production sur la
#: requête du scénario HPAI_last : 894 caractères propres, page 2 en 502 (en 100 comme en
#: 10 par page) ; la même requête réduite à 796 caractères pagine normalement, comme la
#: requête de contrôle de 372 caractères et toute requête courte. La limite est prise
#: sous le plus long qui passe, avec de la marge.
DOAJ_MAX_QUERY_CHARS = 700


def _count_operators(rendered: str) -> int:
    """Les opérateurs d'une requête rendue. Les termes sont en minuscules (tokenisation),
    donc un « AND », un « OR » ou un « NOT » en capitales ne vient que des compilateurs.
    Le NOT compte aussi : la requête réduite n'en porte jamais, mais un morceau du booléen
    découpé (`_split_boolean`) garde ses exclusions, et OpenAIRE les compte."""
    return rendered.count(" AND ") + rendered.count(" OR ") + len(re.findall(r"\bNOT ", rendered))


#: La limite d'URL des API qui reçoivent le booléen dans un GET (OpenAlex,
#: ClinicalTrials.gov, arXiv, OpenAIRE), mesurée sur la requête ENCODÉE et non en
#: caractères : les guillemets, les parenthèses et les deux-points y prennent trois
#: octets chacun, et c'est l'URL encodée que les serveurs et les proxys refusent. La
#: même marge que le passage d'eutils en POST (sources._NCBI_POST_ABOVE) : sous les
#: 2 000 caractères d'URL où les requêtes cessent d'être sûres, une fois l'adresse et
#: les autres paramètres comptés.
URL_QUERY_MAX_ENCODED = 1800
#: Semantic Scholar's bulk query rides in a URL exactly like OpenAlex's, and nothing
#: measured it: every other source weighed its query against a limit and shortened or split
#: it, this one rendered the whole tree and sent it. The longest T2.4 string, COVID exposure
#: at 4,519 characters, came back 400 Bad Request, and the source contributed nothing to a
#: 27,405 article corpus. Measured on the encoded form, as the others are.
S2_MAX_QUERY_ENCODED = 1800


def _url_encoded_len(query: str) -> int:
    """La longueur d'une requête une fois encodée dans une URL (comme `requests` le fait)."""
    import urllib.parse as _up
    return len(_up.quote_plus(query or ""))


def _shorten_boolean(portable: str, limit: int = 1200, render=None,
                     max_operators: int | None = None, measure=None) -> str:
    """La requête RÉDUITE : le même booléen, sans exclusions, chaque bloc OU tronqué à ses
    premiers termes jusqu'à tenir sous `limit` caractères (et, si `max_operators` est
    donné, sous ce nombre d'opérateurs : OpenAIRE n'en accepte que quatre). "" si rien
    ne tient.

    Au-delà de la limite d'URL (1 200 caractères portables), cinq sources recevaient huit
    mots-clés sans structure, et chaque moteur en faisait autre chose. Mesuré sur le
    premier run de production après #326 : arXiv et CORE lisaient le sac de mots en OU
    (2 000 notices chacun, sur « exposure » ou « virus », toutes hors requête) ; OpenAlex,
    DOAJ et ClinicalTrials.gov le lisaient en ET de huit mots dont des variantes qui
    s'excluent (« h5n1 » ET « h7n9 »). Le ET entre les concepts est ce qui dit de quoi
    parle la revue : il est gardé ; c'est la largeur des synonymes qui est sacrifiée.
    `render` est le compilateur de la cible (générique par défaut, arXiv sinon), parce que
    la longueur se mesure dans la syntaxe envoyée ; `measure` la mesure (en caractères par
    défaut, `_url_encoded_len` pour une limite d'URL). PUR/testable."""
    render = render or _boolean_to_generic
    measure = measure or len
    try:
        ast = _positive_boolean(_parse_boolean_ast(_tokenize_boolean(portable or "")))
    except Exception:                                    # noqa: BLE001
        return ""
    if ast is None:
        return ""
    for keep in range(_or_width(ast), 0, -1):
        out = render(_truncate_or_groups(ast, keep))
        if not out or measure(out) > limit:
            continue
        if max_operators is not None and _count_operators(out) > max_operators:
            continue
        return out
    return ""


#: Au-delà, une stratégie découpée serait une rafale de requêtes pour une seule source :
#: on garde alors sa requête réduite, et la carte le dit comme avant.
SPLIT_MAX_PARTS = 16

#: OpenAlex refuse (400) une requête de trop d'opérateurs, même courte : sur la stratégie
#: exposition/transmission du scénario T2.4 HPAI, découpée en quatre morceaux à la limite
#: d'URL, celui de 60 opérateurs (1 548 caractères) a été refusé et les trois autres (23 à
#: 47 opérateurs) sont passés. La borne est prise sous le plus grand qui passe, avec de la
#: marge ; un morceau refusé quand même est coupé en deux et relancé (`_bisect_boolean`).
OPENALEX_MAX_OPERATORS = 40
#: Les morceaux d'OpenAlex, bornés en opérateurs, sont plus nombreux que ceux d'une simple
#: limite d'URL.
OPENALEX_SPLIT_MAX_PARTS = 24
#: Combien de fois un morceau refusé est recoupé en deux : 60 opérateurs en 8 morceaux d'environ
#: 8 au bout de trois coupes, et un refus qui tient à autre chose (un filtre) n'insiste pas.
OPENALEX_BISECT_DEPTH = 3


def _halves(children: list) -> list:
    """Deux moitiés d'une liste de termes OU, chacune un nœud (un terme seul reste lui-même)."""
    mid = len(children) // 2
    return [h[0] if len(h) == 1 else ("or", h) for h in (children[:mid], children[mid:])]


def _split_node_once(node, render) -> list | None:
    """Le nœud en DEUX morceaux dont l'union est le nœud, par la distributivité : un OU coupé
    en deux, ou sous un ET le plus large de ses blocs OU coupé en deux. None si rien ne se
    coupe (un terme seul, ou un ET sans bloc OU de plus d'un terme)."""
    if not isinstance(node, tuple) or node[0] not in ("and", "or"):
        return None
    kids = list(node[1])
    if node[0] == "or":
        return _halves(kids) if len(kids) >= 2 else None
    ors = [i for i, k in enumerate(kids) if isinstance(k, tuple) and k[0] == "or" and len(k[1]) > 1]
    if not ors:
        return None
    widest = max(ors, key=lambda i: len(render(kids[i]) or ""))
    return [("and", kids[:widest] + [half] + kids[widest + 1:]) for half in _halves(list(kids[widest][1]))]


def _bisect_boolean(query: str, render=None) -> list[str]:
    """Une requête booléenne en DEUX dont l'union est la requête, exclusions gardées sur les
    deux : ce qu'on relance quand une source refuse un morceau. [] si rien ne se coupe.
    PUR/testable."""
    render = render or _boolean_to_generic
    try:
        ast = _positive_boolean(_parse_boolean_ast(_tokenize_boolean(query or "")), keep_not=True)
    except Exception:                                    # noqa: BLE001
        return []
    pieces = _split_node_once(ast, render) if ast is not None else None
    if not pieces:
        return []
    out = [render(p) for p in pieces]
    return out if all(out) else []


def _split_boolean(portable: str, limit: int, render=None,
                   max_parts: int = SPLIT_MAX_PARTS, measure=None,
                   max_operators: int | None = None) -> list[str]:
    """Le booléen ENTIER en plusieurs requêtes courtes dont l'UNION est le booléen.

    La requête réduite (`_shorten_boolean`) tient sous la limite en retirant des
    synonymes : DOAJ recevait ainsi 428 notices là où le booléen entier en annonce 514,
    parce qu'elle ne pagine pas une requête de plus de DOAJ_MAX_QUERY_CHARS caractères.
    Pour une recherche exhaustive, on garde tous les termes et on découpe à la place.

    La règle est la distributivité : (A1 OU A2) ET B = (A1 ET B) OU (A2 ET B). Un bloc OU
    trop long est coupé en deux, sous un ET le plus large de ses blocs OU est coupé en
    deux, et ainsi de suite jusqu'à ce que chaque morceau tienne. Les exclusions restent
    attachées à chaque morceau, donc l'union est exactement le booléen. [] si un terme
    seul ne tient pas, ou s'il faudrait plus de `max_parts` requêtes.

    `measure` mesure un morceau (caractères par défaut, `_url_encoded_len` pour une limite
    d'URL) ; `max_operators` borne en plus ses opérateurs (OpenAIRE : quatre). PUR/testable."""
    render = render or _boolean_to_generic
    measure = measure or len
    try:
        ast = _positive_boolean(_parse_boolean_ast(_tokenize_boolean(portable or "")), keep_not=True)
    except Exception:                                    # noqa: BLE001
        return []
    if ast is None:
        return []

    def _fits(node) -> bool:
        out = render(node)
        return (bool(out) and measure(out) <= limit
                and (max_operators is None or _count_operators(out) <= max_operators))

    def _split(node) -> list | None:
        if _fits(node):
            return [node]
        pieces = _split_node_once(node, render)      # None : un terme seul trop long
        if pieces is None:
            return None
        out: list = []
        for piece in pieces:
            sub = _split(piece)
            if sub is None:
                return None
            out.extend(sub)
            if len(out) > max_parts:
                return None
        return out

    nodes = _split(ast)
    if not nodes or len(nodes) > max_parts:
        return []
    return [render(n) for n in nodes]


# ─────────────────────────────────────────────────────────────────────────────
# Search (Hybride & Vectorielle pgvector)
# ─────────────────────────────────────────────────────────────────────────────

#: La borne des correspondances locales lues pour définir un corpus. Un LIMIT sans ORDER
#: BY : atteinte, la liste serait un sous-ensemble arbitraire. La bibliothèque compte
#: environ 494 000 documents : la borne est latente, et elle est DITE quand elle mord.
LOCAL_MATCH_LIMIT = 500_000


class _MatchIds(list):
    """Les identifiants d'une correspondance locale ; `capped` : la borne a été atteinte."""
    capped: bool = False


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
                model=_model("embedding"),
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
            # de document_search - sémantique PAR DOCUMENT, comme le chemin LIKE
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

    # Un document SANS résumé correspond lui aussi, sur son titre (et son texte intégral
    # s'il en a un) : la condition « résumé d'au moins 30 caractères » qui suivait chaque
    # correspondance écartait de la bibliothèque tout article sans résumé, même dont le
    # titre portait la requête. Il entre au corpus signalé (scenario_store.has_abstract_sql),
    # hors du sous-ensemble pertinent tant qu'un relecteur ne l'inclut pas à la main.
    if use_vector:
        params["q_emb"] = str(query_embedding)
        params["threshold"] = threshold
        sql = text(f"""
            SELECT DISTINCT d.id
            FROM document_chunk c
            JOIN literature_document d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
              AND (1 - (c.embedding <=> CAST(:q_emb AS vector))) > :threshold
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              {where_sql}
            LIMIT :limit
        """)
    elif mode == "boolean" and _fts:
        # Plein texte : un seul balayage d'index GIN sur document_search, puis jointure
        # par clé primaire pour les filtres (doublons, facettes). Mêmes filtres externes
        # que le chemin LIKE ci-dessous → même définition du corpus, seule la
        # correspondance des termes change (cf. lexical_search.py).
        sql = text(f"""
            SELECT d.id
            FROM document_search s
            JOIN literature_document d ON d.id = s.document_id
            WHERE ({any_match_sql})
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              {where_sql}
            LIMIT :limit
        """)
    elif mode == "boolean":
        # Chemin LIKE/trigramme : REPLI tant que document_search n'est pas rempli
        # (ou LEXICAL_SEARCH_ENGINE=like). Lent - 55 à 240 s par requête sur le corpus
        # de production - mais correct.
        # PERF : le match booléen est PAR DOCUMENT - `any_match_sql` n'apparie que
        # d.title / d.abstract + un EXISTS corrélé sur document_chunk ; il ne référence
        # PAS le chunk joint `c`. Piloter la requête depuis literature_document (~207k
        # lignes) au lieu de document_chunk (souvent 1M+ lignes, multipliées par doc
        # puis dédupliquées par DISTINCT) renvoie EXACTEMENT le même ensemble d'IDs en
        # balayant 5-10× moins de lignes - cause majeure de la lenteur de la recherche
        # locale et de la reconstruction du corpus (exécutée une fois PAR sous-requête).
        sql = text(f"""
            SELECT d.id
            FROM literature_document d
            WHERE ({any_match_sql})
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
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              {where_sql}
            LIMIT :limit
        """)

    import time as _time_ls
    _t0 = _time_ls.perf_counter()
    with engine.connect() as conn:
        ids = _MatchIds(conn.execute(sql, params).scalars().all())
    # Un LIMIT sans ORDER BY : atteint, la liste est un sous-ensemble ARBITRAIRE des
    # correspondances. Rien ne le disait ; le journal le dit, et la liste le porte
    # (`capped`) pour que l'appelant le dise dans sa réponse.
    ids.capped = bool(limit) and len(ids) >= int(limit)
    if ids.capped:
        logger.warning(f"lexical search: the limit of {limit} matches was reached - the list "
                       f"is an arbitrary subset of the matches - {query[:100]!r}")
    if mode == "boolean":
        # Une ligne par recherche, avec le moteur utilisé : c'est ce que l'on cherche
        # dans le journal quand « la recherche locale est lente ».
        logger.info(f"lexical search [{'fts' if _fts else 'like'}] {len(ids)} docs in "
                    f"{(_time_ls.perf_counter() - _t0) * 1000:.0f} ms - {query[:100]!r}")
        if _fts:
            _ignored = _lex.stopword_terms(_lex.ast_terms(_ast))
            if _ignored:
                logger.warning(f"lexical search: terms made only of stop words were "
                               f"ignored by PostgreSQL: {_ignored} - {query[:100]!r}")
    return ids


# Limite de récupération par source live (PubMed, OpenAlex, …). Appliquée à l'identique
# à la recherche ET à la construction du corpus. Réglable via l'env LIVE_MAX_PER_SOURCE :
# la mettre très haut (p. ex. 100000) « retire » le plafond - la vraie borne devient alors
# le budget temps (POPULATE_FEDERATION_BUDGET). ⚠ multiplier ce plafond multiplie les
# appels API (risque de 429 PubMed/S2) ET le coût d'embedding OpenAI de CHAQUE recherche.
try:
    LIVE_MAX_PER_SOURCE = int(os.getenv("LIVE_MAX_PER_SOURCE", "2000"))
except (TypeError, ValueError):
    LIVE_MAX_PER_SOURCE = 2000

# Le plafond des passages CLASSÉS PAR PERTINENCE : Crossref, qui ne cherche que des
# mots-clés (2,6 millions de réponses sur la question HPAI, une queue de bruit), et le
# passage « meilleures correspondances en texte intégral » d'OpenAlex. Il ne suit PAS
# LIVE_MAX_PER_SOURCE : mis très haut pour lever le plafond d'une recherche standard, ce
# dernier faisait paginer ces deux passages jusqu'au refus de l'API (400 au-delà de
# 10 000 résultats paginés par décalage ou par numéro de page), et le passage d'OpenAlex
# versait au corpus des milliers d'articles qui ne répondent au booléen que dans leur
# texte intégral. Jamais au-delà de 10 000, la borne de ces API.
try:
    RANKED_PASS_MAX = max(1, min(int(os.getenv("RANKED_PASS_MAX", "2000")), 10000))
except (TypeError, ValueError):
    RANKED_PASS_MAX = 2000

# Le plafond d'une recherche EXHAUSTIVE, par base appariant le booléen. Ce n'est pas un
# échantillon : c'est un garde-fou, assez haut pour qu'aucune base ne soit coupée sur une
# question de revue (Europe PMC rend 13 810 notices sur la question HPAI, toutes
# récupérées), assez bas pour qu'une stratégie écrite trop large ne verse pas un index
# entier dans la bibliothèque. Une base qui le dépasse est dite « plafonnée » sur la carte.
try:
    EXHAUSTIVE_MAX_PER_SOURCE = int(os.getenv("EXHAUSTIVE_MAX_PER_SOURCE", "20000"))
except (TypeError, ValueError):
    EXHAUSTIVE_MAX_PER_SOURCE = 20000


def _boolean_corpus_ids(boolean_query: str, filters: dict) -> list:
    """LA source de vérité de l'appartenance au corpus : les documents de la base
    locale qui correspondent à la requête booléenne. Recherche et corpus utilisent
    EXACTEMENT ce helper → le compteur de la recherche == la taille du corpus.
    `capped` sur la liste rendue : LOCAL_MATCH_LIMIT a été atteinte."""
    return _search_local_doc_ids(boolean_query, "boolean", filters, limit=LOCAL_MATCH_LIMIT)


def _looks_boolean(text: str) -> bool:
    """Le texte utilise-t-il une SYNTAXE booléenne (→ utilisé tel quel) plutôt que du
    langage naturel (→ traduit) ? Signaux : opérateurs AND/OR/NOT en MAJUSCULES, tags
    de champ ([dp], [tiab], [mesh]…), guillemets doubles appariés, ou parenthèses.
    Pur/déterministe - le même heuristique est reflété côté client (App.tsx:looksBoolean)."""
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
    utilisateur) ; sinon (auto|absent|invalide) il est DÉTECTÉ par _looks_boolean -
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
    forcément au corpus - les enregistrements booléens-natifs des sources live (qui
    n'ont vu que la requête principale) ne doivent donc pas être unis d'office."""
    return "and" in _facet_ops(facets, combinator)


def _combined_query_text(query: str | None, sub_queries: Any, combinator: str | None) -> str:
    """Expression lisible de la recherche COMPLÈTE : requête principale + sous-requêtes
    avec leurs opérateurs, parenthésée selon le fold gauche→droite réellement appliqué
    (« (A) AND (B) », « ((A) OR (B)) AND (C) »). Mono-requête → la requête telle quelle.
    C'est ce texte qui doit apparaître partout où la recherche est montrée (nom par
    défaut, carte, en-tête, onglet Stratégie) - la colonne `query` ne porte que la
    facette principale, d'où un « ET » invisible auparavant."""
    clean = _normalize_sub_queries(sub_queries)
    if len(clean) < 2:
        return (query or "").strip()
    expr = clean[0]["text"]
    for facet, op in zip(clean[1:], _facet_ops(clean, combinator or "union")):
        expr = f"({expr}) {op.upper()} ({facet['text']})"
    return expr


#: La longueur au-delà de laquelle une facette OU n'élargit plus le booléen fédéré. Les
#: sources à limite d'URL reçoivent leur propre forme (réduite, ou découpée en recherche
#: exhaustive), PubMed reçoit le sien en POST et CORE dans un corps JSON : la borne qui
#: reste est celle d'Europe PMC et de Semantic Scholar, qui reçoivent le booléen entier
#: dans un GET. La requête HPAI de production (3 075 caractères) y passe ; 4 000 laisse
#: une facette s'y ajouter, sous les 8 Ko d'URL que ces serveurs acceptent une fois encodé.
FACET_WIDEN_MAX_CHARS = 4000


def _widen_boolean_for_or_facets(boolean: str, pubmed_q: str, sub_queries: Any, combinator: str | None,
                                 translate=None, max_len: int = FACET_WIDEN_MAX_CHARS,
                                 max_pubmed_len: int | None = None,
                                 skipped: list | None = None) -> tuple[str, str, int]:
    """Élargit les requêtes envoyées aux sources LIVE aux facettes UNIES (OU) d'une
    recherche multi-facettes : « (booléen principal) OR (booléen de la facette) ».

    La fédération n'interrogeait que la requête PRINCIPALE : une facette « OU » ne
    trouvait que ce que la base locale contenait déjà - des articles qui ne
    correspondent qu'à elle n'étaient jamais ramenés de PubMed/Europe PMC/OpenAlex.
    Les facettes intersectées (ET) n'ont pas besoin d'être fédérées : leurs résultats
    sont un sous-ensemble de ceux de la requête principale, re-matché localement.
    Une facette naturelle est traduite via `translate` (par défaut
    _generate_search_strategy) ; sans traduction utilisable, son texte est utilisé
    tel quel. La longueur du booléen élargi est bornée (`max_len`, celle des sources qui
    le reçoivent entier dans une URL) ; PubMed ne l'est plus (POST au-delà de la limite
    d'URL, cf. sources._ncbi_get), sauf `max_pubmed_len` explicite. Une facette qui ne
    tient pas reste hors de la fédération : son texte est ajouté à `skipped`, que la
    recherche enregistre et que la carte PRISMA nomme (un avertissement dans le journal
    ne le disait à personne). Renvoie (booléen, requête PubMed, nb de facettes ajoutées)."""
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
            except Exception as _e:                      # noqa: BLE001 - repli texte brut
                logger.warning(f"facette OU « {facet['text'][:60]} » : traduction échouée ({_e}) ; texte brut")
        portable = _strip_field_tags(fb).strip() or fb
        if portable in boolean:
            continue
        new_bool = f"({boolean}) OR ({portable})"
        new_pub = f"({pubmed_q}) OR ({fp or portable})" if pubmed_q else (fp or portable)
        if len(new_bool) > max_len or (max_pubmed_len is not None and len(new_pub) > max_pubmed_len):
            logger.warning(f"facette OU « {facet['text'][:60]} » ignorée pour la fédération live : requête trop longue")
            if skipped is not None:
                skipped.append(facet["text"])
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
    par correspondance LEXICALE (booléenne) - EXACTEMENT comme la recherche
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
    capped = False
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
        _ids = _search_local_doc_ids(_boolean, "boolean", filters, limit=LOCAL_MATCH_LIMIT)
        capped = capped or bool(getattr(_ids, "capped", False))
        id_sets.append(set(_ids))
    if not id_sets:
        return _MatchIds()
    # Fold gauche→droite selon l'op PAR facette (défaut = `combinator` global).
    out = _MatchIds(_fold_facet_sets(id_sets, clean, combinator))
    out.capped = capped
    return out


def _set_scenario_corpus(scenario_id: str, ids: list, allow_empty: bool = False,
                         keep: list | None = None) -> int:
    """Fixe le corpus d'un scénario à `ids` (appartenance booléenne) : insère les liens
    manquants et supprime ceux qui n'en font plus partie, SAUF trois sortes de liens.

      - Ceux qui portent la décision d'un relecteur (`reviewer_decision_sql`). La
        suppression était inconditionnelle : une recherche relancée effaçait une inclusion
        à la main avec son motif et ses votes dès que sa requête n'appariait plus
        l'article. Ils restent, et la carte PRISMA les compte (`kept_reviewed_links`).
      - Ceux ajoutés à la main (`added_by_hand_sql` : un fichier déposé par un relecteur,
        api/additions.py). Même raison : un article qu'une personne a mis dans le corpus
        n'en sort pas parce qu'une requête ne le retrouve pas. La carte les compte
        (`kept_added_links`).
      - Ceux dont le document est dans `keep` : le corpus d'avant une recherche dont la
        fédération n'a pas abouti (une source en échec, ou coupée par le budget de
        temps). Une source qui n'a pas répondu ne dit rien des liens qu'elle aurait
        ramenés : le 12 octobre 2026, Europe PMC répondait 503 pendant une recherche
        exhaustive de HPAI_last, et le corpus est retombé à 8 494 articles au lieu de
        15 000 à 18 000, sans ce que seul son booléen natif apporte (MeSH, synonymes). La
        carte les compte (`kept_previous_links`) et nomme les sources.

    Si `ids` est vide on ne touche à rien (évite de vider le corpus sur un échec
    transitoire), SAUF si allow_empty=True - cas d'une intersection multi-requêtes
    légitimement vide (deux facettes sans document commun), où le corpus DOIT être
    vidé, les deux sortes de liens ci-dessus exceptées."""
    if not ids and not allow_empty:
        return 0
    with engine.begin() as _c:
        # Un TABLEAU côté serveur, pas une liste de paramètres qui s'étend : la version
        # `NOT IN :ids` expansée fabriquait un paramètre par identifiant, et Postgres
        # refuse au-delà de 65 535. La suppression levait donc sur tout corpus de plus
        # de ~65 000 articles, l'exception était avalée par l'appelant, et la recherche
        # s'annonçait terminée sur un corpus dont les liens obsolètes étaient restés.
        # Même forme que l'INSERT juste en dessous, qui utilisait déjà un tableau.
        _c.execute(text(f"""
            DELETE FROM article_scenarios a
            WHERE a.scenario_id = :sid
              AND NOT (a.document_id = ANY(CAST(:ids AS bigint[])))
              AND NOT (a.document_id = ANY(CAST(:keep AS bigint[])))
              AND NOT {reviewer_decision_sql('a')}
              AND NOT {_added_by_hand_predicate('a')}
        """), {"sid": scenario_id, "ids": list(ids), "keep": [int(k) for k in (keep or [])]})
        if ids:
            # Insertion en masse (un seul aller-retour) plutôt qu'une requête par
            # document : la (ré)construction du corpus local doit être quasi immédiate.
            _c.execute(text("""
                INSERT INTO article_scenarios (document_id, scenario_id, similarity_score)
                SELECT unnest(CAST(:ids AS bigint[])), :s, NULL
                ON CONFLICT (document_id, scenario_id) DO NOTHING
            """), {"ids": list(ids), "s": scenario_id})
    return len(ids)


def _ids_within_criteria(ids, filters: dict | None) -> list[int]:
    """Les documents de `ids` qui satisfont les critères propres de la recherche : ses
    bornes d'années et, quand elle les exclut, l'absence de préprint.

    La correspondance locale les applique déjà (_build_where), mais l'union des sources
    booléennes-natives entre au corpus sans re-correspondance : une recherche
    « 2015-2024 » gardait les notices PubMed de 1987, une recherche sans préprints ceux
    qu'OpenAlex, Semantic Scholar ou CORE rendent. Deux DELETE les retiraient APRÈS
    l'assemblage ; le filtre passe désormais AVANT, pour que l'appartenance écrite, les
    liens gardés d'une recherche incomplète et les comptes de la carte partent du même
    ensemble. Même sémantique que ces DELETE : un document n'est écarté que si la
    condition le dit VRAIMENT (une année inconnue n'est pas hors période)."""
    ids = [int(i) for i in (ids or [])]
    _f = filters or {}
    conds: list[str] = []
    params: dict[str, Any] = {}
    if excludes_preprints(_f):
        conds.append(f"({preprint_sql('d')}) IS NOT TRUE")
    _years = {k: _f.get(k) for k in ("year_min", "year_max") if _f.get(k) not in (None, "")}
    if _years:
        _yw, _yp = _build_where(_years)
        conds.append(f"(TRUE {_yw}) IS TRUE")
        params.update(_yp)
    if not ids or not conds:
        return ids
    with engine.connect() as _c:
        rows = _c.execute(text(f"SELECT d.id FROM literature_document d "
                               f"WHERE d.id = ANY(CAST(:ids AS bigint[])) AND {' AND '.join(conds)}"),
                          {"ids": ids, **params}).scalars().all()
    return [int(r) for r in rows]


#: `scenario_additions` existe-t-elle ? Mémorisé une fois VRAI (la table est créée au
#: démarrage, cf. api/additions.py), re-sondé tant que la réponse est non : un import qui
#: poserait la question avant la DDL ne doit pas figer « non » pour la vie du processus.
_ADDITIONS_TABLE = [False]


def _added_by_hand_predicate(link: str = "a") -> str:
    """`added_by_hand_sql`, ou `FALSE` quand la table des ajouts n'existe pas.

    Le prédicat est sur le CHEMIN DE LA RECHERCHE (_set_scenario_corpus) : sur une base
    antérieure à la fonctionnalité, ou dont la DDL n'a pas pu passer, une sous-requête vers
    une table absente ferait échouer le DELETE, donc toute recherche. `FALSE` rend
    exactement le comportement d'avant : aucun lien n'est tenu pour ajouté à la main."""
    if not _ADDITIONS_TABLE[0]:
        try:
            with engine.connect() as _c:
                _ADDITIONS_TABLE[0] = bool(_c.execute(text(
                    "SELECT to_regclass('scenario_additions') IS NOT NULL")).scalar())
        except Exception as _e:                              # noqa: BLE001 - jamais bloquant
            logger.warning(f"_added_by_hand_predicate: {_e}")
            return "FALSE"
    return added_by_hand_sql(link) if _ADDITIONS_TABLE[0] else "FALSE"


def _kept_link_counts(scenario_id: str, matched) -> dict[str, int]:
    """Les liens du corpus que la recherche n'a PAS appariés (hors de `matched`) et qui
    sont restés, par raison : la décision d'un relecteur (`reviewed`), sinon un ajout à
    la main (`added`), sinon le corpus d'avant une fédération incomplète (`previous`).

    Lu après l'assemblage et la dédup, sur les documents non doublons : ce sont des
    « passés au screening » que la recherche n'a pas produits. Sans leur ligne,
    identifiés moins doublons moins retraits ne menait plus au corpus, ou les retraits
    étaient rognés pour que le compte retombe juste."""
    _decided = reviewer_decision_sql("a")
    _added = _added_by_hand_predicate("a")
    with engine.connect() as _c:
        row = _c.execute(text(f"""
            SELECT COUNT(DISTINCT a.document_id) FILTER (WHERE {_decided})                    AS reviewed,
                   COUNT(DISTINCT a.document_id) FILTER (WHERE NOT {_decided} AND {_added})     AS added,
                   COUNT(DISTINCT a.document_id) FILTER (WHERE NOT {_decided} AND NOT {_added}) AS previous
            FROM article_scenarios a
            JOIN literature_document d ON d.id = a.document_id
            WHERE a.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
              AND NOT (a.document_id = ANY(CAST(:ids AS bigint[])))
        """), {"sid": scenario_id, "ids": [int(i) for i in (matched or [])]}).mappings().first() or {}
    return {"reviewed": int(row.get("reviewed") or 0), "added": int(row.get("added") or 0),
            "previous": int(row.get("previous") or 0)}


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
    `document_id` (le canonique - MÊME règle que `scripts/_softdedup.py`, donc aucun
    « glissement » quand la dédup globale tournera : elle marquera exactement les id
    supérieurs déjà retirés ici). Aucune fusion abusive : seules des clés EXACTES
    (même DOI, même external_id normalisé, ou même titre long) sont réunies ; un
    external_id vide retombe sur le titre puis sur l'id (jamais fusionné). Renvoie le
    nombre de liens supprimés. Tolérant aux pannes (journalise et renvoie 0).

    Le lien supprimé est FUSIONNÉ dans le survivant avant de disparaître. La règle « on
    garde le plus petit document_id » est purement arbitraire vis-à-vis du contenu : le
    lien du plus grand id pouvait porter la décision d'un relecteur, ses motifs, ses
    votes de double aveugle et ses deux scores, et la suppression les emportait sans
    laisser de trace. Un relecteur voyait son exclusion revenir en « en attente » après
    une relance de la recherche. On ne remplace jamais une valeur du survivant, on ne
    comble que ses trous, et le choix du canonique ne change pas (même règle que
    `scripts/_softdedup.py`, donc aucun glissement quand la dédup globale tournera).

    Le titre ne réunit JAMAIS deux DOI de revue différents : deux articles au titre
    générique (« Avian influenza update », dix éditions) sont deux travaux. Mais la clé
    est le DOI du TRAVAIL (`documents.same_work_survivors`) : la ponctuation finale, la
    casse et les versions (/v1, /v2) ne séparent plus, et sous un même titre long un
    préprint ou une figure déposée sur figshare rejoint la version publiée. Sur la
    recherche HPAI du 12 octobre, 40 paires d'un même article entraient ainsi deux fois
    dans le sous-ensemble pertinent, et l'extraction comptait l'étude deux fois. Une
    notice SANS DOI rejoint le travail de même titre long quand il n'y en a qu'un : c'est
    la règle de l'ingestion (`_ingest_doc_direct`), un DOI manquant ne sépare pas.

    Le survivant est la version publiée plutôt que le préprint ou le dépôt, puis le plus
    petit identifiant, comme la dédup globale. Les notices sans résumé ne sortent plus du
    corpus avant la dédup : quand le survivant n'a pas de
    résumé et qu'un de ses doublons en a un, le résumé est recopié sur le survivant (on
    comble un champ vide, on ne remplace rien), et son chunk titre+résumé reconstruit."""
    from .sources import _refresh_title_abstract_chunks  # lazy: sources is loaded after this module
    _MERGED = ("similarity_score", "rerank_score", "screening_status", "screening_reason",
               "screening_notes", "screened_at", "reviewer_1_status", "reviewer_1_reason",
               "reviewer_2_status", "reviewer_2_reason", "kappa_final_status",
               "cluster_id", "cluster_label")
    # Le survivant de chaque doublon, calculé une fois en Python (la règle du travail ne
    # s'écrit pas en SQL), puis servi aux trois instructions comme une table.
    _RANKED = """
        ranked AS (
            SELECT x.document_id, x.keep_id
            FROM jsonb_to_recordset(CAST(:pairs AS jsonb)) AS x(document_id bigint, keep_id bigint)
        )
    """
    try:
        with engine.connect() as _r:
            _rows = _r.execute(text("""
                SELECT a.document_id AS id, d.doi, d.title_norm, d.external_id
                FROM article_scenarios a
                JOIN literature_document d ON d.id = a.document_id
                WHERE a.scenario_id = :sid
            """), {"sid": scenario_id}).mappings().all()
        _survivor = same_work_survivors(_rows)
        if not _survivor:
            return 0
        _p = {"sid": scenario_id, "pairs": json.dumps(
            [{"document_id": d, "keep_id": k} for d, k in sorted(_survivor.items())])}
        with engine.begin() as _c:
            # ── Fusion AVANT suppression ─────────────────────────────────────
            # Un seul donneur par survivant : celui qui porte le plus d'information
            # (une décision de screening d'abord, puis un score de rerank, puis un
            # score de similarité), pour que deux doublons ne se contredisent pas.
            _c.execute(text(f"""
                WITH {_RANKED},
                donor AS (
                    SELECT DISTINCT ON (r.keep_id) r.keep_id, a.*
                    FROM ranked r
                    JOIN article_scenarios a
                      ON a.scenario_id = :sid AND a.document_id = r.document_id
                    WHERE r.document_id <> r.keep_id
                    ORDER BY r.keep_id,
                             (a.screening_status IS NOT NULL) DESC,
                             (a.reviewer_1_status IS NOT NULL) DESC,
                             (a.rerank_score IS NOT NULL) DESC,
                             (a.similarity_score IS NOT NULL) DESC,
                             a.document_id
                )
                UPDATE article_scenarios k
                SET {", ".join(f"{c} = COALESCE(k.{c}, donor.{c})" for c in _MERGED)}
                FROM donor
                WHERE k.scenario_id = :sid AND k.document_id = donor.keep_id
            """), _p)
            # ── Le résumé d'un doublon comble celui du survivant ─────────────
            _filled = _c.execute(text(f"""
                WITH {_RANKED},
                src AS (
                    SELECT DISTINCT ON (r.keep_id) r.keep_id, d.abstract
                    FROM ranked r
                    JOIN literature_document d ON d.id = r.document_id
                    WHERE r.document_id <> r.keep_id AND {has_abstract_sql('d')}
                    ORDER BY r.keep_id, r.document_id
                )
                UPDATE literature_document k
                SET abstract = src.abstract
                FROM src
                WHERE k.id = src.keep_id AND NOT {has_abstract_sql('k')}
                RETURNING k.id
            """), _p).scalars().all()
            if _filled:
                _refresh_title_abstract_chunks(_c, _filled)
            n = _c.execute(text(f"""
                WITH {_RANKED}
                DELETE FROM article_scenarios a
                USING ranked r
                WHERE a.scenario_id = :sid
                  AND a.document_id = r.document_id
                  AND r.document_id <> r.keep_id
            """), _p).rowcount
        if n:
            logger.info(f"Dédup intra-scénario {scenario_id}: {n} lien(s) doublon(s) "
                        f"supprimé(s) (même travail : DOI, PMID ou titre).")
        return n or 0
    except Exception as _e:
        logger.warning(f"Dédup intra-scénario {scenario_id}: {_e}")
        return 0


#: Les issues possibles d'une source, dans l'ordre où elles se lisent. `ok` et `empty`
#: sont les deux seules qui veuillent dire « interrogée, et sa réponse est dans ces
#: chiffres » ; `cached` les rejoint, la réponse venant d'un appel antérieur identique.
SOURCE_OUTCOMES = ("ok", "empty", "cached", "skipped", "error", "cut_by_budget")
#: Celles qui comptent dans la couverture annoncée.
SOURCE_OUTCOMES_COUNTED = ("ok", "empty", "cached")


def _outcome_summary(outcomes: dict) -> str:
    """« ok: 4 | empty: 2 | error: 5 | skipped: 1 », pour le journal et le message."""
    counts: dict[str, int] = {}
    for v in (outcomes or {}).values():
        counts[str(v)] = counts.get(str(v), 0) + 1
    return " | ".join(f"{k}: {counts[k]}" for k in SOURCE_OUTCOMES if k in counts) or "aucune"


def _source_label(fetcher_name: str) -> str:
    """Le nom de source derrière un nom de fetcher (`_fetch_europepmc` → `europepmc`)."""
    return str(fetcher_name or "").replace("_fetch_", "", 1)


#: La ou les clés sous lesquelles un fetcher ÉCRIT ses enregistrements, quand elles
#: diffèrent de son propre nom : `_fetch_preprints` écrit sous « preprint » au singulier.
#: Le tableau d'identification portait donc une ligne fantôme à zéro juste à côté de la
#: vraie (« preprint 5 » et « preprints 0 »), et l'issue de la source se décidait sur un
#: compteur qui n'était pas le sien. (`_fetch_biorxiv_medrxiv`, qui écrivait sous
#: « biorxiv » et « medrxiv », a quitté la fédération : l'API bioRxiv ne cherche pas par
#: mots-clés, et Europe PMC indexe ces deux serveurs. Les chiffres déjà enregistrés sont
#: stockés tout calculés ; l'interface replie encore ses clés pour les afficher.)
SOURCE_RECORD_KEYS = {
    "preprints": ("preprint",),
}


def source_record_keys(name: str) -> tuple[str, ...]:
    """Les clés de `records_by_source` qu'alimente ce fetcher (ou cette source)."""
    label = _source_label(name)
    return SOURCE_RECORD_KEYS.get(label, (label,))


def _coverage_caveat(outcomes: dict) -> str:
    """Ce que la ligne de couverture doit dire en plus du nombre de sources.

    « Sources interrogées : 7/12 » laissait au lecteur le soin de deviner ce qu'étaient
    les cinq autres. Une recherche dont la moitié des sources ont échoué n'est pas une
    recherche sur sept sources, et un relecteur qui cite cette ligne doit pouvoir le
    lire sans ouvrir les journaux."""
    buckets = {o: sorted(_source_label(k) for k, v in (outcomes or {}).items() if v == o)
               for o in ("error", "skipped", "cut_by_budget")}
    parts = []
    if buckets["error"]:
        parts.append(f"en échec : {', '.join(buckets['error'])}")
    if buckets["cut_by_budget"]:
        parts.append(f"coupées par le budget de temps : {', '.join(buckets['cut_by_budget'])}")
    if buckets["skipped"]:
        parts.append(f"non interrogées (clé d'API absente) : {', '.join(buckets['skipped'])}")
    return (" ; " + " ; ".join(parts)) if parts else ""


def _prisma_identification_figures(records_by_source: dict, unique_records: int,
                                   duplicate_rows_removed: int, corpus_total: int,
                                   method: str = "populate",
                                   federation_incomplete: bool = False,
                                   removed_no_abstract: int = 0,
                                   removed_not_matching: int = 0,
                                   source_outcomes: dict | None = None,
                                   per_source_cap: int | None = None,
                                   records_from_library: int = 0,
                                   keyword_fallback_sources: list | None = None,
                                   keyword_fallback_query: str | None = None,
                                   source_error_reasons: dict | None = None,
                                   source_totals: dict | None = None,
                                   keyword_fallback_queries: dict | None = None,
                                   search_mode: str = "standard",
                                   source_caps: dict | None = None,
                                   title_abstract_sources: list | None = None,
                                   unranked_sources: list | None = None,
                                   split_queries: dict | None = None,
                                   preprints_excluded: bool = False,
                                   removed_preprints: int = 0,
                                   preprints_excluded_at_source: list | None = None,
                                   preprint_sources_not_searched: list | None = None,
                                   kept_no_abstract: int = 0,
                                   facets_skipped: list | None = None,
                                   local_match_capped: bool = False,
                                   kept_previous_links: int = 0,
                                   kept_previous_failed_sources: list | None = None,
                                   kept_reviewed_links: int = 0,
                                   kept_added_links: int = 0,
                                   year_slices: dict | None = None,
                                   year_slices_capped: dict | None = None) -> dict[str, Any]:
    """Chiffres PRISMA 2020 de l'étape « identification », calculés à partir de ce qu'une
    recherche a RÉELLEMENT ramené - et non du corpus déjà dédupliqué.

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
    corpus_total           : liens restants après nettoyage (doublons, préprints exclus)
    source_outcomes        : issue de CHAQUE source lancée (cf. SOURCE_OUTCOMES). Une
                             source à zéro n'est pas une source absente : sans ce
                             paramètre, le tableau ne nommait que celles qui avaient
                             rapporté quelque chose et annonçait pourtant un nombre de
                             sources interrogées.
    per_source_cap         : le plafond par source RÉELLEMENT appliqué à ce run, pour
                             qu'un chiffre puisse être rattaché au run qui l'a produit.
    records_from_library   : les correspondances déjà présentes dans la bibliothèque
                             locale. PRISMA 2020 les met dans « autres méthodes », pas
                             dans « bases de données interrogées » : les y compter
                             gonflait les identifiés ET les doublons de tout article
                             trouvé à la fois en local et en ligne.
    keyword_fallback_*     : les sources qui ont reçu des MOTS-CLÉS et non la requête
                             booléenne, et ces mots-clés. Au-delà de 1 200 caractères de
                             booléen portable, cinq sources sur douze basculent en repli
                             (limite d'URL d'OpenAlex). Rien ne le disait : le tableau
                             montrait leurs notices sous l'étiquette d'une recherche
                             booléenne, et PRISMA-S demande la stratégie RÉELLEMENT
                             soumise à chaque base. Sur le scénario HPAI, arXiv et CORE
                             ont ainsi rapporté 12 438 notices à une requête de huit mots
                             dont aucun ne nommait la grippe.
    search_mode            : « exhaustive » (toutes les notices que chaque base appariant
                             le booléen renvoie), le seul mode d'une recherche depuis le
                             13 octobre 2026 ; « standard » (au plus `per_source_cap`
                             notices par source, dans l'ordre de pertinence de la source)
                             reste celui des cartes des recherches d'avant.
    source_caps            : le plafond APPLIQUÉ à chaque source quand il n'est pas le même
                             pour toutes : Crossref, qui classe des mots-clés au lieu
                             d'appliquer le booléen, garde ses RANKED_PASS_MAX plus
                             pertinentes. « Plafonnée » se juge contre le plafond de la
                             source.
    title_abstract_sources : les sources dont TOUTES les correspondances sur titre et résumé
                             ont été prises (OpenAlex en mode exhaustif : sa recherche par
                             défaut lit aussi le texte intégral, 43 181 notices sur HPAI
                             contre 1 648 pour PubMed), en plus des plus pertinentes de leur
                             recherche par défaut, comme en mode standard.
    unranked_sources       : les sources dont l'ordre N'EST PAS la pertinence (Semantic
                             Scholar, endpoint bulk : ordre des identifiants). Au plafond,
                             ce qui est gardé y est un sous-ensemble arbitraire.
    split_queries          : les sources qui ont reçu le booléen entier découpé en
                             plusieurs requêtes dont l'union est le booléen (DOAJ ne pagine
                             pas une requête longue ; en recherche exhaustive, toute base
                             que l'URL ou le nombre d'opérateurs limite).
    preprints_excluded     : la recherche exclut les préprints (critère choisi par
                             l'utilisateur). Les serveurs de préprints ne sont alors pas
                             interrogés (`preprint_sources_not_searched`), certaines bases
                             les excluent dans la requête elle-même
                             (`preprints_excluded_at_source` : « NOT preprint[pt] » pour
                             PubMed, « NOT SRC:PPR » pour Europe PMC), et ceux que les
                             autres bases rendent sont retirés avant le screening
                             (`removed_preprints`), sur leur propre ligne.
    kept_no_abstract       : les notices SANS résumé gardées au corpus. Elles étaient
                             retirées (`removed_no_abstract`, 2 434 sur HPAI_last) ; elles
                             sont désormais passées au screening, signalées, à juger sur
                             leur titre à la main : un sous-compte de `records_screened`,
                             pas un retrait. `removed_no_abstract` reste servi (0 pour une
                             recherche nouvelle) pour qui le lit encore.
    facets_skipped         : les facettes OU qu'aucune base n'a reçues (requête trop
                             longue) : elles n'apportent que ce que la bibliothèque contient.
    local_match_capped     : la correspondance locale a atteint sa borne (LOCAL_MATCH_LIMIT) ;
                             le corpus en est alors un sous-ensemble arbitraire.
    kept_previous_links    : les liens du corpus d'AVANT la recherche qu'elle n'a pas
                             appariés et qui restent parce que sa fédération n'a pas abouti
                             (`kept_previous_failed_sources` : les sources en échec ou
                             coupées par le budget). Une source qui n'a pas répondu ne dit
                             rien de ce qu'elle aurait ramené ; une recherche complète les
                             retire.
    kept_reviewed_links    : les liens que la recherche n'a pas appariés et qui restent parce
                             qu'un relecteur en a décidé (inclus, exclu, vote, arbitrage).
    kept_added_links       : ceux qui restent parce qu'un relecteur les a ajoutés à la main
                             (un fichier déposé, api/additions.py), sans décision encore.
    year_slices            : les sources dont la requête a été lue par tranches d'années de
                             publication, et ces tranches (PubMed : eutils ne rend pas plus
                             de 10 000 identifiants par recherche). Le total de la source
                             reste celui de la requête entière.
    year_slices_capped     : parmi elles, les tranches d'une seule année qui en annonçaient
                             encore plus de 10 000, et ce qu'elles annonçaient : seules leurs
                             10 000 plus pertinentes ont pu être lues.

    PRISMA 2020 : identifiés → doublons retirés → (retirés pour d'autres raisons) →
    passés au screening. « Autres raisons » ici : préprint quand la recherche les exclut,
    ou enregistrement d'une source par mots-clés qui ne correspond pas à la requête
    booléenne en local. Les liens gardés sans avoir été appariés s'y AJOUTENT, sur leurs
    lignes : identifiés − doublons − retraits + gardés = passés au screening."""
    from datetime import datetime as _dt, timezone as _tz
    _raw = {str(k): int(v or 0) for k, v in (records_by_source or {}).items()}
    library = max(0, int(records_from_library or 0) or _raw.pop("db_cache", 0))
    _raw.pop("db_cache", None)
    outcomes = {_source_label(k): str(v) for k, v in (source_outcomes or {}).items()}
    # Les sources LANCÉES, y compris celles qui n'ont rien rapporté : le tableau doit
    # porter la ligne et son issue, sinon le lecteur ne peut pas savoir que PubMed a
    # échoué, il peut seulement constater que PubMed n'y est pas. Et chaque source sous
    # SON nom : un fetcher qui écrit ses enregistrements sous une autre clé (« preprint »
    # pour `_fetch_preprints`, et naguère « biorxiv » et « medrxiv ») donnait deux lignes
    # à la carte, « preprint 249 » sans issue et « preprints 0 · a répondu », parce que
    # l'issue et le compte ne se retrouvaient pas sous la même clé.
    records: dict[str, int] = {}
    folded: set[str] = set()
    for name in outcomes:
        keys = source_record_keys(name)
        records[name] = sum(_raw.get(k, 0) for k in keys)
        folded.update(keys)
    for k, v in _raw.items():
        if k not in folded and v > 0:
            records[k] = v
    by_outcome = {o: sorted(k for k, v in outcomes.items() if v == o) for o in SOURCE_OUTCOMES}
    searched = sum(len(by_outcome[o]) for o in SOURCE_OUTCOMES_COUNTED)
    totals = {}
    for k, v in (source_totals or {}).items():
        try:
            totals[_source_label(k)] = int(v)
        except (TypeError, ValueError):
            continue
    # « Plafonnée » = l'API a annoncé PLUS que le plafond par source : c'est le plafond qui
    # a coupé, et le compte est un plancher. Un écart SOUS le plafond n'en est pas un :
    # sur le premier run de production après #326, PubMed avait 1 176 gardés sur 1 182
    # annoncés sous un plafond de 2 000 (six notices perdues au nettoyage), et la carte la
    # disait « plafonnée », avec la phrase sur les 2 000 plus pertinents. Le « n / total »
    # reste affiché, lui, dès que l'API a annoncé plus que ce qu'on a gardé. Une source
    # coupée par le budget ou en échec n'est pas « plafonnée » non plus : son issue le dit.
    caps: dict[str, int] = {}
    for k, v in (source_caps or {}).items():
        try:
            caps[_source_label(k)] = int(v)
        except (TypeError, ValueError):
            continue

    def _cap_of(name: str) -> int:
        return int(caps.get(name, per_source_cap or 0) or 0)

    capped = sorted(name for name in records
                    if _cap_of(name) > 0 and totals.get(name) is not None
                    and totals[name] > _cap_of(name)
                    and outcomes.get(name, "ok") in SOURCE_OUTCOMES_COUNTED)
    from_databases = sum(records.values())
    identified = from_databases + library
    across = max(0, identified - max(0, int(unique_records or 0)))
    rows = max(0, int(duplicate_rows_removed or 0))
    duplicates = min(identified, across + rows)
    unique_after = identified - duplicates
    corpus = max(0, int(corpus_total or 0))
    # Les liens restés au corpus sans que la recherche les apparie : décision d'un
    # relecteur, puis corpus d'avant une fédération incomplète. Ils sont parmi les passés
    # au screening, jamais au-delà.
    kept_reviewed = min(corpus, max(0, int(kept_reviewed_links or 0)))
    kept_added = min(corpus - kept_reviewed, max(0, int(kept_added_links or 0)))
    kept_previous = min(corpus - kept_reviewed - kept_added, max(0, int(kept_previous_links or 0)))
    # Retraits AVANT screening, ventilés : sans résumé (l'ancienne règle, 0 désormais),
    # préprints d'une recherche qui les exclut, hors requête (enregistrement d'une source
    # par mots-clés qui ne correspond pas au booléen en local), et le reste (résiduel -
    # p. ex. un document marqué doublon global). Chacun est borné à ce qui reste à
    # expliquer, dans cet ordre, pour que identifiés − doublons − retraits + gardés =
    # passés au screening tienne toujours.
    to_explain = max(0, unique_after + kept_reviewed + kept_added + kept_previous - corpus)
    # Zéro pour une recherche nouvelle (les notices sans résumé sont gardées, cf.
    # `kept_no_abstract`) ; le paramètre reste pour un appelant qui en retire encore.
    no_abstract = min(to_explain, max(0, int(removed_no_abstract or 0)))
    preprints = min(to_explain - no_abstract, max(0, int(removed_preprints or 0)))
    not_matching = min(to_explain - no_abstract - preprints, max(0, int(removed_not_matching or 0)))
    other = to_explain - no_abstract - preprints - not_matching
    # La requête RÉELLEMENT soumise à chaque source en repli. Elles ne reçoivent pas toutes
    # la même : les sources à la limite d'URL partagent une réduction, arXiv a la sienne
    # (syntaxe `all:`), OpenAIRE la sienne (quatre opérateurs au plus). La chaîne unique
    # `keyword_fallback_query` reste servie pour qui ne lit qu'elle : celle donnée, sinon
    # la plus partagée.
    fb_queries = {_source_label(k): str(v) for k, v in (keyword_fallback_queries or {}).items() if v}
    fb_sources = sorted({_source_label(s) for s in (keyword_fallback_sources or [])} | set(fb_queries))
    fb_query = str(keyword_fallback_query or "") or None
    if fb_query is None and fb_queries:
        from collections import Counter as _Counter
        fb_query = _Counter(fb_queries.values()).most_common(1)[0][0]
    return {
        "method": method,
        "computed_at": _dt.now(_tz.utc).isoformat(),
        "federation_incomplete": bool(federation_incomplete),
        "records_by_source": records,
        "records_identified": identified,
        # Les deux moitiés de l'identification, nommées. « identified » reste leur somme
        # pour que identifiés − doublons − retraits = passés au screening tienne.
        "records_identified_databases": from_databases,
        "records_identified_library": library,
        # Ce que la fédération a fait, source par source. `sources_searched` ne compte
        # que ok/empty/cached : un échec n'est pas une interrogation.
        "source_outcomes": outcomes,
        "sources_launched": len(outcomes),
        "sources_searched": searched,
        # La stratégie RÉELLEMENT soumise, quand elle n'a pas été la même pour toutes :
        # les sources en repli mots-clés, et les mots-clés qu'elles ont reçus.
        "keyword_fallback_sources": fb_sources,
        "keyword_fallback_query": fb_query,
        "keyword_fallback_queries": fb_queries,
        # POURQUOI une source a échoué. L'issue `error` était servie nue : la carte
        # disait « openalex : échec » et diagnostiquer demandait les journaux du serveur.
        "source_error_reasons": {_source_label(k): str(v)[:200]
                                 for k, v in (source_error_reasons or {}).items() if v},
        # Le TOTAL VRAI que chaque API a annoncé, et les sources dont on a gardé MOINS que
        # ce total. Au plafond par source, « pubmed 2 000 » se lisait comme un total : c'est
        # un plancher. Sur le scénario de contrôle de production, cinq sources étaient au
        # plafond et rien ne le disait ; et le lot gardé dépend de l'ordre de la source.
        # PubMed, OpenAlex et Europe PMC sont désormais interrogées par pertinence ; PubMed
        # triait par date et gardait les plus récents. PRISMA-S demande ce nombre.
        "source_totals": totals,
        "sources_capped": capped,
        "sources_ok": by_outcome["ok"],
        "sources_empty": by_outcome["empty"],
        "sources_cached": by_outcome["cached"],
        "sources_skipped": by_outcome["skipped"],
        "sources_failed": by_outcome["error"],
        "sources_cut_off": by_outcome["cut_by_budget"],
        "per_source_cap": (int(per_source_cap) if per_source_cap is not None else None),
        "search_mode": "exhaustive" if search_mode == "exhaustive" else "standard",
        "source_caps": caps,
        "title_abstract_sources": sorted({_source_label(s) for s in (title_abstract_sources or [])}),
        "unranked_sources": sorted({_source_label(s) for s in (unranked_sources or [])}),
        "split_queries": {_source_label(k): [str(q) for q in v]
                          for k, v in (split_queries or {}).items() if v},
        "year_slices": {_source_label(k): [str(y) for y in v]
                        for k, v in (year_slices or {}).items() if v},
        "year_slices_capped": {_source_label(k): {str(y): int(n) for y, n in (v or {}).items()}
                               for k, v in (year_slices_capped or {}).items() if v},
        "duplicate_records_across_sources": across,
        "duplicate_rows_in_database": rows,
        "duplicates_removed": duplicates,
        "unique_records": unique_after,
        "removed_no_abstract": no_abstract,
        "removed_not_matching": not_matching,
        "removed_other_reasons": other,
        "removed_before_screening": to_explain,
        "records_screened": corpus,
        # Passés au screening sans que la recherche les apparie, chacun sur sa ligne : le
        # corpus d'avant une fédération incomplète (et les sources qui ont manqué), puis
        # les liens qu'un relecteur a jugés. identifiés − doublons − retraits + gardés =
        # passés au screening.
        "kept_previous_links": kept_previous,
        "kept_previous_failed_sources": (sorted({_source_label(s) for s in (kept_previous_failed_sources or [])})
                                         if kept_previous else []),
        "kept_reviewed_links": kept_reviewed,
        "kept_added_links": kept_added,
        # Dont sans résumé : gardées, à juger sur le titre à la main (un sous-compte des
        # passés au screening, jamais un retrait).
        "kept_no_abstract": min(corpus, max(0, int(kept_no_abstract or 0))),
        "facets_skipped": [str(f) for f in (facets_skipped or []) if f],
        "local_match_capped": bool(local_match_capped),
        # L'exclusion des préprints, quand la recherche la demande : un critère
        # d'éligibilité appliqué par l'outil, à dire comme tel (PRISMA 2020).
        "preprints_excluded": bool(preprints_excluded),
        "removed_preprints": preprints,
        "preprints_excluded_at_source": sorted({_source_label(s) for s in (preprints_excluded_at_source or [])}),
        "preprint_sources_not_searched": sorted({_source_label(s) for s in (preprint_sources_not_searched or [])}),
    }


def _reconcile_prisma_identification(figures: dict, corpus_now: int) -> dict[str, Any]:
    """The identification figures of the last search, brought to the corpus AS IT STANDS.

    The figures are computed when a search closes. The corpus can change afterwards: a
    rebuild from the local database, duplicates marked later by the maintenance and -
    before the corpus was frozen at assembly - pages of a slow source arriving after
    the accounting. The panel used to show the live corpus as "records screened" next
    to duplicates and removals computed for another total, so identified − duplicates
    − removals no longer equalled screened (3,623 − 731 − 401 ≠ 3,602). The difference
    is now a line of its own, in the direction it happened:

      added_after_search - documents in the corpus that this search did not count
      removed_after_search - documents the search screened that have left the corpus

    and identified − duplicates − removals + kept + added − removed_after = screened
    holds for every scenario, whatever happened since the search ("kept": the links the
    search did not match but left in place, because its federation did not complete or
    because a reviewer had decided on them; counted at search time). The per-source
    counts and the search-time figures are kept as they were (they describe the search)."""
    identified = int(figures.get("records_identified") or 0)
    duplicates = int(figures.get("duplicates_removed") or 0)
    unique = int(figures.get("unique_records") or 0)
    no_abstract = int(figures.get("removed_no_abstract") or 0)
    preprints = int(figures.get("removed_preprints") or 0)
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
        "removed_preprints": preprints,
        "removed_not_matching": not_matching,
        "removed_other_reasons": other,
        "kept_previous_links": int(figures.get("kept_previous_links") or 0),
        "kept_reviewed_links": int(figures.get("kept_reviewed_links") or 0),
        "kept_added_links": int(figures.get("kept_added_links") or 0),
        "removed_before_screening": no_abstract + preprints + not_matching + other,
        "records_screened_at_search": screened_then,
        "added_after_search": max(0, corpus_now - screened_then),
        "removed_after_search": max(0, screened_then - corpus_now),
        "records_screened": corpus_now,
    })
    return out


def _store_prisma_identification(scenario_id: str, figures: dict) -> None:
    """Persiste les chiffres (colonne JSONB user_scenarios.prisma_identification).
    best-effort : une colonne absente ou une panne ne doit jamais faire échouer un
    populate - le PRISMA retombe alors sur le calcul historique (cf. get_user_scenario_prisma)."""
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
# (préfixe pmid/pmcid retiré, casse ignorée) › titre normalisé ≥ 20 › id. Deux DOI
# différents n'y sont donc jamais réunis ; seule la dédup d'un scénario fait adopter à
# une notice sans DOI le DOI unique de son titre. On PARTITIONNE toujours par
# project_context (un « literev » et un « gesica » de même titre ne sont PAS des doublons
# - cf. l'index unique partiel uq_litdoc_title_norm_doi sur (project_context,…)).
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

#: Mots que la syntaxe de PubMed laisse derrière elle (les crochets retirés, le tag
#: reste un mot) et liaisons sans valeur de recherche. « mh » et « tiab » cherchés comme
#: des termes, c'est ce que cinq sources sur douze ont reçu du scénario HPAI.
_TAG_WORDS = {
    "mh", "majr", "tiab", "ti", "ab", "tw", "dp", "pt", "la", "sb", "mesh", "terms",
    "title", "abstract", "noexp", "publication", "type", "all", "fields", "author",
    "journal", "subheading", "supplementary", "concept", "words",
}
_FILLER_WORDS = {"a", "an", "the", "of", "in", "on", "to", "and", "or", "not", "by",
                 "for", "with", "at", "as", "is", "are"}


def _terms_in_order(node) -> list[str]:
    """Les termes d'un sous-arbre, dans l'ordre, ceux sous un NOT exclus.

    Un terme nié est ce que la requête REFUSE : le repli mots-clés le cherchait, parce
    qu'il se contentait de retirer le mot « not » et de garder ses voisins. Une requête
    « ... NOT H1N1 » demandait donc du H1N1 aux sources en repli."""
    if not isinstance(node, tuple):
        return []
    if node[0] == "term":
        return [node[1]]
    if node[0] == "not":
        return []
    if node[0] in ("and", "or"):
        out: list[str] = []
        for child in node[1]:
            out.extend(_terms_in_order(child))
        return out
    return []


def _plain_keywords(query: str, max_words: int = 8) -> str:
    """Convertit une requête booléenne en mots-clés simples pour les API qui
    n'acceptent PAS la syntaxe booléenne (OpenAlex `search` renvoie 400, les
    serveurs de prépublications n'ont pas de recherche plein-texte).

    Les mots sont pris UN PAR CONJONCTION, à tour de rôle, et non dans l'ordre du texte.
    La version positionnelle gardait les `max_words` premiers mots, donc le PREMIER bloc
    de concept et lui seul : sur le scénario HPAI, une requête de 3 075 caractères
    structurée en « (exposition OU fomites OU transmission OU perception) ET (virus
    aviaires) SAUF H1N1 » devenait « environmental exposure mh tiab exposure,
    occupational diseases disease ». Pas un terme de grippe. L'ancre de la revue, le bloc
    ET qui dit de quelle maladie il s'agit, n'atteignait aucune des cinq sources en repli,
    qui ont rapporté 6 644 notices depuis arXiv et 5 794 depuis CORE pour une revue sur la
    grippe aviaire. Un tour de table garantit à chaque conjonction d'être représentée.

    PUR/testable : aucune entrée/sortie."""
    import re as _re
    # Les tags de champ D'ABORD : `_strip_field_tags` enlève `[mh]` en entier, là où un
    # simple retrait des crochets laissait `mh` dans les mots cherchés.
    plain = _strip_field_tags(query or "")

    groups: list[list[str]] = []
    parsed = False
    try:
        ast = _parse_boolean_ast(_tokenize_boolean(plain))
        if ast is not None:
            parsed = True
            if isinstance(ast, tuple) and ast[0] == "and":
                groups = [_terms_in_order(child) for child in ast[1]]
            else:
                groups = [_terms_in_order(ast)]
    except Exception:                                    # noqa: BLE001 - repli positionnel
        parsed = False
    groups = [g for g in groups if g]
    # L'arbre a été lu et ne contient AUCUN terme positif (« NOT "H5N1" ») : la réponse est
    # « rien ». Retomber sur le balayage positionnel rendrait justement le terme nié, qui
    # est l'ancien bug : il retirait le mot « not » et gardait ses voisins.
    if parsed and not groups:
        return ""

    def _add(word: str, into: list[str]) -> None:
        wl = word.strip().lower().strip(",;:.")
        if not wl or wl in _TAG_WORDS or wl in _FILLER_WORDS or wl in into:
            return
        into.append(wl)

    words: list[str] = []
    if groups:
        # Tour de table : le i-ème terme de chaque conjonction avant de passer au i+1-ème,
        # pour qu'aucun bloc ne soit muet même quand le budget est petit.
        for depth in range(max(len(g) for g in groups)):
            for g in groups:
                if depth >= len(g) or len(words) >= max_words:
                    continue
                # La graphie d'origine, ponctuation séparée sauf le point dans un mot
                # (« 2.3.4.4b » reste un mot-clé, non quatre).
                for w in _re.split(r"[\s\*\"()\[\]]+", _api_phrase(g[depth])):
                    if len(words) >= max_words:
                        break
                    _add(w, words)
            if len(words) >= max_words:
                break
    else:
        for w in _re.split(r"[\s\*\"()\[\]]+", _re.sub(r"[\[\]]", " ", plain)):
            if len(words) >= max_words:
                break
            _add(w, words)
    return " ".join(words[:max_words])


@app.get("/user-scenarios/{scenario_id}/search-strategy")
def get_search_strategy(scenario_id: str, lang: str | None = Query(None, max_length=8)) -> dict[str, Any]:
    """Returns the stored search_strategy JSON for this scenario.
    If not yet generated, generates it now and stores it (without a proposed correction:
    that is offered on the search screen, and the choice belongs to the user)."""
    row = _get_user_scenario_or_404(scenario_id)
    query = row["query"]
    strategy = row.get("search_strategy")
    # Régénérer si absent OU si la valeur stockée est un repli dégradé (p. ex.
    # généré pendant une panne de quota OpenAI → requête brute échoée). On ne
    # persiste QUE les stratégies valides, pour ne pas figer un cache empoisonné.
    if _strategy_is_degraded(strategy, query):
        strategy = _strategy_in_lang(_generate_search_strategy(query), lang)
        if not _strategy_is_degraded(strategy, query):
            try:
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE user_scenarios SET search_strategy = CAST(:strategy AS jsonb) WHERE id = :id
                    """), {"id": scenario_id, "strategy": json.dumps(_strategy_to_store(strategy))})
            except Exception as _e:
                logger.warning(f"get_search_strategy store error: {_e}")
    return strategy if isinstance(strategy, dict) else {}


def _source_plan_payload(query: str) -> dict[str, Any]:
    """The plan of a query, with the shortcuts a screen reads at a glance.

    One shape for both callers: the search page, which has a query and no scenario yet,
    and the scenario's strategy tab, which has a scenario."""
    plan = source_query_plan(query)
    return {
        "query_chars": len(query),
        "plan": plan,
        # What a reader wants at a glance: nothing to worry about, or which sources are
        # adapted. `reduced` is the only one that loses terms; `parts` and `body` do not.
        "whole": [r["source"] for r in plan if r["how"] == "whole"],
        "body": [r["source"] for r in plan if r["how"] == "body"],
        "parts": {r["source"]: r.get("parts") for r in plan if r["how"] == "parts"},
        "reduced": [r["source"] for r in plan if r["how"] == "reduced"],
        "keywords": [r["source"] for r in plan if r["how"] == "keywords"],
    }


@app.get("/user-scenarios/{scenario_id}/source-query-plan")
def get_source_query_plan(scenario_id: str) -> dict[str, Any]:
    """What each database will receive of this scenario's query, BEFORE a search runs.

    Until now a reviewer learned this afterwards, from the PRISMA notes, and only for the
    sources that were reduced or split: a source that refused the query outright simply
    appeared as failed. The plan is computed from the stored query with the same renderers
    and the same limits the search uses, so it costs a few milliseconds and no network."""
    row = _get_user_scenario_or_404(scenario_id)
    out = _source_plan_payload(row.get("query") or "")
    out["scenario_id"] = scenario_id
    return out


# ── Une stratégie collée : gardée telle quelle, vérifiée, une correction PROPOSÉE ──────
# Décision du propriétaire du produit : une stratégie de revue collée passe par le chemin
# du modèle quelle que soit sa longueur, elle est utilisée TELLE QUE SAISIE, et si elle a
# une erreur réelle, une correction est PROPOSÉE à l'utilisateur, qui l'accepte ou garde
# sa requête. Jamais de réécriture silencieuse. Le prompt disait à la fois « préserve le
# booléen » et « 2 à 4 groupes de concepts au plus, des requêtes pas trop longues » : une
# stratégie de 3 000 caractères en ressortait raccourcie, sans que rien ne le dise.

#: La plus longue requête que la traduction en stratégie accepte. Une stratégie de revue
#: systématique écrite pour PubMed fait couramment 3 000 à 8 000 caractères (la requête
#: HPAI de production en fait 3 075) : la borne de 2 000 la refusait d'un 422.
SEARCH_STRATEGY_MAX_CHARS = 8000
#: Le plafond de la réponse du modèle. Une correction proposée réécrit la stratégie ENTIÈRE
#: (8 000 caractères, de l'ordre de 2 500 tokens), avec son explication ; une traduction,
#: ses synonymes. 1 500 coupait une correction en son milieu (finish_reason « length »).
SEARCH_STRATEGY_MAX_TOKENS = 6000
#: Le plafond du second essai quand le modèle refuse le premier sans dire le sien : la plus
#: basse des limites de sortie des modèles qu'on peut lui avoir pointés.
_SEARCH_STRATEGY_FALLBACK_TOKENS = 4096
#: Les champs d'une correction proposée : servis à l'écran de recherche, jamais enregistrés
#: sur un scénario (le choix appartient à l'utilisateur, et il le fait une fois).
_PROPOSAL_KEYS = ("proposed_query", "proposal_reason", "proposal_reasons", "proposal_issues")

_STRATEGY_OPERATOR_RE = re.compile(r"\b(AND|OR|NOT)\b")
_STRATEGY_FIELD_TAG_RE = re.compile(r"\[[A-Za-z][A-Za-z /:\-]{0,40}\]")
_TYPOGRAPHIC_DOUBLE_QUOTES = "“”„‟«»"


def _is_boolean_strategy(query: str) -> bool:
    """Une stratégie booléenne ÉCRITE : un opérateur en majuscules ou un tag de champ.

    Plus strict que `_looks_boolean`, qui compte aussi des guillemets ou des parenthèses :
    « influenza (H5N1) in poultry workers » reste une phrase à traduire, et c'est alors le
    modèle qui en juge. Une stratégie ainsi reconnue est gardée telle que saisie, quoi que
    le modèle en dise. PUR."""
    q = query or ""
    return bool(_STRATEGY_OPERATOR_RE.search(q) or _STRATEGY_FIELD_TAG_RE.search(q))


def _balance_parentheses(query: str) -> str:
    """Les parenthèses de `query` équilibrées au plus prudent : une fermante sans ouvrante
    est retirée, une ouvrante jamais fermée est fermée à la fin. Les phrases entre
    guillemets et les tags de champ ne comptent pas, comme pour le tokeniseur. Aucun terme
    n'est retiré. PUR."""
    out: list[str] = []
    depth = 0
    in_quote = in_tag = False
    for ch in query or "":
        if in_quote:
            in_quote = ch != '"'
        elif in_tag:
            in_tag = ch != "]"
        elif ch == '"':
            in_quote = True
        elif ch == "[":
            in_tag = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                continue                                  # fermante orpheline : retirée
            depth -= 1
        out.append(ch)
    return "".join(out).rstrip() + ")" * depth


def _boolean_syntax_check(query: str) -> dict[str, Any]:
    """Ce qu'un booléen collé a de déséquilibré, dit SANS modèle, et la correction la plus
    prudente : {issues: [{kind, count}], fixed: str | None}.

    Le tokeniseur et le parseur de la recherche (`_tokenize_boolean`, `_parse_boolean_ast`)
    sont tolérants, et c'est ce qui rend l'erreur muette : un guillemet orphelin colle au
    mot voisin, une parenthèse fermante en trop ARRÊTE la lecture et le reste de la requête
    est ignoré (« (A OR B)) AND C » se lit « A OR B »), une ouvrante jamais fermée garde
    tout ce qui suit dans son groupe, et des guillemets typographiques (“ ” « », ceux d'un
    traitement de texte) ne font pas une phrase. Les parenthèses sont lues par CE
    tokeniseur : le contrôle voit ce que la recherche voit, même quand le modèle passe à
    côté. PUR."""
    t = query or ""
    issues: list[dict[str, Any]] = []
    n_typo = sum(t.count(c) for c in _TYPOGRAPHIC_DOUBLE_QUOTES)
    if n_typo:
        t = re.sub(r"«\s*", '"', t)
        t = re.sub(r"\s*»", '"', t)
        t = re.sub("[“”„‟]", '"', t)
        issues.append({"kind": "typographic_quotes", "count": n_typo})
    if t.count('"') % 2:
        # Le tokeniseur apparie les guillemets de gauche à droite : le dernier est
        # l'orphelin, et la recherche le lit comme s'il n'était pas là.
        _i = t.rfind('"')
        t = t[:_i] + t[_i + 1:]
        issues.append({"kind": "unpaired_quote", "count": 1})
    depth = closing = 0
    for kind, _v in _tokenize_boolean(t):
        if kind == "(":
            depth += 1
        elif kind == ")":
            if depth:
                depth -= 1
            else:
                closing += 1
    if closing:
        issues.append({"kind": "unmatched_closing_parenthesis", "count": closing})
    if depth:
        issues.append({"kind": "unclosed_parenthesis", "count": depth})
    if closing or depth:
        t = _balance_parentheses(t)
    fixed = t if issues else None
    # Une correction qui ne laisse rien à chercher n'en est pas une.
    if fixed is not None and _parse_boolean_ast(_tokenize_boolean(fixed)) is None:
        fixed = None
    return {"issues": issues, "fixed": fixed}


#: Ce que dit le contrôle sans modèle, dans les deux langues de l'interface.
_SYNTAX_REASONS: dict[str, dict[str, str]] = {
    "typographic_quotes": {
        "en": "Typographic quotes (“ ” « ») do not make a phrase for the search: the proposal "
              "uses straight quotes.",
        "fr": "Les guillemets typographiques (“ ” « ») ne font pas une phrase pour la recherche : "
              "la proposition met des guillemets droits.",
    },
    "unpaired_quote": {
        "en": "A double quote has no partner: the proposal removes the last one, which the "
              "search ignores anyway.",
        "fr": "Un guillemet double n'a pas son pendant : la proposition retire le dernier, que la "
              "recherche ignore de toute façon.",
    },
    "unmatched_closing_parenthesis": {
        "en": "Closing parentheses without an opening one: {n}. The search stops reading at the "
              "first one and ignores what follows; the proposal removes them.",
        "fr": "Parenthèses fermantes sans ouvrante : {n}. La recherche s'arrête de lire à la "
              "première et ignore la suite ; la proposition les retire.",
    },
    "unclosed_parenthesis": {
        "en": "Opening parentheses never closed: {n}. The proposal closes them at the end of "
              "the query.",
        "fr": "Parenthèses ouvrantes jamais fermées : {n}. La proposition les ferme à la fin de "
              "la requête.",
    },
}
_GENERIC_PROPOSAL_REASON = {
    "en": "The query seems to contain an error: a correction is proposed.",
    "fr": "La requête semble contenir une erreur : une correction est proposée.",
}


def _syntax_reasons(issues: list[dict[str, Any]]) -> dict[str, str]:
    return {lang: " ".join(_SYNTAX_REASONS[i["kind"]][lang].replace("{n}", str(i.get("count", 1)))
                           for i in issues if i.get("kind") in _SYNTAX_REASONS)
            for lang in ("en", "fr")}


def _strategy_terms(query: str) -> int:
    return sum(1 for kind, _v in _tokenize_boolean(query or "") if kind == "TERM")


def _strategy_proposal(entered: str, model_query: object = None, model_reason: object = None) -> dict[str, Any]:
    """La correction PROPOSÉE d'une requête utilisée telle que saisie, ou {} : celle du
    modèle quand elle tient, sinon celle du contrôle sans modèle.

    Celle du modèle est écartée quand elle ne change rien (aux espaces près) ou quand elle
    perd plus d'un quart des termes d'une longue stratégie : une « correction » qui
    raccourcit une stratégie de revue est la simplification que ce chemin ne fait plus, et
    elle passerait inaperçue au milieu de 8 000 caractères. Une correction du modèle qui
    laisse des parenthèses ou des guillemets déséquilibrés est complétée par celle du
    contrôle. Rien n'est appliqué ici : l'utilisateur accepte, ou garde sa requête."""
    def _same(a: str, b: str) -> bool:
        return " ".join(a.split()) == " ".join(b.split())

    def _reasons(raw: object) -> dict[str, str]:
        if isinstance(raw, dict):
            got = {k: str(raw.get(k) or "").strip() for k in ("en", "fr")}
        else:
            got = {k: str(raw or "").strip() for k in ("en", "fr")}
        return {k: v for k, v in got.items() if v}

    entered = entered or ""
    check = _boolean_syntax_check(entered)
    issues = [i["kind"] for i in check["issues"]]
    proposal = model_query.strip() if isinstance(model_query, str) else ""
    if proposal and _same(proposal, entered):
        proposal = ""
    if proposal:
        n0, n1 = _strategy_terms(entered), _strategy_terms(proposal)
        if n0 >= 12 and n1 < 0.75 * n0:
            logger.warning(f"_strategy_proposal: correction du modèle écartée, elle perd des termes "
                           f"({n1} contre {n0} dans la stratégie saisie)")
            proposal = ""
    reasons: dict[str, str] = {}
    if proposal:
        reasons = _reasons(model_reason)
        rest = _boolean_syntax_check(proposal)
        if rest["fixed"]:
            proposal = rest["fixed"]
            extra = _syntax_reasons(rest["issues"])
            reasons = {k: " ".join(x for x in (reasons.get(k, ""), extra[k]) if x) for k in ("en", "fr")}
        issues = issues + ["model"]
    elif check["fixed"]:
        proposal = check["fixed"]
        reasons = _syntax_reasons(check["issues"])
    if not proposal or _same(proposal, entered):
        return {}
    reasons = {k: (reasons.get(k) or reasons.get("en" if k == "fr" else "fr") or _GENERIC_PROPOSAL_REASON[k])
               for k in ("en", "fr")}
    return {"proposed_query": proposal, "proposal_reasons": reasons, "proposal_issues": issues}


def _finish_strategy(query: str, result: object, strict: bool) -> dict[str, Any]:
    """La stratégie servie. Une stratégie booléenne saisie est gardée TELLE QUELLE : `general`
    et `pubmed` sont la requête saisie, au caractère près, quelle que soit sa longueur ; une
    phrase est traduite. Une requête utilisée telle que saisie est vérifiée, et sa correction
    éventuelle est PROPOSÉE (`proposed_query`, sa raison dans les deux langues) sans être
    appliquée. Calculé à chaque appel, jamais mis en cache : le cache garde la réponse du
    modèle. PUR (hors journal)."""
    out = dict(result) if isinstance(result, dict) else {}
    raw_query, raw_reason = out.get("proposed_query"), out.get("proposal_reason")
    for k in _PROPOSAL_KEYS:
        out.pop(k, None)
    kept = strict or str(out.get("input_kind") or "").strip().lower() == "boolean"
    if kept:
        out["general"] = query
        out["pubmed"] = query
    out["input_kind"] = "boolean" if kept else "natural"
    out["entered_query"] = query
    # Une correction n'a de sens que pour une requête utilisée telle que saisie : une phrase
    # traduite n'est pas lue, sa traduction l'est. Un repli dégradé utilise la saisie, lui.
    as_entered = (kept or bool(out.get("degraded"))
                  or str(out.get("general") or "").strip() == (query or "").strip())
    proposal = _strategy_proposal(query, raw_query if kept else None, raw_reason) if as_entered else {}
    out["proposed_query"] = proposal.get("proposed_query")
    out["proposal_reasons"] = proposal.get("proposal_reasons")
    out["proposal_issues"] = proposal.get("proposal_issues") or []
    out["proposal_reason"] = (proposal.get("proposal_reasons") or {}).get("fr")
    return out


def _strategy_in_lang(strategy: object, lang: str | None) -> dict[str, Any]:
    """La raison de la correction proposée dans la langue de l'interface (français par
    défaut, comme le reste de l'application). PUR."""
    out = dict(strategy) if isinstance(strategy, dict) else {}
    reasons = out.get("proposal_reasons") if isinstance(out.get("proposal_reasons"), dict) else {}
    if out.get("proposed_query") and reasons:
        lang_ = _norm_lang(lang) or "fr"
        out["proposal_reason"] = reasons.get(lang_) or reasons.get("fr") or reasons.get("en")
    return out


def _strategy_to_store(strategy: object) -> dict[str, Any]:
    """La stratégie telle qu'un scénario l'enregistre : sans correction proposée en
    attente, qui ne vaut que pour l'écran de recherche. PUR."""
    out = dict(strategy) if isinstance(strategy, dict) else {}
    for k in _PROPOSAL_KEYS:
        out.pop(k, None)
    return out


_RE_CEILING_REFUSED = re.compile(r"\b(max_tokens|max_completion_tokens)\b.{0,80}?"
                                 r"(too large|at most|maximum|exceed)", re.I | re.S)
_RE_CEILING_AT_MOST = re.compile(r"(?:at most|maximum(?: value)?(?: is| of)?)\s*(\d{3,6})", re.I)


def _ceiling_refused(exc: BaseException) -> int | None:
    """Le plafond à redemander quand le modèle refuse celui de la requête, ou None si
    l'erreur est autre. L'API répond « max_tokens is too large: 6000. This model supports at
    most 4096 completion tokens » : refusée avant toute lecture, la requête ne coûte rien,
    et `llm_usage` ne répare que les paramètres refusés, pas leurs valeurs. PUR."""
    msg = str(exc or "")
    if not _RE_CEILING_REFUSED.search(msg):
        return None
    m = _RE_CEILING_AT_MOST.search(msg)
    n = int(m.group(1)) if m else _SEARCH_STRATEGY_FALLBACK_TOKENS
    return max(256, min(n, SEARCH_STRATEGY_MAX_TOKENS - 1))


_STRATEGY_SYSTEM_PROMPT = (
    "You are a systematic review librarian. The user gives EITHER a natural-language research "
    "question OR a Boolean search strategy they wrote (AND/OR/NOT operators, quoted phrases, "
    "parentheses, field tags such as [tiab] or [Mesh]). Decide which it is, and set "
    "\"input_kind\" to \"boolean\" or \"natural\".\n"
    "If it is a BOOLEAN strategy:\n"
    "- It is used exactly as the user wrote it, however long. Do NOT rewrite, shorten, "
    "simplify, reorder or translate it, and do not repeat it: leave \"general\" and \"pubmed\" "
    "empty, and say in \"explanation\" that the strategy was kept as entered.\n"
    "- CHECK it for real errors only: unbalanced parentheses or double quotes; operators that "
    "make no sense (an operator doubled, an operator at the start or the end of a group, NOT "
    "with nothing after it); misspelled or lowercase operators clearly meant as operators "
    "(ANDD, OT, \"and\" between two parenthesized groups); a group that cannot match anything "
    "(an empty group, X AND NOT X); a misused PubMed field tag (a tag that does not exist, "
    "such as [tiabs], or a tag placed on an operator). Style, synonyms, missing concepts and "
    "breadth are NOT errors.\n"
    "- If you find at least one real error, put in \"proposed_query\" the WHOLE strategy with "
    "only those errors corrected and every other character unchanged, and in "
    "\"proposal_reason\" one or two sentences saying what is wrong and what you changed, in "
    "English under \"en\" and in French under \"fr\". Otherwise leave both empty.\n"
    "If it is NATURAL language: translate it into a Boolean query. \"general\" uses AND/OR/NOT "
    "and quotes for phrases; \"pubmed\" is the PubMed version with MeSH terms [MeSH Terms] and "
    "field tags [Title/Abstract]. Keep a translation practical: 2 to 4 concept groups. Leave "
    "\"proposed_query\" and \"proposal_reason\" empty.\n"
    "Return ONLY valid JSON with these fields:\n"
    '{"input_kind": "boolean or natural",\n'
    '"general": "the Boolean query, empty for a Boolean input",\n'
    '"pubmed": "the PubMed query, empty for a Boolean input",\n'
    '"explanation": "1-2 sentences on the term choices, or that the strategy was kept as entered",\n'
    '"synonyms": [["term1", "synonym1a", "synonym1b"], ["term2", "synonym2a"]],\n'
    '"proposed_query": "the corrected strategy, or an empty string",\n'
    '"proposal_reason": {"en": "1-2 sentences, or empty", "fr": "1-2 phrases, ou vide"}}\n'
    "Never write the em dash character; use a comma, a colon or a hyphen."
)


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
    sur la même stratégie booléenne - condition d'un résultat déterministe."""
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def _strategy_cache_key(query: str, strict: bool) -> str:
    """La clé de cache d'une requête. Une phrase : sa forme normalisée (`_strategy_key`).
    Une stratégie écrite : son texte EXACT, espaces compactés, la casse d'un opérateur y
    ayant un sens (« and » n'est pas « AND » pour PubMed) et sa vérification ne valant que
    pour elle. Au-delà de 2 000 caractères, l'empreinte du texte : `query_key` est la clé
    primaire d'un index btree, dont une entrée ne passe pas ~2 700 octets, et une stratégie
    de 8 000 caractères n'aurait jamais été enregistrée."""
    key = ("b:" + re.sub(r"\s+", " ", (query or "").strip())) if strict else _strategy_key(query)
    if len(key) > 2000:
        import hashlib
        key = f"{'b' if strict else 'n'}:sha256:{hashlib.sha256(key.encode('utf-8')).hexdigest()}"
    return key


def _generate_search_strategy(query: str) -> dict:
    """La stratégie booléenne d'une requête, par le modèle `bulk`. Une phrase est traduite
    (`general` : booléen ; `pubmed` : MeSH et tags de champ ; `explanation`, `synonyms`).
    Une stratégie booléenne saisie est gardée TELLE QUELLE, quelle que soit sa longueur, et
    vérifiée : une erreur réelle donne une correction PROPOSÉE (`proposed_query`,
    `proposal_reasons`), jamais appliquée (cf. `_finish_strategy`). `entered_query` porte la
    requête saisie, `input_kind` ce qu'elle était.

    Résultats déterministes mis en cache (mémoire puis base) ; renvoie une COPIE."""
    _strict = _is_boolean_strategy(query)
    _cache_key = _strategy_cache_key(query, _strict)
    _cached = _STRATEGY_CACHE.get(_cache_key)
    if _cached is not None:
        return _finish_strategy(query, _cached, _strict)
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
            return _finish_strategy(query, _srow["strategy"], _strict)
    except Exception as _ce:
        logger.warning(f"_generate_search_strategy DB cache read: {_ce}")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return _finish_strategy(query, {"general": query, "pubmed": query, "explanation": "",
                                        "synonyms": [], "degraded": True}, _strict)
    try:
        from llm_usage import LLMTruncated as _LLMTruncated, MeteredOpenAI as _OAI_ss, json_content as _json_content
        _client = _OAI_ss(api_key=openai_key)
        _messages = [{"role": "system", "content": _STRATEGY_SYSTEM_PROMPT},
                     {"role": "user", "content": f"Research query: {query}"}]

        def _ask(ceiling: int):
            return _client.chat.completions.create(
                model=_model("bulk"),
                messages=_messages,
                temperature=0,
                seed=42,
                # Une correction réécrit la stratégie entière : cf. SEARCH_STRATEGY_MAX_TOKENS.
                # Une coupure qui reste est nommée (`_json_content`), jamais servie tronquée.
                max_tokens=ceiling,
                response_format={"type": "json_object"},
            )
        try:
            response = _ask(SEARCH_STRATEGY_MAX_TOKENS)
        except Exception as _ce:
            # Un modèle dont la limite de sortie est plus basse refuse le plafond : un second
            # essai à la sienne, plutôt qu'une recherche qui part en mode dégradé.
            _lower = _ceiling_refused(_ce)
            if _lower is None:
                raise
            logger.warning(f"_generate_search_strategy: plafond de {SEARCH_STRATEGY_MAX_TOKENS} tokens "
                           f"refusé ({str(_ce)[:160]}) ; nouvel essai à {_lower}")
            response = _ask(_lower)
        try:
            _content = _json_content(response, label="search strategy")
        except _LLMTruncated as _te:
            # Une réponse coupée n'est pas une réponse plus courte : c'est l'absence de
            # réponse, et elle se dit comme telle (jamais mise en cache, réessayée).
            logger.warning(f"_generate_search_strategy: {_te}")
            return _finish_strategy(query, {
                "general": query, "pubmed": query, "explanation": "", "synonyms": [],
                "degraded": True, "truncated": True, "degraded_reason": "truncated",
                "reason": ("La stratégie booléenne rédigée par le modèle a été coupée au plafond "
                           "de tokens : la recherche utilise la requête telle qu'elle a été saisie.")},
                _strict)
        _result = json.loads(_content)
        # Une stratégie booléenne saisie est la requête, au caractère près : le modèle ne la
        # recopie pas (il raccourcissait une stratégie longue), le code la remet.
        if isinstance(_result, dict) and (_strict or str(_result.get("input_kind") or "").strip().lower() == "boolean"):
            _result["general"] = query
            _result["pubmed"] = query
        # Validité au sens du RESTE du code (_strategy_is_degraded : `general` vide, ou
        # sans opérateur booléen), pas seulement l'absence du drapeau `degraded`. Une
        # traduction qui « réussit » en renvoyant une requête inexploitable était sinon
        # épinglée à vie dans le cache persistant (ON CONFLICT DO NOTHING, jamais
        # réécrit), et le réessai promis « quand OpenAI redevient disponible » n'avait
        # jamais lieu pour cette requête.
        if isinstance(_result, dict) and not _result.get("degraded") and not _strategy_is_degraded(_result):
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
        return _finish_strategy(query, _result, _strict)
    except Exception as _e:
        logger.warning(f"_generate_search_strategy failed: {_e}")
        return _finish_strategy(query, {"general": query, "pubmed": query, "explanation": "",
                                        "synonyms": [], "degraded": True}, _strict)


class SearchStrategyIn(BaseModel):
    # BORNÉE. Ce texte devient le prompt d'un appel au modèle ET la clé d'une ligne de
    # cache persistée : sans longueur maximale, une chaîne arbitrairement longue
    # coûtait un appel et laissait une ligne, autant de fois qu'on la faisait varier.
    # 8 000 et non 2 000 : la longueur d'une stratégie de revue collée telle quelle.
    query: str = Field(..., min_length=1, max_length=SEARCH_STRATEGY_MAX_CHARS)
    # La langue de l'interface, celle de l'explication d'une correction proposée.
    lang: str | None = Field(default=None, max_length=8)


@app.post("/search-strategy")
def post_search_strategy(payload: SearchStrategyIn) -> dict[str, Any]:
    """Traduit une requête en langage naturel en stratégie booléenne (LLM), ou garde telle
    quelle une stratégie booléenne saisie, en la vérifiant.

    Permet à la recherche d'AFFICHER la requête booléenne et de l'utiliser comme
    base du corpus : la même stratégie est ensuite persistée sur le scénario, de
    sorte que le compteur de recherche == la taille du corpus (même requête).
    Une stratégie saisie qui a une erreur réelle revient avec une correction PROPOSÉE
    (`proposed_query`, `proposal_reason` dans la langue `lang`) : `general` reste la
    requête saisie, et c'est l'utilisateur qui accepte la correction ou garde sa requête.
    """
    return _strategy_in_lang(_generate_search_strategy(payload.query), payload.lang)


class SourcePlanIn(BaseModel):
    # Même plafond que la stratégie elle-même : on veut pouvoir planifier la requête
    # la plus longue qu'un utilisateur puisse coller, pas une version coupée.
    query: str = Field(..., min_length=1, max_length=SEARCH_STRATEGY_MAX_CHARS)


@app.post("/source-query-plan")
def post_source_query_plan(payload: SourcePlanIn) -> dict[str, Any]:
    """Ce que chaque base recevra d'une requête, AVANT qu'un scénario existe.

    La page de recherche affiche la requête booléenne compilée puis lance la recherche :
    c'est là, et non après coup dans les notes PRISMA, qu'il faut dire qu'OpenAlex la
    recevra en neuf morceaux et que DOAJ n'en prendra qu'une forme réduite. Fonction pure,
    aucun appel réseau, aucun modèle."""
    return _source_plan_payload(payload.query)


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
    NB : ces comptes portent sur la BIBLIOTHÈQUE INDEXÉE - la recherche en direct
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
        _found = _search_local_doc_ids(boolean, "boolean", filters, limit=LOCAL_MATCH_LIMIT)
        ids = set(_found)
        id_sets.append(ids)
        facets.append({"kind": sq["kind"], "text": sq["text"],
                       "boolean": boolean, "count": len(ids), "op": sq.get("op"),
                       # La borne atteinte : le compte est un plancher, pas un total.
                       "capped": bool(getattr(_found, "capped", False))})
    if id_sets:
        union = len(set().union(*id_sets))
        intersection = len(set.intersection(*id_sets))
        # `combined` reflète le fold gauche→droite avec l'op PAR facette (les totaux
        # union/intersection restent affichés comme repères).
        combined = len(_fold_facet_sets(id_sets, clean, payload.combinator))
    else:
        union = intersection = combined = 0
    return {"facets": facets, "union": union, "intersection": intersection,
            "combined": combined, "combinator": payload.combinator,
            "capped": any(f["capped"] for f in facets)}


# ─── What each source will actually receive, before anything is sent ─────────
#: One row per source: its name, how its query travels, and the limit it is measured
#: against. The pipeline tailors the query to nine APIs, and until now the only way to
#: learn what each one got was to run the search and read the PRISMA notes afterwards. A
#: reviewer writing a strategy deserves to know beforehand which sources will take it
#: whole, which will receive it reduced, and which would refuse it.
#:
#: This mirrors the decisions `_run_populate_job` makes; the constants are the SAME objects,
#: so a limit cannot drift between the plan and the search, and a test pins the pairing.
SOURCE_QUERY_CARRIAGE = (
    ("pubmed", "body", None),                 # eutils takes a long query in a POST body
    ("europepmc", "body_above", None),        # EPMC_POST_ABOVE_ENCODED, read at call time
    ("core", "body", None),                   # its query goes in a POST JSON body
    ("openalex", "url", "URL_QUERY_MAX_ENCODED"),
    ("clinicaltrials", "url", "URL_QUERY_MAX_ENCODED"),
    ("semantic_scholar", "url", "S2_MAX_QUERY_ENCODED"),
    ("doaj", "url", "DOAJ_MAX_QUERY_CHARS"),
    ("openaire", "operators", "OPENAIRE_MAX_OPERATORS"),
    ("arxiv", "url", "URL_QUERY_MAX_ENCODED"),
    ("crossref", "keywords", None),           # free text only, never a boolean
)


def source_query_plan(boolean: str) -> list[dict]:
    """For each source, what it will receive of `boolean`, and why.

    `how` is one of: `whole` (the query as written), `body` (whole, in the request body, so
    no length applies), `parts` (split into queries whose union is the query), `reduced`
    (the same tree with each OR block cut to its first terms, exclusions dropped) or
    `keywords` (free text, this source takes no boolean). PURE: no network, no database."""
    from .sources import EPMC_POST_ABOVE_ENCODED, epmc_query
    out: list[dict] = []
    portable = " ".join(_strip_field_tags(boolean or "").split())
    real = bool(portable) and _looks_boolean(portable)
    clean = _clean_boolean(portable) if real else ""
    for name, carriage, limit_name in SOURCE_QUERY_CARRIAGE:
        row: dict = {"source": name, "limit": limit_name, "carriage": carriage}
        if not real:
            row.update(how="keywords", note="no boolean query: every source gets keywords")
            out.append(row)
            continue
        if carriage == "keywords":
            row.update(how="keywords", note="this API reads free text, never a boolean")
        elif carriage == "body":
            row.update(how="body", encoded=_url_encoded_len(clean or portable))
        elif carriage == "body_above":
            enc = _url_encoded_len(epmc_query(boolean))
            row.update(how="body" if enc > EPMC_POST_ABOVE_ENCODED else "whole",
                       encoded=enc, limit_value=EPMC_POST_ABOVE_ENCODED)
        elif carriage == "operators":
            n = _count_operators(clean)
            cap = OPENAIRE_MAX_OPERATORS
            parts = _split_boolean(portable, 10 ** 6, max_operators=cap) if n > cap else []
            row.update(how="parts" if parts else ("whole" if n <= cap else "reduced"),
                       operators=n, limit_value=cap, parts=len(parts) or None)
        else:                                   # url
            if name == "semantic_scholar":
                rendered, cap = (_boolean_to_s2(_parse_boolean_ast(_tokenize_boolean(portable)))
                                 or ""), S2_MAX_QUERY_ENCODED
            elif name == "arxiv":
                rendered, cap = (_boolean_to_arxiv(_parse_boolean_ast(_tokenize_boolean(portable)))
                                 or ""), URL_QUERY_MAX_ENCODED
            elif name == "doaj":
                rendered, cap = clean, DOAJ_MAX_QUERY_CHARS
            else:
                rendered, cap = clean, URL_QUERY_MAX_ENCODED
            measure = len if name == "doaj" else _url_encoded_len
            if not rendered:
                row.update(how="keywords", note="this API cannot express the query (a NOT it "
                                                "has no syntax for)")
            elif measure(rendered) <= cap:
                row.update(how="whole", encoded=measure(rendered), limit_value=cap)
            else:
                parts = _split_boolean(portable, cap, measure=measure)
                row.update(how="parts" if parts else "reduced", encoded=measure(rendered),
                           limit_value=cap, parts=len(parts) or None)
        out.append(row)
    return out
