"""OpenAIRE refusait toute stratégie booléenne : « Too many logical operators found. Max
allowed is 4 ».

Mesuré en production avec la sonde /sources/health, après #327 : 48 opérateurs sur la
requête recommandée, 52 sur la requête HPAI réduite, 95 sur la requête HPAI entière, 400
à chaque fois. Depuis la migration vers l'API Graph v2 (2026-05-31), OpenAIRE ne
répondait donc qu'aux requêtes sans booléen, et la carte disait « en échec » sans que
personne ne sache pourquoi.

Elle reçoit désormais sa propre réduction, bornée à quatre opérateurs : le ET entre les
blocs, et un ou deux termes par bloc, les plus courts d'abord. Ses notices sont
ré-appariées en local contre le booléen entier, et la carte dit ce qu'elle a reçu, à côté
de ce que les autres sources en repli ont reçu.
"""
import ast
import inspect
import pathlib
import textwrap

import pytest

pytest.importorskip("fastapi")

from api.search import (  # noqa: E402
    OPENAIRE_MAX_OPERATORS,
    _count_operators,
    _prisma_identification_figures,
    _shorten_boolean,
    _strip_field_tags,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent

HPAI = (
    '( ( ("Environmental Exposure"[mh] OR "Environmental Exposure*"[tiab] '
    'OR "Exposure, Environmental"[tiab]) OR ("Occupational Exposure"[mh] '
    'OR "Occupational Diseases"[mh] OR "Occupational Disease*"[tiab]) ) '
    'OR ( "Fomites"[mh] OR "Fomite*"[tiab] ) '
    'OR ( "Disease Transmission, Infectious"[mh] OR "Pathogen Transmission"[tiab] ) ) '
    'AND ( "Influenza in Birds"[mh] OR "Influenza A Virus, H5N1 Subtype"[mh] '
    'OR "avian influenza"[tiab] OR "H5N1"[tiab] OR "H7N9"[tiab] ) '
    'NOT ( "Influenza A Virus, H1N1 Subtype"[mh] OR "H1N1 Influenza Virus*"[tiab] )'
)


def test_the_limit_is_the_one_openaire_states():
    assert OPENAIRE_MAX_OPERATORS == 4


def test_the_query_for_openaire_holds_four_operators_and_every_concept_block():
    out = _shorten_boolean(_strip_field_tags(HPAI), 1200, max_operators=OPENAIRE_MAX_OPERATORS)
    assert out, "rien ne tient : OpenAIRE retomberait sur les mots-clés"
    assert _count_operators(out) <= OPENAIRE_MAX_OPERATORS, out
    assert " AND " in out, out
    low = out.lower()
    assert any(w in low for w in ("h5n1", "h7n9", "avian influenza", "influenza")), out
    assert any(w in low for w in ("fomite", "exposure", "occupational", "transmission")), out
    assert "h1n1" not in low and "NOT" not in out, out


def test_two_blocks_get_two_terms_each_and_three_blocks_one():
    two = '("a" OR "b" OR "c") AND ("d" OR "e" OR "f")'
    assert _shorten_boolean(two, 1200, max_operators=4) == "(a OR b) AND (d OR e)"
    three = '("a" OR "b") AND ("c" OR "d") AND ("e" OR "f")'
    assert _shorten_boolean(three, 1200, max_operators=4) == "a AND c AND e"
    assert _shorten_boolean(three, 1200, max_operators=1) == "", (
        "trois blocs ET font déjà deux opérateurs : rien ne tient sous un seul")


def test_the_shortest_terms_of_each_block_are_the_ones_kept():
    """Sous quatre opérateurs, « h5n1 » et « avian influenza » valent mieux que la vedette
    MeSH « influenza a virus h5n1 subtype », que personne n'écrit dans un résumé."""
    q = ('("Influenza A Virus, H5N1 Subtype"[mh] OR "avian influenza"[tiab] OR "H5N1"[tiab]) '
         'AND ("Occupational Exposure"[mh] OR "farmers"[tiab])')
    # Quatre opérateurs : tout tient, les plus courts en tête de chaque bloc.
    assert _shorten_boolean(_strip_field_tags(q), 1200, max_operators=4) == (
        '(h5n1 OR "avian influenza" OR "influenza a virus h5n1 subtype") AND (farmers OR "occupational exposure")')
    # Trois : la vedette MeSH est la première sacrifiée.
    assert _shorten_boolean(_strip_field_tags(q), 1200, max_operators=3) == (
        '(h5n1 OR "avian influenza") AND (farmers OR "occupational exposure")')


def test_counting_operators_ignores_the_words_inside_terms():
    assert _count_operators('"mortality and or morbidity" AND h5n1') == 1
    assert _count_operators("(a OR b) AND (c OR d)") == 3
    assert _count_operators("h5n1") == 0


def test_the_populate_gives_openaire_its_own_query_and_never_trusts_it_as_native():
    from api import pipeline as P
    tree = ast.parse(textwrap.dedent(inspect.getsource(P._run_user_scenario_populate)))
    assigned: dict[str, list[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for tgt in n.targets:
                elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
                for i, e in enumerate(elts):
                    if isinstance(e, ast.Name):
                        value = n.value.elts[i] if (isinstance(tgt, ast.Tuple)
                                                    and isinstance(n.value, ast.Tuple)) else n.value
                        assigned.setdefault(e.id, []).append(ast.unparse(value))
    assert any("_shorten_boolean(" in v and "max_operators=OPENAIRE_MAX_OPERATORS" in v
               for v in assigned.get("_openaire_q", [])), assigned.get("_openaire_q")
    assert any(v == "_openaire_q" for v in assigned.get("_oa_q", [])), assigned.get("_oa_q")
    assert all(v == "False" for v in assigned.get("_oa_native", [])), (
        "une requête réduite n'est jamais « native » : ses notices doivent être ré-appariées")
    assert any("openaire" in v for v in assigned.get("_kw_fallback", [])), (
        "OpenAIRE reçoit une requête réduite : la carte doit la nommer")
    # Et ce qu'elle a reçu est écrit dans le dictionnaire par source que la carte lit.
    subs = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Subscript)
            and ast.unparse(n.targets[0]).startswith("_fallback_queries[")]
    assert any("'openaire'" in s and "_openaire_q" in s for s in subs), subs


def test_the_figures_carry_each_sources_query():
    f = _prisma_identification_figures(
        {"openaire": 12, "pubmed": 30}, 42, 0, 42,
        source_outcomes={"_fetch_openaire": "ok", "_fetch_pubmed": "ok"},
        keyword_fallback_sources=["openaire"],
        keyword_fallback_queries={"_fetch_openaire": '"occupational exposure" AND h5n1'})
    assert f["keyword_fallback_sources"] == ["openaire"]
    assert f["keyword_fallback_queries"] == {"openaire": '"occupational exposure" AND h5n1'}
    # La chaîne unique reste servie : celle que le plus de sources ont reçue.
    assert f["keyword_fallback_query"] == '"occupational exposure" AND h5n1'
    g = _prisma_identification_figures(
        {"pubmed": 30}, 30, 0, 30, source_outcomes={"_fetch_pubmed": "ok"},
        keyword_fallback_queries={"openalex": "R", "doaj": "R", "core": "R", "arxiv": "all:R", "openaire": "Q4"})
    assert g["keyword_fallback_sources"] == ["arxiv", "core", "doaj", "openaire", "openalex"]
    assert g["keyword_fallback_query"] == "R"


def test_the_card_groups_the_sources_by_the_query_they_received():
    page = (ROOT / "frontend" / "src" / "components" / "ScenarioDetailPage.tsx").read_text(encoding="utf-8")
    assert "keyword_fallback_queries" in page
    for loc in ("fr", "en"):
        txt = (ROOT / "frontend" / "src" / "i18n" / "locales" / f"{loc}.ts").read_text(encoding="utf-8")
        line = next(l for l in txt.splitlines() if "keywordFallback:" in l)
        assert "{sources}" in line and "{keywords}" in line, line
        assert ("opérateurs" in line) if loc == "fr" else ("operator" in line), (
            f"{loc}: la phrase ne doit plus attribuer le repli à la seule limite d'URL")
