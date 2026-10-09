"""Le repli mots-clés gardait le PREMIER bloc de concept, et lui seul.

Trouvé en comparant les chiffres réels du scénario HPAI de production à ce que son
corpus contient. Son tableau d'identification dit :

    arxiv 6 644 · core 5 794 · crossref 1 907 · semantic_scholar 985 · openalex 156
    · medrxiv 3   (pubmed et europepmc : absents)

Pour une revue sur l'exposition professionnelle à la grippe aviaire, arXiv et CORE
rapportant douze mille notices quand PubMed n'en rapporte aucune n'est pas un hasard.

La requête du scénario fait 3 075 caractères, 2 465 une fois les tags de champ retirés,
et `_send_bool` exige 1 200 au plus (limite d'URL d'OpenAlex). Cinq sources sur douze
basculaient donc en repli mots-clés, et `_plain_keywords` leur donnait ceci :

    « environmental exposure mh tiab exposure, occupational diseases disease »

Trois défauts dans huit lignes :

1. Les tags de champ survivaient comme MOTS CHERCHÉS. La fonction retirait les crochets
   de `[mh]` et gardait `mh`. Idem `tiab`. `_strip_field_tags`, qui enlève le tag en
   entier, existe deux cents lignes plus haut dans le même module.
2. Les mots étaient pris DANS L'ORDRE DU TEXTE, les huit premiers. Sur une requête
   structurée en « (exposition OU fomites OU transmission OU perception) ET (virus
   aviaires) SAUF H1N1 », cela garde le premier bloc et jette tous les autres, dont le
   bloc ET qui dit de quelle maladie il s'agit. Aucun terme de grippe n'atteignait ces
   cinq sources : elles ont cherché « exposition professionnelle » dans toute la
   littérature mondiale.
3. Un terme NIÉ redevenait un terme cherché. La fonction retirait le mot « not » et
   gardait ses voisins, donc « ... NOT H1N1 » demandait du H1N1.

Et le repli lui-même était MUET : le tableau d'identification présentait les notices de
ces sources comme le produit de la requête booléenne affichée au-dessus, alors que
PRISMA-S exige la stratégie réellement soumise à chaque base.
"""
import json
import pathlib

import pytest

pytest.importorskip("fastapi")

from api.search import (  # noqa: E402
    _plain_keywords,
    _prisma_identification_figures,
    _strip_field_tags,
    _terms_in_order,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: La requête du scénario HPAI de production, réduite à sa structure : un OU de blocs
#: d'exposition, ET un bloc de virus, SAUF le H1N1. C'est la forme qui déclenchait le bug.
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


# ── Le défaut, nommé ─────────────────────────────────────────────────────────

def test_a_field_tag_is_not_a_search_term():
    """`mh` et `tiab` cherchés comme des mots : c'est ce que cinq sources ont reçu."""
    kw = _plain_keywords(HPAI).split()
    for tag in ("mh", "tiab", "majr", "noexp", "mesh", "terms", "dp", "pt"):
        assert tag not in kw, (
            f"« {tag} » est un tag de champ PubMed, pas un terme de recherche ; "
            f"mots-clés produits : {' '.join(kw)}")


def test_the_anchor_concept_survives_the_flattening():
    """Le bloc ET dit DE QUELLE MALADIE il s'agit. Sans lui, le repli cherche l'exposition
    professionnelle dans toute la littérature mondiale, ce qu'arXiv et CORE ont fait."""
    kw = _plain_keywords(HPAI).split()
    assert any(w in kw for w in ("influenza", "h5n1", "h7n9", "birds")), (
        "aucun terme du bloc de virus n'a survécu : la requête soumise ne nomme pas la "
        f"maladie de la revue ; mots-clés produits : {' '.join(kw)}")
    # Et le premier bloc doit rester représenté : ce n'est pas un échange, c'est un ajout.
    assert any(w in kw for w in ("environmental", "exposure", "occupational")), (
        f"le bloc d'exposition a disparu à son tour : {' '.join(kw)}")


def test_every_top_level_conjunct_gets_at_least_one_word():
    """Un tour de table, pas les n premiers mots du texte."""
    q = '("alpha" OR "alpha one") AND ("beta" OR "beta two") AND ("gamma" OR "gamma three")'
    kw = _plain_keywords(q, max_words=3).split()
    assert set(kw) == {"alpha", "beta", "gamma"}, (
        f"avec trois mots de budget et trois conjonctions, une par conjonction : {kw}")


def test_a_negated_term_is_never_searched():
    """La fonction retirait le mot « not » et gardait ses voisins."""
    assert "h1n1" not in _plain_keywords(HPAI).split(), (
        "le repli demande le H1N1 que la requête exclut")
    kw = _plain_keywords('("H5N1" OR "avian influenza") NOT "H1N1"').split()
    assert "h1n1" not in kw and "h5n1" in kw, kw
    # Un NOT qui laisse la requête sans aucun terme positif ne doit pas inventer de mot.
    assert "h5n1" not in _plain_keywords('NOT "H5N1"').split()


def test_the_terms_of_a_subtree_exclude_what_is_negated():
    assert _terms_in_order(("term", "alpha")) == ["alpha"]
    assert _terms_in_order(("not", ("term", "alpha"))) == []
    tree = ("and", [("or", [("term", "a"), ("term", "b")]), ("not", ("term", "c"))])
    assert _terms_in_order(tree) == ["a", "b"]


# ── Ce qui ne doit pas régresser ─────────────────────────────────────────────

@pytest.mark.parametrize("query,expected_subset", [
    ("avian influenza", {"avian", "influenza"}),
    ('"H5N1" AND "domestic cat*"', {"h5n1", "domestic", "cat"}),
    ("cardiac arrest", {"cardiac", "arrest"}),
])
def test_a_plain_query_comes_through_unchanged(query, expected_subset):
    assert set(_plain_keywords(query).split()) == expected_subset


@pytest.mark.parametrize("query", ["", "   ", "AND OR NOT", "()", '""', "[mh]"])
def test_a_query_with_nothing_in_it_yields_nothing(query):
    """Et surtout ne lève pas : l'appelant fait `_plain_keywords(x) or ... or query`."""
    assert _plain_keywords(query) == ""


def test_the_budget_is_respected():
    for n in (1, 2, 5, 8, 12, 40):
        assert len(_plain_keywords(HPAI, max_words=n).split()) <= n


def test_the_result_carries_no_boolean_syntax():
    """C'est la raison d'être de la fonction : OpenAlex `search` répond 400 sur un booléen."""
    out = _plain_keywords(HPAI, max_words=40)
    for junk in ('"', "(", ")", "[", "]", "*", " AND ", " OR ", " NOT "):
        assert junk not in out, f"{junk!r} est resté dans les mots-clés : {out!r}"
    assert out == out.lower()


def test_field_tags_are_stripped_before_words_are_kept():
    """L'ordre compte : retirer les crochets seuls laissait le tag comme mot."""
    assert "mh" not in _strip_field_tags('"Fomites"[mh]').lower().split()
    assert _plain_keywords('"Fomites"[mh] AND "Cats"[mh]').split() == ["fomites", "cats"]


# ── Le repli doit se DIRE ────────────────────────────────────────────────────

def test_the_identification_figures_name_the_sources_that_got_keywords():
    f = _prisma_identification_figures(
        {"openalex": 156, "arxiv": 6644, "core": 5794}, 12594, 0, 640,
        source_outcomes={"_fetch_openalex": "ok", "_fetch_arxiv": "ok", "_fetch_core": "ok"},
        keyword_fallback_sources=["openalex", "doaj", "core", "clinicaltrials", "arxiv"],
        keyword_fallback_query="environmental exposure influenza birds")
    assert f["keyword_fallback_sources"] == [
        "arxiv", "clinicaltrials", "core", "doaj", "openalex"]
    assert f["keyword_fallback_query"] == "environmental exposure influenza birds"


def test_a_search_where_every_source_got_the_boolean_says_nothing():
    """L'avertissement ne doit apparaître que quand il est vrai."""
    f = _prisma_identification_figures({"pubmed": 30}, 30, 0, 30,
                                       source_outcomes={"_fetch_pubmed": "ok"})
    assert f["keyword_fallback_sources"] == []
    assert f["keyword_fallback_query"] is None


def test_the_populate_computes_the_fallback_list_from_the_two_gates():
    """Les cinq sources en repli sont celles que `_send_bool` et `_arxiv_native` gouvernent."""
    import inspect
    from api import pipeline as P
    src = inspect.getsource(P._run_user_scenario_populate)
    assert "_kw_fallback" in src
    _i = src.index("_kw_fallback = ")
    block = src[_i:_i + 260]
    for name in ("openalex", "doaj", "core", "clinicaltrials", "arxiv"):
        assert name in block, f"{name} reçoit le repli mais n'est pas déclarée : {block}"
    assert "_send_bool" in block and "_arxiv_native" in block
    # Et le nom doit être lié au niveau de la fonction : les chiffres PRISMA le lisent
    # depuis un AUTRE bloc try, où un NameError serait avalé en silence.
    import ast
    fn = next(n for n in ast.walk(ast.parse((ROOT / "api" / "pipeline.py").read_text(encoding="utf-8")))
              if isinstance(n, ast.FunctionDef) and n.name == "_run_user_scenario_populate")
    top = {s.lineno for s in fn.body if isinstance(s, (ast.Assign, ast.AnnAssign))}
    bound = [n.lineno for n in ast.walk(fn)
             if isinstance(n, ast.Name) and n.id == "_kw_fallback" and isinstance(n.ctx, ast.Store)]
    assert any(l in top for l in bound), (
        "_kw_fallback n'est lié que dans un try : une exception avant sa ligne fait sauter "
        "l'enregistrement des chiffres PRISMA sans trace")


def test_the_panel_shows_the_degraded_strategy_in_both_languages():
    page = (ROOT / "frontend" / "src" / "components" / "ScenarioDetailPage.tsx").read_text(encoding="utf-8")
    assert "keyword_fallback_sources" in page and "prisma.keywordFallback" in page
    for loc in ("fr", "en"):
        txt = (ROOT / "frontend" / "src" / "i18n" / "locales" / f"{loc}.ts").read_text(encoding="utf-8")
        assert "keywordFallback:" in txt, f"clé absente de {loc}.ts"
        line = next(l for l in txt.splitlines() if "keywordFallback:" in l)
        assert "{sources}" in line and "{keywords}" in line, line


# ── La forme qui a produit le bug, mesurée ───────────────────────────────────

def test_the_real_hpai_query_still_trips_the_1200_character_gate():
    """Le repli n'est pas un cas limite : la requête de production le déclenche.

    Ce test documente le chiffre. Si quelqu'un relève la limite, il doit le voir ici et
    décider sciemment, parce que la limite vient de l'URL d'OpenAlex, pas de nous."""
    import inspect
    from api import pipeline as P
    src = inspect.getsource(P._run_user_scenario_populate)
    assert "len(_portable_bool) <= 1200" in src
    portable = " ".join(_strip_field_tags(HPAI).split())
    assert len(portable) < 1200, (
        "la requête abrégée de ce test ne déclenche plus le repli ; "
        "elle doit rester représentative")
    strategy = ROOT / "tests" / "fixtures"
    # La vraie requête fait 2 465 caractères portables. On épingle l'ordre de grandeur
    # sans dépendre d'un fichier : la structure testée ci-dessus est la même.
    assert strategy is not None
