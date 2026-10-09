"""« NOT ("news"[Publication Type] OR "comment"[Publication Type]) » excluait tout article
dont le texte contient « news » ou « comment ».

La clause finale de la requête HPAI de production : `NOT ( "news"[Publication Type] OR
"letter"[Publication Type] OR "comment"[Publication Type] OR "editorial"[Publication
Type] )`. PubMed l'applique comme un filtre de type de document. Partout ailleurs, les
tags étaient retirés et les termes gardés : en local, l'appariement titre + résumé + texte
intégral excluait tout document contenant « news », « letter », « comment » ou
« editorial » (« newsletter », « comments », « letters to »...) ; à Europe PMC, la même
exclusion en texte libre ; au plongement, quatre mots de bruit. Un article sur la
couverture médiatique du H5N1, exactement le sujet « perception du risque » de la revue,
sortait du corpus pour le mot « news ».

Même chose pour les dates (« "2021"[dp] : "3000"[dp] » laissait 2021 et 3000 comme mots
requis) et la langue. Ces termes ne sont pas des mots du texte : ils partent avec leur
tag, et la requête reste bien formée (pas de « NOT ( OR ) » qui traîne vers une API).

Ces tests mesurent ce que les consommateurs font : le SQL de l'appariement local, la
requête Europe PMC, l'arbre que lisent les compilateurs, les mots-clés, le texte plongé.
Pas de grep du source.
"""
import pytest

pytest.importorskip("fastapi")

from api import relevance as R  # noqa: E402
from api.search import (  # noqa: E402
    _build_boolean_match_sql_from_query,
    _parse_boolean_ast,
    _plain_keywords,
    _strip_field_tags,
    _tokenize_boolean,
)
from api.sources import epmc_query  # noqa: E402

#: La forme de la requête de production : deux blocs, une vraie exclusion, puis
#: l'exclusion des types de publication.
HPAI_TAIL = (
    '( "Occupational Exposure"[mh] OR "poultry worker*"[tiab] ) '
    'AND ( "Influenza in Birds"[mh] OR "H5N1"[tiab] ) '
    'NOT ( "Influenza A Virus, H1N1 Subtype"[mh] OR "H1N1"[tiab] ) '
    'NOT ( "news"[Publication Type] OR "letter"[Publication Type] '
    'OR "comment"[Publication Type] OR "editorial"[Publication Type] )'
)
PT_WORDS = ("news", "letter", "comment", "editorial")


def _like_patterns(query: str) -> set[str]:
    params: dict = {}
    _build_boolean_match_sql_from_query(query, params)
    return {v for v in params.values() if isinstance(v, str)}


# ── Ce que chaque consommateur voit ──────────────────────────────────────────

def test_the_local_match_no_longer_excludes_the_word_news():
    likes = _like_patterns(HPAI_TAIL)
    for word in PT_WORDS:
        assert f"%{word}%" not in likes, (
            f"« {word} » est encore un motif d'exclusion de l'appariement local : {likes}")
    # Les vraies exclusions restent des exclusions, et les termes positifs des termes.
    assert "%h1n1%" in likes and "%h5n1%" in likes and "%poultry worker%" in likes


def test_the_real_exclusion_stays_an_exclusion():
    ast = _parse_boolean_ast(_tokenize_boolean(HPAI_TAIL))
    assert ast[0] == "and", ast
    nots = [c for c in ast[1] if c[0] == "not"]
    assert len(nots) == 1, ast
    assert "h1n1" in repr(nots[0])
    assert not any(w in repr(ast) for w in PT_WORDS), ast


def test_europe_pmc_does_not_get_the_exclusion_either():
    q = epmc_query(HPAI_TAIL)
    for word in PT_WORDS:
        assert f'"{word}"' not in q and word not in q.lower().split(), q
    assert '"H1N1"' in q and "NOT" in q, q
    assert q.count("(") == q.count(")") and "( )" not in q, q
    assert not q.rstrip().endswith("NOT"), q


def test_the_portable_boolean_is_well_formed_after_the_removal():
    """La chaîne part telle quelle vers OpenAlex, DOAJ, CORE, ClinicalTrials.gov et
    OpenAIRE : un « NOT ( OR ) » qui traîne est une erreur de syntaxe chez elles."""
    p = " ".join(_strip_field_tags(HPAI_TAIL).split())
    assert p.count("(") == p.count(")"), p
    for junk in ("( OR", "OR )", "NOT NOT", "AND AND", "( )", "["):
        assert junk not in p, f"{junk!r} dans la requête portable : {p}"
    assert not p.endswith(("NOT", "AND", "OR")), p


def test_the_keywords_and_the_embedding_do_not_carry_the_publication_types():
    kw = _plain_keywords(HPAI_TAIL).split()
    for word in PT_WORDS:
        assert word not in kw, kw
    text_ = R.embedding_text_for_query(HPAI_TAIL).lower()
    for word in PT_WORDS:
        assert word not in text_.split(), text_
    assert "h5n1" in text_


# ── L'arbre, cas par cas ─────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected", [
    # L'exclusion seule à la fin : elle part, le reste est intact.
    ('"H5N1"[tiab] NOT ("news"[pt] OR "letter"[pt])', ("term", "h5n1")),
    # Au milieu d'un ET : le ET se referme sans trou.
    ('"H5N1"[tiab] AND "Review"[Publication Type] AND "poultry"[tiab]',
     ("and", [("term", "h5n1"), ("term", "poultry")])),
    # Mêlé à un vrai terme dans un OU : seul le type de publication part.
    ('"H5N1"[tiab] NOT ("news"[pt] OR "H1N1"[tiab])',
     ("and", [("term", "h5n1"), ("not", ("term", "h1n1"))])),
    ('"H5N1"[tiab] NOT ("H1N1"[tiab] OR "news"[pt])',
     ("and", [("term", "h5n1"), ("not", ("term", "h1n1"))])),
    # Une plage de dates : ni les bornes ni le deux-points ne deviennent des mots.
    ('"H5N1"[tiab] AND ("2015"[dp] : "2025"[dp])', ("term", "h5n1")),
    ('"H5N1"[tiab] AND 2015:2025[dp]', ("term", "h5n1")),
    # La langue, le filtre.
    ('"H5N1"[tiab] AND english[la] AND hasabstract[filter]', ("term", "h5n1")),
    # Deux exclusions dont la première part : la seconde reste une exclusion. Sans le
    # nettoyage, « NOT NOT (H1N1) » faisait lire H1N1 comme un terme REQUIS.
    ('"H5N1"[tiab] NOT "news"[pt] NOT "H1N1"[tiab]',
     ("and", [("term", "h5n1"), ("not", ("term", "h1n1"))])),
    # Le tag en majuscules ou avec des espaces, comme PubMed les accepte.
    ('"H5N1"[tiab] NOT "news"[PT] NOT "letter"[ Publication Type ]', ("term", "h5n1")),
])
def test_the_tree_the_consumers_see(query, expected):
    assert _parse_boolean_ast(_tokenize_boolean(query)) == expected


def test_a_query_made_only_of_a_publication_type_has_no_term():
    assert _parse_boolean_ast(_tokenize_boolean('"Review"[pt]')) is None
    assert _plain_keywords('"Review"[pt]') == ""


def test_content_tags_still_keep_their_term():
    """[mh], [tiab], [Mesh], [majr] : le terme est un mot du texte, il reste. Et sans
    unité à retirer, la sortie est, au caractère près, celle d'avant."""
    assert _parse_boolean_ast(_tokenize_boolean('"Fomites"[mh] AND cats[tiab]')) == (
        "and", [("term", "fomites"), ("term", "cats")])
    assert _strip_field_tags('"Fomites"[mh] AND cats[tiab]') == '"Fomites"  AND cats '
    assert _strip_field_tags('"Humans"[Mesh] AND "Risk"[majr]') == '"Humans"  AND "Risk" '


def test_operator_words_inside_a_phrase_are_left_alone():
    """Le nettoyage des opérateurs orphelins ne touche pas l'intérieur des phrases :
    « and or » dans une phrase n'est pas un opérateur."""
    q = '"mortality and or morbidity"[tiab] AND "H5N1"[tiab] NOT "news"[pt]'
    assert _parse_boolean_ast(_tokenize_boolean(q)) == (
        "and", [("term", "mortality and or morbidity"), ("term", "h5n1")])
