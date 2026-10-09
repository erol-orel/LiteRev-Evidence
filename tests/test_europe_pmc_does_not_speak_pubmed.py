"""Europe PMC recevait les tags de champ de PubMed, qu'elle apparie comme du TEXTE.

Troisième cause du tableau d'identification du scénario HPAI, après le repli mots-clés et
l'URL de GET trop longue. Ce tableau porte `europepmc: 0`.

Europe PMC ne parle pas `[mh]` ni `[tiab]` : sa syntaxe de champ est la sienne
(`MESH:`, `TITLE:`, `ABSTRACT:`). Un tag de PubMed n'y est pas une erreur, c'est du texte,
que presque aucune notice ne contient. Mesuré sur la sonde de production, même question,
trois formulations :

    avian influenza AND occupational exposure                   2 398 notices
    "Influenza in Birds"[mh] AND "Occupational Exposure"[mh]         3 notices
    "avian influenza"[tiab] AND "occupational*"[tiab]               6 notices

Les tags coûtaient donc 99,9 % du rappel de la deuxième source biomédicale de la
fédération. La requête du scénario HPAI en porte des dizaines.

Quatre points d'appel envoyaient la requête taguée : Europe PMC et les préprints, dans la
recherche en direct comme dans le populate. Plus la sonde de diagnostic, qui mesurait donc
un rappel que la fédération n'obtenait pas.
"""
import inspect
import pathlib

import pytest

pytest.importorskip("fastapi")

import api.sources as S  # noqa: E402
from api.sources import EPMC_SEARCH_URL, epmc_query  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
HPAI = ('( ("Environmental Exposure"[mh] OR "Occupational Exposure"[mh] '
        'OR "Occupational Disease*"[tiab]) '
        'AND ("Influenza in Birds"[mh] OR "avian influenza"[tiab] OR "H5N1"[tiab]) ) '
        'NOT ("news"[Publication Type] OR "letter"[Publication Type])')


# ── La requête qu'Europe PMC reçoit ──────────────────────────────────────────

def test_no_pubmed_field_tag_survives():
    q = epmc_query(HPAI)
    for tag in ("[mh]", "[tiab]", "[Publication Type]", "[mh:noexp]", "[majr]", "[dp]"):
        assert tag not in q, f"{tag} part encore vers Europe PMC : {q[:160]}"
    assert "[" not in q and "]" not in q


def test_the_terms_and_the_operators_survive():
    """Seuls les tags partent : la requête doit rester la même recherche.

    À une exception près, qui est l'objet de `test_a_publication_type_is_not_a_word.py` :
    l'exclusion des types de publication n'est pas une recherche de texte. Envoyée telle
    quelle, elle faisait exclure à Europe PMC tout enregistrement dont le texte contient
    « news » ou « letter »."""
    q = epmc_query(HPAI)
    for kept in ('"Environmental Exposure"', '"Occupational Exposure"', '"avian influenza"',
                 '"H5N1"', '"Influenza in Birds"', "AND", "OR", "(", ")"):
        assert kept in q, f"{kept} a disparu de la requête : {q}"
    for gone in ('"news"', '"letter"', "NOT"):
        assert gone not in q, f"{gone} part encore vers Europe PMC comme texte : {q}"
    assert q.count("(") == q.count(")") and "( )" not in q, q


def test_an_input_made_only_of_tags_yields_nothing_not_the_tag():
    """Un repli « si le retrait ne laisse rien, garde l'original » rendait le tag."""
    assert epmc_query("[mh]") == ""
    assert epmc_query("[tiab][mh][dp]") == ""
    assert epmc_query("") == ""
    assert epmc_query("   ") == ""
    assert epmc_query(None) == ""


def test_the_whitespace_the_stripping_leaves_is_collapsed():
    """`_strip_field_tags` remplace le tag par une espace : deux termes collés par un tag
    se retrouvaient séparés par des espaces multiples, et la requête n'était plus
    comparable d'un appel à l'autre."""
    q = epmc_query('"A"[mh]   OR    "B"[tiab]')
    assert q == '"A" OR "B"', repr(q)
    assert "  " not in q


# ── Les quatre points d'appel ────────────────────────────────────────────────

def test_the_live_fetchers_both_go_through_the_helper():
    for fn in (S._live_fetch_europepmc, S._live_fetch_preprints):
        src = inspect.getsource(fn)
        assert "epmc_query(" in src, (
            f"{fn.__name__} envoie encore la requête telle quelle à Europe PMC")
        assert '"query": query' not in src, (
            f"{fn.__name__} passe encore `query` brute")


def test_the_populate_fetchers_both_go_through_the_helper():
    from api import pipeline as P
    src = inspect.getsource(P._run_user_scenario_populate)
    assert "_epmc_q = epmc_query(_boolean)" in src
    # Europe PMC et les préprints lisent la variable préparée, pas `_boolean`.
    assert '"query": _epmc_q' in src, "le fetcher europepmc envoie encore `_boolean`"
    assert '_pp_query = f"({_epmc_q}) AND (SRC:PPR)"' in src, (
        "le fetcher préprints envoie encore `_boolean`")
    assert '"query": _boolean' not in src, "un appel Europe PMC envoie encore la requête taguée"


def test_the_diagnostic_probe_measures_what_the_federation_sends():
    """La sonde annonçait un rappel que la fédération n'obtenait pas."""
    src = inspect.getsource(S.sources_health)
    _i = src.index('("EuropePMC"')
    assert "epmc_query(query)" in src[_i:_i + 200], (
        "la sonde Europe PMC mesure encore la requête brute")


def test_the_url_is_written_once():
    """Quatre appels répétaient la même URL ; une faute de frappe dans l'un d'eux aurait
    donné une source muette sans que rien ne le dise."""
    for mod in ("api/sources.py", "api/pipeline.py"):
        txt = (ROOT / mod).read_text(encoding="utf-8")
        literal = txt.count("https://www.ebi.ac.uk/europepmc/webservices/rest/search")
        assert literal <= 1, (
            f"{mod} écrit encore l'URL d'Europe PMC {literal} fois au lieu d'utiliser "
            "EPMC_SEARCH_URL")
    assert EPMC_SEARCH_URL.startswith("https://www.ebi.ac.uk/europepmc/")


def test_the_full_text_fetcher_is_not_in_scope():
    """Le récupérateur de texte intégral interroge Europe PMC par DOI et PMCID, pas par
    requête : il n'a pas de tag à retirer, et ce test dit pourquoi il n'a pas changé."""
    from api import pipeline as P
    src = inspect.getsource(P)
    _i = src.index("_EPMC_BASE_FT")
    assert "_EPMC_BASE_FT" in src[_i:_i + 80]
