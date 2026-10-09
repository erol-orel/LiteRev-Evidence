"""L'issue `error` était servie nue : « openalex : échec », et rien de plus.

Une recherche de contrôle lancée en production a rendu `openalex: error` et
`doaj: error`. Comprendre pourquoi a demandé de lire le code des deux fetchers, de
comparer trois points d'appel et de deviner, parce que la raison n'existait que dans une
ligne de journal du serveur, inaccessible depuis l'interface. L'équipe italienne aura le
même problème, et plus souvent que moi.

La raison est maintenant enregistrée avec l'issue, servie dans les chiffres
d'identification, affichée dans l'infobulle de la source ET en clair sous le paragraphe de
couverture, parce qu'une infobulle ne se recopie pas dans un rapport.

Ce fichier épingle aussi le défaut que cette recherche de contrôle a mis au jour :
`per_page` au lieu de `per-page` dans l'appel OpenAlex du populate, seul des trois appels
du dépôt à l'écrire ainsi, et justement celui qui construit le corpus.
"""
import inspect
import pathlib
import re

import pytest

pytest.importorskip("fastapi")

from api.pipeline import _run_user_scenario_populate  # noqa: E402
from api.search import _prisma_identification_figures  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = inspect.getsource(_run_user_scenario_populate)


# ── La raison voyage ─────────────────────────────────────────────────────────

def test_every_source_failure_passes_its_reason():
    """Treize `except` marquent une source en échec ; aucun ne doit le faire en silence."""
    bare = SRC.count("_mark_source_error()")
    assert bare == 0, (
        f"{bare} site(s) marquent encore un échec sans sa raison ; la carte PRISMA dira "
        "« échec » sans dire de quoi")
    with_reason = len(re.findall(r"_mark_source_error\(_e[_a-z]*\)", SRC))
    assert with_reason >= 13, (
        f"seulement {with_reason} sites passent la raison, 13 attendus")


def test_the_reasons_are_kept_by_source_not_counted():
    """Un ensemble ne retient pas de raison : il fallait un dictionnaire."""
    assert "_fetcher_errors: dict[str, str] = {}" in SRC, (
        "_fetcher_errors est encore un ensemble, qui ne peut porter aucune raison")
    assert "_fetcher_errors.add(" not in SRC
    # La future en échec, dans la boucle as_completed, porte aussi sa raison.
    assert "_fetcher_errors[_fname] = str(_fe)[:200]" in SRC


def test_a_second_failure_on_the_same_source_does_not_erase_the_first_reason():
    """`_mark_source_error()` sans argument existe encore pour les appels internes :
    il ne doit pas remplacer une raison déjà connue par une chaîne vide."""
    _i = SRC.index("def _mark_source_error")
    body = SRC[_i:_i + 1000]
    assert "_fetcher_errors.get(_f)" in body, (
        "un marquage sans raison écrase la raison déjà enregistrée")


def test_the_reason_reaches_the_identification_figures():
    f = _prisma_identification_figures(
        {"openalex": 0, "pubmed": 30}, 30, 0, 30,
        source_outcomes={"_fetch_openalex": "error", "_fetch_pubmed": "ok"},
        source_error_reasons={"_fetch_openalex": "403 Client Error: Forbidden for url: ..."})
    assert f["source_error_reasons"]["openalex"].startswith("403 Client Error")
    assert "pubmed" not in f["source_error_reasons"], (
        "une source qui a réussi ne doit pas porter de raison")


def test_an_empty_reason_is_not_served_as_a_reason():
    f = _prisma_identification_figures(
        {"doaj": 0}, 0, 0, 0,
        source_outcomes={"_fetch_doaj": "error"},
        source_error_reasons={"_fetch_doaj": "", "_fetch_core": None})
    assert f["source_error_reasons"] == {}, (
        "une raison vide affichée donnerait « doaj :  » sous le paragraphe de couverture")


def test_a_search_with_no_failure_carries_no_reasons():
    f = _prisma_identification_figures({"pubmed": 30}, 30, 0, 30,
                                       source_outcomes={"_fetch_pubmed": "ok"})
    assert f["source_error_reasons"] == {}


def test_the_reason_is_bounded():
    """Une trace de pile entière dans un champ affiché casserait la carte."""
    f = _prisma_identification_figures(
        {"core": 0}, 0, 0, 0, source_outcomes={"_fetch_core": "error"},
        source_error_reasons={"_fetch_core": "x" * 5000})
    assert len(f["source_error_reasons"]["core"]) <= 200


# ── La carte l'affiche, et le rapport peut le recopier ───────────────────────

def test_the_card_shows_the_reason_in_the_tooltip_and_in_the_text():
    page = (ROOT / "frontend" / "src" / "components" / "ScenarioDetailPage.tsx").read_text(encoding="utf-8")
    assert page.count("source_error_reasons") >= 2, (
        "la raison doit être à la fois dans l'infobulle et dans le paragraphe : une "
        "infobulle ne se recopie pas dans un rapport")
    api_ts = (ROOT / "frontend" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "source_error_reasons?: Record<string, string>;" in api_ts


# ── Le défaut que la recherche de contrôle a mis au jour ─────────────────────

def test_openalex_pagination_parameter_is_spelled_the_same_everywhere():
    """`per-page` avec un trait d'union : le nom du paramètre chez OpenAlex.

    Trois appels OpenAlex dans le dépôt, deux l'écrivaient ainsi, et le troisième, celui
    du populate qui construit le corpus, écrivait `per_page`. OpenAlex refuse un paramètre
    inconnu."""
    offenders = []
    for mod in ("api/pipeline.py", "api/sources.py"):
        txt = (ROOT / mod).read_text(encoding="utf-8")
        for m in re.finditer(r"api\.openalex\.org/works\"?,?\n(?:[^\n]*\n){0,12}", txt):
            # Les commentaires retirés : l'un d'eux cite l'ancienne orthographe pour
            # expliquer le correctif, et ce n'est pas un défaut.
            block = re.sub(r"#[^\n]*", "", m.group(0))
            if '"per_page"' in block:
                offenders.append(f"{mod}:{txt[:m.start()].count(chr(10)) + 1}")
    assert not offenders, (
        "`per_page` au lieu de `per-page` dans un appel OpenAlex : " + ", ".join(offenders))
    # Et la forme correcte est bien présente aux trois endroits.
    both = (ROOT / "api" / "pipeline.py").read_text(encoding="utf-8") + \
           (ROOT / "api" / "sources.py").read_text(encoding="utf-8")
    assert both.count('"per-page"') >= 3, (
        f"seulement {both.count(chr(34) + 'per-page' + chr(34))} appels OpenAlex paginent "
        "avec le bon nom de paramètre")
