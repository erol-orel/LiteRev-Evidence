"""Les chiffres que la recherche enregistre doivent ATTEINDRE la carte.

Trouvé en vérifiant mon propre travail : les chiffres d'identification portaient, depuis
les premiers commits de #326, les sources passées en mots-clés, les mots-clés reçus et la
raison de chaque échec ; l'interface savait les afficher ; mais l'endpoint PRISMA, qui relit
les chiffres stockés et construit le bloc `identification`, ne les recopiait pas. La ligne
orange de stratégie dégradée et les raisons d'échec n'auraient jamais été vues.

Ce test enregistre des chiffres par le même chemin que la recherche (`_store_prisma_
identification`) et les relit par le même endpoint que la carte. Il échoue pour toute clé
que l'un écrit et que l'autre n'émet pas.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from api.core import engine  # noqa: E402
from api.search import _prisma_identification_figures, _store_prisma_identification  # noqa: E402

SID = "usr-prismapass0001"


@pytest.fixture
def scenario():
    with engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
        c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count) "
                       "VALUES (:s, :n, :q, 0)"), {"s": SID, "n": "prisma pass-through", "q": "hpai"})
    yield SID
    with engine.begin() as c:
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})


def _figures_with_everything():
    return _prisma_identification_figures(
        {"pubmed": 1994, "openalex": 0, "preprint": 5, "doaj": 999}, 2998, 0, 2998,
        method="populate", federation_incomplete=False,
        source_outcomes={"_fetch_pubmed": "ok", "_fetch_openalex": "error",
                         "_fetch_preprints": "ok", "_fetch_doaj": "ok", "_fetch_core": "skipped"},
        per_source_cap=2000,
        keyword_fallback_sources=["openalex", "doaj", "core", "clinicaltrials", "arxiv"],
        keyword_fallback_query="occupational exposure influenza birds",
        keyword_fallback_queries={"_fetch_openaire": '"occupational exposure" AND "avian influenza"'},
        source_error_reasons={"_fetch_openalex": "403 Client Error: Forbidden"},
        source_totals={"pubmed": 7412, "doaj": 999})


def test_every_recorded_key_reaches_the_card(scenario):
    figures = _figures_with_everything()
    _store_prisma_identification(scenario, figures)
    with TestClient(main.app) as c:
        r = c.get(f"/user-scenarios/{scenario}/prisma")
    assert r.status_code == 200, r.text[:300]
    ident = r.json()["identification"]

    # OpenAIRE, qui a sa propre requête, rejoint la liste par le dictionnaire des requêtes.
    assert ident["keyword_fallback_sources"] == ["arxiv", "clinicaltrials", "core", "doaj", "openaire", "openalex"]
    assert ident["keyword_fallback_query"] == "occupational exposure influenza birds"
    assert ident["keyword_fallback_queries"] == {"openaire": '"occupational exposure" AND "avian influenza"'}
    assert ident["source_error_reasons"] == {"openalex": "403 Client Error: Forbidden"}
    assert ident["source_totals"] == {"pubmed": 7412, "doaj": 999}
    assert ident["sources_capped"] == ["pubmed"], (
        "1 994 gardés sur 7 412 annoncés : PubMed doit être dite plafonnée")
    # Et ce qui marchait déjà marche toujours.
    assert ident["per_source_cap"] == 2000
    assert ident["sources_failed"] == ["openalex"]
    assert ident["sources_skipped"] == ["core"]
    # Une ligne par source, sous son nom : le compte des préprints est sous « preprints »,
    # la clé d'écriture du fetcher ne remonte plus à côté.
    assert ident["by_source"]["preprints"] == 5 and "preprint" not in ident["by_source"]


def test_a_legacy_run_without_outcomes_emits_none_of_the_new_keys(scenario):
    """Une recherche antérieure au registre : absence, pas zéro, pas liste vide inventée."""
    legacy = _prisma_identification_figures({"pubmed": 300, "db_cache": 120}, 400, 0, 400)
    legacy.pop("source_outcomes", None)
    for k in ("keyword_fallback_sources", "keyword_fallback_query", "keyword_fallback_queries",
              "source_error_reasons", "source_totals", "sources_capped", "sources_launched",
              "sources_searched"):
        legacy.pop(k, None)
    _store_prisma_identification(scenario, legacy)
    with TestClient(main.app) as c:
        r = c.get(f"/user-scenarios/{scenario}/prisma")
    assert r.status_code == 200
    ident = r.json()["identification"]
    for k in ("keyword_fallback_sources", "keyword_fallback_query", "keyword_fallback_queries",
              "source_error_reasons", "source_totals", "sources_capped", "sources_searched"):
        assert k not in ident, f"{k} est servi pour une recherche qui ne l'a jamais enregistré"
    # Le partage bases / bibliothèque reste déduit de la ligne db_cache.
    assert ident["records_identified_library"] == 120
    assert ident["records_identified_databases"] == 300


def test_the_stored_and_served_figures_use_the_same_vocabulary():
    """Toute clé que `_prisma_identification_figures` produit dans le registre des sources
    doit être relue par l'endpoint. Lu sur le code : la liste des clés émises sous
    `if _figures.get("source_outcomes") is not None`."""
    import inspect
    from api import review as R
    src = inspect.getsource(R)
    _i = src.index('if _figures.get("source_outcomes") is not None:')
    block = src[_i:src.index("# La séparation", _i)]
    produced = _figures_with_everything()
    for key in ("source_outcomes", "sources_launched", "sources_searched", "sources_failed",
                "sources_skipped", "sources_cut_off", "keyword_fallback_sources",
                "keyword_fallback_query", "source_error_reasons", "source_totals",
                "sources_capped"):
        assert key in produced, f"{key} n'est plus produit par les chiffres"
        assert f'"{key}"' in block, f"{key} est produit par la recherche mais pas relu par l'endpoint"
