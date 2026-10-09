"""Une recherche dit ce qu'elle a FAIT, source par source.

Trois mensonges tenaient ensemble, et ils portaient tous sur la même chose : la
couverture annoncée d'une recherche.

1. Les cinq fetchers de la recherche en direct avalaient leur propre exception et
   renvoyaient une liste vide. L'appelant a pourtant une branche `error` et une branche
   `empty` : la branche `error` ne pouvait jamais s'allumer. Une source injoignable
   s'affichait donc « aucun résultat », qui est une affirmation sur la LITTÉRATURE, pas
   sur le réseau.

2. `sources_queried` recevait le nom de chaque future terminée, échecs compris, si bien
   que le panneau listait parmi les sources interrogées celles qui n'avaient rien pu
   interroger.

3. Le tableau d'identification PRISMA filtrait les sources à zéro
   (`if int(v or 0) > 0`) tout en annonçant un nombre de sources interrogées qui les
   comptait. Sur le scénario HPAI, il affichait « 16 130 / 7 sources interrogées » pour
   une recherche où douze fetchers étaient partis, et PubMed n'y figurait pas pendant que
   327 des 640 articles screenés portaient pubmed comme source.

Ces tests épinglent les trois issues et l'arithmétique PRISMA qui doit tenir avec elles.
"""
import pytest

pytest.importorskip("fastapi")

from api.search import (  # noqa: E402
    SOURCE_OUTCOMES,
    SOURCE_OUTCOMES_COUNTED,
    _coverage_caveat,
    _outcome_summary,
    _prisma_identification_figures,
    _source_label,
)


# ── Les issues, nommées ──────────────────────────────────────────────────────

def test_the_five_outcomes_are_exclusive_and_only_three_count_as_searched():
    assert set(SOURCE_OUTCOMES_COUNTED) < set(SOURCE_OUTCOMES)
    assert set(SOURCE_OUTCOMES_COUNTED) == {"ok", "empty", "cached"}
    # Un échec, une coupure et une absence de clé ne sont PAS des interrogations.
    for o in ("error", "cut_by_budget", "skipped"):
        assert o not in SOURCE_OUTCOMES_COUNTED


def test_the_fetcher_name_becomes_the_source_name():
    assert _source_label("_fetch_europepmc") == "europepmc"
    assert _source_label("_fetch_core") == "core"
    assert _source_label("pubmed") == "pubmed"


def test_the_coverage_line_names_what_did_not_answer():
    caveat = _coverage_caveat({
        "_fetch_pubmed": "ok", "_fetch_openalex": "error",
        "_fetch_core": "skipped", "_fetch_crossref": "cut_by_budget",
    })
    assert "openalex" in caveat and "échec" in caveat
    assert "core" in caveat and "clé" in caveat
    assert "crossref" in caveat and "budget" in caveat
    assert "pubmed" not in caveat          # celle qui a répondu n'a rien à expliquer


def test_the_coverage_line_is_silent_when_everything_answered():
    assert _coverage_caveat({"_fetch_pubmed": "ok", "_fetch_crossref": "empty"}) == ""
    assert _coverage_caveat({}) == ""


def test_the_outcome_summary_counts_by_issue():
    s = _outcome_summary({"a": "ok", "b": "ok", "c": "error", "d": "skipped"})
    assert "ok: 2" in s and "error: 1" in s and "skipped: 1" in s


# ── Le tableau d'identification ──────────────────────────────────────────────

def _figures(**kw):
    base = dict(records_by_source={"pubmed": 300, "europepmc": 27},
                unique_records=300, duplicate_rows_removed=0, corpus_total=300)
    base.update(kw)
    return _prisma_identification_figures(
        base.pop("records_by_source"), base.pop("unique_records"),
        base.pop("duplicate_rows_removed"), base.pop("corpus_total"), **base)


def test_a_source_that_failed_keeps_its_row_at_zero():
    f = _figures(source_outcomes={"_fetch_pubmed": "ok", "_fetch_openalex": "error",
                                  "_fetch_core": "skipped"})
    assert f["records_by_source"]["openalex"] == 0, (
        "la source en échec a disparu du tableau : le lecteur ne peut pas distinguer "
        "« a échoué » de « n'existe pas »")
    assert f["records_by_source"]["core"] == 0
    assert f["sources_failed"] == ["openalex"]
    assert f["sources_skipped"] == ["core"]


def test_sources_searched_excludes_the_failures_it_used_to_count():
    f = _figures(source_outcomes={
        "_fetch_pubmed": "ok", "_fetch_europepmc": "ok", "_fetch_crossref": "empty",
        "_fetch_openalex": "error", "_fetch_doaj": "error", "_fetch_core": "skipped",
        "_fetch_arxiv": "cut_by_budget",
    })
    assert f["sources_launched"] == 7
    assert f["sources_searched"] == 3      # ok + ok + empty
    assert sorted(f["sources_failed"]) == ["doaj", "openalex"]
    assert f["sources_cut_off"] == ["arxiv"]


def test_a_source_at_zero_adds_nothing_to_the_identified_total():
    f = _figures(source_outcomes={"_fetch_pubmed": "ok", "_fetch_openalex": "error"})
    assert f["records_identified"] == 327   # 300 + 27, l'échec n'invente rien


def test_the_local_library_is_identification_by_another_method():
    f = _figures(records_by_source={"pubmed": 300, "db_cache": 120},
                 unique_records=400, corpus_total=400)
    assert f["records_identified_databases"] == 300
    assert f["records_identified_library"] == 120
    assert "db_cache" not in f["records_by_source"], (
        "la bibliothèque locale est restée dans le tableau des bases interrogées")
    # La somme reste le total identifié, pour que l'arithmétique PRISMA tienne.
    assert f["records_identified"] == 420


def test_the_prisma_arithmetic_still_balances_with_the_library_split():
    f = _figures(records_by_source={"pubmed": 300, "db_cache": 120},
                 unique_records=380, duplicate_rows_removed=5, corpus_total=360)
    assert (f["records_identified"] - f["duplicates_removed"]
            - f["removed_before_screening"]) == f["records_screened"]


def test_the_cap_that_produced_the_figures_is_stored_with_them():
    f = _figures(per_source_cap=2000)
    assert f["per_source_cap"] == 2000, (
        "sans le plafond, un chiffre ne peut pas être rattaché au run qui l'a produit, "
        "et trois chemins de l'application en appliquaient trois différents")


def test_no_outcomes_given_leaves_the_old_shape_intact():
    """Les scénarios déjà en base n'ont pas d'issues enregistrées : le panneau doit
    rester lisible, sans inventer des sources ni un dénominateur."""
    f = _figures()
    assert f["source_outcomes"] == {}
    assert f["sources_launched"] == 0 and f["sources_searched"] == 0
    assert f["records_by_source"] == {"pubmed": 300, "europepmc": 27}
    assert f["per_source_cap"] is None


# ── Les fetchers live lèvent, au lieu de rendre une liste vide ───────────────

def test_the_live_fetchers_no_longer_swallow_their_own_failure():
    import inspect
    from api import sources as S
    for fn in (S._live_fetch_openalex, S._live_fetch_crossref,
               S._live_fetch_europepmc, S._live_fetch_preprints):
        src = inspect.getsource(fn)
        assert "raise SourceFetchError" in src, (
            f"{fn.__name__} avale encore son exception : la branche `error` de "
            "l'appelant ne peut pas s'allumer et l'échec se lit « aucun résultat »")
        assert "raise_for_status()" in src, (
            f"{fn.__name__} ne regarde pas le code HTTP : un 429 se lit « aucun résultat »")


def test_pubmed_reports_a_partial_fetch_rather_than_an_empty_one():
    import inspect
    from api import sources as S
    src = inspect.getsource(S._live_fetch_pubmed)
    assert "raise PartialSourceFetch" in src, (
        "l'esearch réussi puis l'esummary en échec donnent un vrai total et aucune "
        "notice : ni « ok » ni « vide »")


def test_the_partial_fetch_carries_what_came_back():
    from api.sources import PartialSourceFetch
    e = PartialSourceFetch([{"title": "a"}], 306, "esummary 500")
    assert e.items and e.total == 306 and "esummary" in e.reason


def test_a_failed_source_is_not_listed_among_the_sources_searched():
    import inspect
    from api import sources as S
    src = inspect.getsource(S._federated_live_search)
    head, _, tail = src.partition("except Exception as _fe:")
    assert "sources_queried.append(name)" not in tail, (
        "le nom est encore ajouté dans la branche d'échec : le panneau liste la source "
        "parmi celles qu'il dit avoir interrogées")
    assert tail.count("source_status[name]") == 1
