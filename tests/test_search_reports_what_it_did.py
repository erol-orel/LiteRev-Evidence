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


def test_the_health_probe_counts_the_same_thing_for_every_source():
    """Le diagnostic des sources sert à les COMPARER : six compteurs, une définition.

    La sonde PubMed rendait `len(idlist)`, qui vaut au plus `retmax` (1 ici), pendant que
    les cinq autres rendaient leur total réel. Sur une requête à 300 000 enregistrements,
    PubMed s'affichait « 1 » et passait pour la source la plus maigre de la fédération."""
    import inspect
    from api import sources as S
    src = inspect.getsource(S.sources_health)
    _probes = src[src.index("probes = ["):src.index("def _probe")]
    assert "len(j.get(\"esearchresult\"" not in _probes, (
        "la sonde PubMed compte encore les identifiants rendus, pas les enregistrements")
    assert 'esearchresult", {}).get("count"' in _probes
    # Les cinq autres lisent bien un total, pas une longueur de page.
    for marker in ('meta", {}).get("count")', 'message", {}).get("total-results")',
                   'get("hitCount")', 'get("totalCount")'):
        assert marker in _probes, f"une sonde ne lit plus son total : {marker}"


class _EutilsSpy:
    """eutils simulé : l'esearch trouve `n_ids` identifiants, l'esummary en rend `n_back`."""

    def __init__(self, n_ids: int, n_back: int, count: int | None = None):
        self.n_ids, self.n_back = n_ids, n_back
        self.count = n_ids if count is None else count

    def _resp(self, payload):
        class _R:
            status_code = 200

            def json(_self):
                return payload

            def raise_for_status(_self):
                return None
        return _R()

    def get(self, url, params=None, timeout=None, **kw):
        if "esearch" in url:
            return self._resp({"esearchresult": {"count": str(self.count),
                                                 "idlist": [str(39000000 + i) for i in range(self.n_ids)]}})
        if "esummary" in url:
            uids = [str(39000000 + i) for i in range(self.n_back)]
            result = {"uids": uids}
            for u in uids:
                result[u] = {"title": f"t{u}", "pubdate": "2025 Jan", "articleids": [], "authors": []}
            return self._resp({"result": result})
        raise AssertionError(url)

    post = get


def test_an_esummary_that_succeeds_with_nothing_in_it_is_also_partial(monkeypatch):
    """`{"result": {"uids": []}}` est un 200. Rien ne levait, donc rien ne le disait.

    Sous charge, eutils rend ce corps pour des identifiants que l'esearch venait de
    donner. La fonction rendait alors `([], 306)` et l'appelant lisait « vide » : une
    affirmation sur la littérature pour un aller-retour manqué.

    MESURÉ, et non lu dans le source : la première version de ce test cherchait la ligne
    `len(results) < len(ids)` dans le texte, et la relecture a montré qu'on pouvait
    déplacer la garde APRÈS le `return`, ce qui la rend morte, sans que le test rougisse."""
    import sys
    from api import sources as S
    monkeypatch.setitem(sys.modules, "requests", _EutilsSpy(n_ids=30, n_back=0, count=306))
    monkeypatch.setattr(S, "_NCBI_LAST", [0.0], raising=False)
    monkeypatch.setattr(S, "_NCBI_MIN_INTERVAL", 0.0, raising=False)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    with pytest.raises(S.PartialSourceFetch) as exc:
        S._live_fetch_pubmed("avian influenza", 30)
    assert exc.value.total == 306, "le vrai total de l'esearch doit voyager avec le partiel"
    assert exc.value.items == []
    assert "30" in exc.value.reason and "0" in exc.value.reason


def test_an_esummary_that_returns_fewer_than_asked_is_partial_and_keeps_what_came(monkeypatch):
    """Vingt notices sur trente : partiel, et les vingt sont gardées."""
    import sys
    from api import sources as S
    monkeypatch.setitem(sys.modules, "requests", _EutilsSpy(n_ids=30, n_back=20))
    monkeypatch.setattr(S, "_NCBI_LAST", [0.0], raising=False)
    monkeypatch.setattr(S, "_NCBI_MIN_INTERVAL", 0.0, raising=False)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    with pytest.raises(S.PartialSourceFetch) as exc:
        S._live_fetch_pubmed("avian influenza", 30)
    assert len(exc.value.items) == 20
    assert exc.value.total == 30


def test_a_complete_esummary_is_not_partial(monkeypatch):
    """Et le chemin normal ne doit pas lever : trente demandés, trente rendus."""
    import sys
    from api import sources as S
    monkeypatch.setitem(sys.modules, "requests", _EutilsSpy(n_ids=30, n_back=30))
    monkeypatch.setattr(S, "_NCBI_LAST", [0.0], raising=False)
    monkeypatch.setattr(S, "_NCBI_MIN_INTERVAL", 0.0, raising=False)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    results, total = S._live_fetch_pubmed("avian influenza", 30)
    assert len(results) == 30 and total == 30


def test_a_failed_source_is_not_listed_among_the_sources_searched():
    import inspect
    from api import sources as S
    src = inspect.getsource(S._federated_live_search)
    head, _, tail = src.partition("except Exception as _fe:")
    assert "sources_queried.append(name)" not in tail, (
        "le nom est encore ajouté dans la branche d'échec : le panneau liste la source "
        "parmi celles qu'il dit avoir interrogées")
    assert tail.count("source_status[name]") == 1


# ── Le fichier doit s'ouvrir dans un tableur ────────────────────────────────

def test_the_csv_keeps_its_byte_order_mark_at_the_very_first_byte():
    """Excel ne détecte l'UTF-8 que par un BOM en PREMIER octet.

    Le bloc de provenance ajouté en tête le repoussait de sept lignes : le fichier
    s'ouvrait alors en « Rossi MÃ¼ller » au lieu de « Rossi Müller ». Vu sur l'export
    réel du scénario HPAI, juste après le déploiement."""
    from api.exports import render_export
    body = render_export("csv", [{"rank": 1, "id": 9, "title": "Rossi Müller"}], "T",
                         {"scenario": "S", "subset_label": "relevant", "n_articles": 1,
                          "similarity_threshold": 0.4226})
    assert body[:3] == b"\xef\xbb\xbf", "le BOM n'est plus le premier octet"
    text = body.decode("utf-8-sig")
    assert text.startswith("# Scenario: S")
    assert "Rossi Müller" in text
    # Et une seule fois : le BOM ne doit pas réapparaître au milieu du fichier.
    assert text.count("\ufeff") == 0


def test_every_format_carries_the_provenance_in_its_own_comment_syntax():
    from api.exports import render_export
    meta = {"scenario": "S", "scenario_id": "usr-1", "query": "q",
            "subset_label": "relevant", "n_articles": 1,
            "similarity_threshold": 0.4226, "rerank_threshold": 0.2,
            "coverage": "tous les pertinents"}
    # Une ligne COMPLÈTE : les formateurs lisent toutes les colonnes de l'export.
    from api.exports import EXPORT_COLUMNS
    rows = [{**{c: "" for c in EXPORT_COLUMNS},
             "rank": 1, "id": 9, "title": "A paper", "authors": "Rossi M",
             "url": "https://doi.org/10.1/a", "doi": "10.1/a", "year": 2025}]
    for fmt, marker in (("csv", "# Scenario: S"), ("md", "> Scenario: S"),
                        ("bibtex", "% Scenario: S"), ("ris", "N1  - Scenario: S")):
        out = render_export(fmt, rows, "T", meta).decode("utf-8-sig")
        assert marker in out, f"{fmt} ne porte pas sa provenance"
        assert "0.4226" in out, f"{fmt} ne porte pas le seuil qui a défini son lot"
    # Le JSON la porte dans son bloc `meta`, et le XLSX sur une feuille à part.
    import json as _json
    assert _json.loads(render_export("json", rows, "T", meta))["meta"]["similarity_threshold"] == 0.4226
    _x = render_export("xlsx", rows, "T", meta)
    assert _x[:2] == b"PK" and len(_x) > 3000


def test_a_threshold_of_zero_is_called_out_rather_than_labelled_relevant():
    """`?threshold=0` rend le corpus ENTIER. Le fichier doit le dire."""
    from api.exports import provenance_lines
    lines = " ".join(provenance_lines({"subset_label": "relevant-articles", "n_articles": 640,
                                       "similarity_threshold": 0}))
    assert "WHOLE corpus" in lines

