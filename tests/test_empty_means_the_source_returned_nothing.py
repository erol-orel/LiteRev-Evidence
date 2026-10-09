"""`empty` se décidait sur les lignes INSÉRÉES, pas sur les notices RENVOYÉES.

Deux défauts que la relecture adversariale a trouvés dans mon propre correctif, celui qui
a introduit la table des six issues. Vérifiés ensuite à la main, avec la fonction de
l'application.

1. Le compteur `count` de chaque fetcher ne s'incrémente que `if _new`, c'est-à-dire sur
   une ligne NOUVELLE en base. Une source dont les 500 notices étaient déjà toutes dans la
   bibliothèque rendait donc 0, et l'issue calculée était `empty` : « cette base n'a rien
   sur le sujet ». La carte affichait « OPENALEX 500 · aucun résultat », un compte et son
   démenti côte à côte. C'est exactement le mensonge que la table des issues existe pour
   empêcher, dans la table elle-même.

2. Deux fetchers sur douze écrivent leurs enregistrements sous une clé qui n'est pas leur
   nom : `_fetch_preprints` sous « preprint » au singulier, et `_fetch_biorxiv_medrxiv`
   sous « biorxiv » et « medrxiv » séparément. Le tableau portait donc une ligne fantôme à
   zéro juste à côté de la vraie, et l'issue se décidait sur un compteur qui n'était pas
   celui de la source.
"""
import inspect

import pytest

pytest.importorskip("fastapi")

from api.pipeline import _run_user_scenario_populate  # noqa: E402
from api.search import (  # noqa: E402
    SOURCE_RECORD_KEYS,
    _prisma_identification_figures,
    source_record_keys,
)

SRC = inspect.getsource(_run_user_scenario_populate)


# ── L'issue se décide sur ce que la source a renvoyé ─────────────────────────

def test_the_outcome_is_not_decided_on_inserted_rows():
    """`_returned` porte le compteur du fetcher, qui ne compte que le NOUVEAU."""
    _i = SRC.index('_outcome = "ok" if')
    decision = SRC[_i - 700:_i + 120]
    assert "_ident_records.get(" in decision, (
        "l'issue se décide encore sur `_returned`, donc sur les lignes insérées : une "
        "source dont tout le lot était déjà en base se lira « aucun résultat »")
    assert "_returned[_n] > 0" not in SRC, (
        "le test sur les lignes insérées est encore là")


def test_the_record_count_is_read_under_the_lock():
    _i = SRC.index("_ident_records.get(")
    assert "with _counter_lock:" in SRC[_i - 200:_i], (
        "`_ident_records` est muté par les fils des fetchers : la lecture doit prendre "
        "le verrou comme les autres")


def test_a_source_whose_whole_batch_was_already_known_is_not_empty():
    """Le cas réel : 500 notices renvoyées, zéro ligne nouvelle."""
    f = _prisma_identification_figures(
        {"openalex": 500, "pubmed": 30}, 530, 0, 530,
        source_outcomes={"_fetch_openalex": "ok", "_fetch_pubmed": "ok"})
    assert f["records_by_source"]["openalex"] == 500
    assert f["source_outcomes"]["openalex"] == "ok"
    # Et le contraire doit rester possible : une source vraiment vide garde sa ligne.
    g = _prisma_identification_figures(
        {"pubmed": 30}, 30, 0, 30,
        source_outcomes={"_fetch_pubmed": "ok", "_fetch_doaj": "empty"})
    assert g["records_by_source"]["doaj"] == 0
    assert g["source_outcomes"]["doaj"] == "empty"


def test_no_outcome_contradicts_its_own_count():
    """La règle, énoncée : `empty` et un compte positif ne peuvent pas coexister."""
    f = _prisma_identification_figures(
        {"preprint": 5, "biorxiv": 2, "medrxiv": 1, "pubmed": 30, "crossref": 0},
        38, 0, 38,
        source_outcomes={"_fetch_preprints": "ok", "_fetch_biorxiv_medrxiv": "ok",
                         "_fetch_pubmed": "ok", "_fetch_crossref": "empty"})
    for src, outcome in f["source_outcomes"].items():
        count = sum(f["records_by_source"].get(k, 0) for k in source_record_keys(src))
        if outcome == "empty":
            assert count == 0, f"{src} est dite vide avec {count} enregistrements"
        if outcome == "ok":
            assert count > 0, f"{src} est dite « ok » avec {count} enregistrements"


# ── Les clés d'enregistrement ────────────────────────────────────────────────

def test_the_two_fetchers_that_write_under_another_name_are_mapped():
    assert source_record_keys("_fetch_preprints") == ("preprint",)
    assert source_record_keys("_fetch_biorxiv_medrxiv") == ("biorxiv", "medrxiv")
    # Les dix autres écrivent sous leur propre nom.
    for n in ("pubmed", "openalex", "crossref", "europepmc", "semantic_scholar",
              "doaj", "clinicaltrials", "core", "arxiv", "openaire"):
        assert source_record_keys(n) == (n,)
        assert source_record_keys(f"_fetch_{n}") == (n,)


def test_the_mapping_matches_what_the_fetchers_actually_write():
    """Épingle la correspondance contre le CODE, pour qu'un fetcher renommé la casse ici.

    `_fetch_preprints` passe `source="preprint"`, et `_fetch_biorxiv_medrxiv` passe
    `_server`, qui vaut « biorxiv » ou « medrxiv »."""
    assert 'source="preprint"' in SRC, (
        "le fetcher préprints n'écrit plus sous « preprint » : la correspondance "
        "SOURCE_RECORD_KEYS doit suivre")
    assert "_ingest_parsed(_server, _parse_biorxiv(" in SRC, (
        "le fetcher bioRxiv/medRxiv n'écrit plus sous le nom du serveur")
    assert set(SOURCE_RECORD_KEYS) == {"preprints", "biorxiv_medrxiv"}, (
        "une correspondance a été ajoutée ou retirée sans que ce test le dise")


def test_no_phantom_zero_row_beside_the_real_one():
    f = _prisma_identification_figures(
        {"preprint": 5, "biorxiv": 2, "medrxiv": 1}, 8, 0, 8,
        source_outcomes={"_fetch_preprints": "ok", "_fetch_biorxiv_medrxiv": "ok"})
    assert "preprints" not in f["records_by_source"], (
        "« preprint 5 » et « preprints 0 » se lisaient comme deux sources dont l'une "
        "n'avait rien trouvé")
    assert "biorxiv_medrxiv" not in f["records_by_source"]
    assert f["records_identified"] == 8


def test_a_source_with_no_records_at_all_still_gets_its_row():
    """Le correctif ne doit pas faire disparaître la ligne qui dit « cette source a été
    lancée et n'a rien rapporté » : c'est la seule façon de lire un échec."""
    f = _prisma_identification_figures(
        {"pubmed": 30}, 30, 0, 30,
        source_outcomes={"_fetch_pubmed": "ok", "_fetch_openalex": "error",
                         "_fetch_preprints": "empty", "_fetch_biorxiv_medrxiv": "empty"})
    for name in ("openalex", "preprints", "biorxiv_medrxiv"):
        assert name in f["records_by_source"], f"{name} a perdu sa ligne"
        assert f["records_by_source"][name] == 0
