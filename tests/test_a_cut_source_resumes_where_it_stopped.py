"""Une source coupée par le budget reprend sa lecture, au lieu de la recommencer.

Une recherche exhaustive demande à chaque base TOUT ce qu'elle apparie. Sur la question
COVID exposition, Europe PMC en annonce 72 847 et OpenAlex 66 395 : cela ne tient pas dans
le budget d'une fédération (45 minutes), donc la source est coupée en route.

Jusqu'ici la reprise repartait du DÉBUT. Elle relisait les mêmes pages, le budget
s'épuisait au même endroit, et le quatrième tour sur ce scénario n'a rapporté AUCUNE notice
nouvelle : enchaîner les reprises ne convergeait pas, et « exhaustif » ne pouvait donc pas
se terminer au-delà d'une certaine taille. La position de lecture est désormais gardée par
scénario, par source et par requête.

Ces tests pinent les deux moitiés : la décision (garder ou oublier la position, selon
l'issue de la source) est pure ; le stockage a besoin d'une base.
"""
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from api import pipeline as P  # noqa: E402

SID = "usr-test-resume-0001"
OTHER = "usr-test-resume-0002"
QHASH = "a" * 40
QHASH2 = "b" * 40


# ── La décision, sans base ───────────────────────────────────────────────────────
def test_a_source_cut_in_the_middle_keeps_where_it_was():
    assert P.cursor_action("cut_by_budget", has_position=True, resumed=False) == "save"


def test_a_source_cut_before_reading_a_single_page_keeps_nothing():
    """Rien à reprendre : garder « * » ferait croire à une reprise qui n'avance pas."""
    assert P.cursor_action("cut_by_budget", has_position=False, resumed=False) is None


@pytest.mark.parametrize("outcome", ["ok", "empty"])
def test_a_source_that_read_to_the_end_forgets_its_position(outcome):
    """Sinon, relancer la même question repartirait du milieu de la base."""
    assert P.cursor_action(outcome, has_position=True, resumed=True) == "clear"


def test_a_resumed_source_that_could_not_read_a_page_forgets_its_position():
    """Un curseur peut périmer côté API : l'oublier fait repartir du début au tour
    suivant, au lieu de rester bloqué sur une position que la base refuse."""
    assert P.cursor_action("error", has_position=False, resumed=True) == "clear"


def test_an_error_without_a_resume_changes_nothing():
    assert P.cursor_action("error", has_position=False, resumed=False) is None


@pytest.mark.parametrize("outcome", ["cached", "skipped"])
def test_a_source_that_was_not_queried_changes_nothing(outcome):
    assert P.cursor_action(outcome, has_position=False, resumed=False) is None


# ── Le stockage, avec une base ───────────────────────────────────────────────────
_DB = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
pytestmark_db = pytest.mark.skipif(not _DB, reason="pas de base")


@pytest.fixture()
def clean_cursors():
    if not _DB:
        pytest.skip("pas de base")
    try:
        P._ensure_source_fetch_cursor()
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"base injoignable : {e}")
    for sid in (SID, OTHER):
        P._clear_fetch_cursor(sid)
    yield
    for sid in (SID, OTHER):
        P._clear_fetch_cursor(sid)


def test_the_position_comes_back_for_the_same_question(clean_cursors):
    P._save_fetch_cursor(SID, "_fetch_europepmc", QHASH, "AoIIP4AAACdQTUM=", 20398, 1)
    got = P._load_fetch_cursors(SID, QHASH)
    assert got["_fetch_europepmc"]["position"] == "AoIIP4AAACdQTUM="
    assert got["_fetch_europepmc"]["fetched"] == 20398
    assert got["_fetch_europepmc"]["rounds"] == 1


def test_a_position_belongs_to_one_question_and_one_scenario(clean_cursors):
    """Le curseur d'une base ne veut rien dire pour une autre question : changer la
    requête doit faire repartir du début, pas sauter au milieu d'une autre lecture."""
    P._save_fetch_cursor(SID, "_fetch_europepmc", QHASH, "cursor-1", 1000, 1)
    assert P._load_fetch_cursors(SID, QHASH2) == {}
    assert P._load_fetch_cursors(OTHER, QHASH) == {}


def test_a_later_round_replaces_the_position_and_counts_the_rounds(clean_cursors):
    P._save_fetch_cursor(SID, "_fetch_openalex", QHASH, "cursor-1", 1000, 1)
    P._save_fetch_cursor(SID, "_fetch_openalex", QHASH, "cursor-2", 2500, 2)
    got = P._load_fetch_cursors(SID, QHASH)["_fetch_openalex"]
    assert (got["position"], got["fetched"], got["rounds"]) == ("cursor-2", 2500, 2)


def test_forgetting_one_source_leaves_the_others(clean_cursors):
    P._save_fetch_cursor(SID, "_fetch_europepmc", QHASH, "c1", 10, 1)
    P._save_fetch_cursor(SID, "_fetch_openalex", QHASH, "c2", 20, 1)
    P._clear_fetch_cursor(SID, "_fetch_europepmc", QHASH)
    assert set(P._load_fetch_cursors(SID, QHASH)) == {"_fetch_openalex"}


def test_a_fresh_search_forgets_every_position_of_that_question(clean_cursors):
    """`force_live` veut dire « relis tout depuis le début » : sauter ce qui a déjà été
    lu serait l'inverse de ce qui est demandé."""
    P._save_fetch_cursor(SID, "_fetch_europepmc", QHASH, "c1", 10, 1)
    P._save_fetch_cursor(SID, "_fetch_openalex", QHASH, "c2", 20, 1)
    P._clear_fetch_cursor(SID, query_hash=QHASH)
    assert P._load_fetch_cursors(SID, QHASH) == {}


def test_a_position_longer_than_the_column_is_cut_rather_than_lost(clean_cursors):
    """Un curseur d'API peut être long ; mieux vaut une position tronquée et un tour
    reparti du début qu'une exception au moment d'enregistrer."""
    P._save_fetch_cursor(SID, "_fetch_openaire", QHASH, "x" * 5000, 5, 1)
    got = P._load_fetch_cursors(SID, QHASH)["_fetch_openaire"]
    assert len(got["position"]) == 2000
