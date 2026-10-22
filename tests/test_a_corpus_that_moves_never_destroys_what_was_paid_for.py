"""One article finishing its scoring must not destroy a dozen model calls.

Measured on production, twice in four days, on two of the partners' reviews. The last
unscored article of the HPAI susceptibility review received its similarity score (15 033
of 15 034 became 15 034 of 15 034). The scoring pass correctly concluded that the relevant
subset might have moved, and called what was then `reset_corpus_derived_caches`, which
NULLED the clustering, the similarity graph, the concept map and the recommended actions
in one statement. The clustering alone is a written summary per group, a dozen model
calls. The Visualization tab went blank, nothing said why, and the same thing happened
again three days later on the COVID exposure review, which lost its corpus portrait too.

The rule now picks by price, and these tests pin it:

  - what costs nothing is DROPPED (the portrait, the similarity graph, the SEIR
    projection): their own tab recomputes them on finding them absent, so dropping them
    is how they refresh;
  - what was paid for is KEPT and DATED (the clustering's summaries, the concept map's
    annotation pass, the recommended actions);
  - and the third thing, the one that makes keeping them honest, never happens: a kept
    artefact is served with its own date and with the date the corpus last moved, so no
    tab can present the previous corpus's answer as the current one's.

`corpus_moved_at` exists because none of the dates the readiness panel already watched
moves when a score is written: not the link's, not the decision's, not the threshold's.
"""
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.scenario_store import (  # noqa: E402
    FREE_CACHE_RESET,
    corpus_moved_at,
    is_stale,
    mark_corpus_moved,
)
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-corpus-moves"
IDS = (9771, 9772)

#: Ce que le clustering coûte à refaire, en une phrase : c'est tout l'enjeu.
PAID = ("clustering_json", "concept_graph_json", "recommended_actions_json")
FREE = ("knowledge_graph_json", "seir_projection_json", "evidence_brief_json")


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:                                        # noqa: BLE001
        return False


@pytest.fixture()
def seeded(db_conn):
    """Un scénario dont les six artefacts sont là, datés d'hier."""
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created = ensure_document_columns(cur)
        # Toutes les colonnes que le panneau de complétude lit : sans elles sa lecture
        # échoue et il répond « non mesuré », ce qui ferait passer ce fichier pour vert
        # ou rouge selon l'ordre des tests.
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("country", "TEXT"),
                         ("has_fulltext", "BOOLEAN"), ("pico_json", "JSONB"),
                         ("extraction_json", "JSONB"), ("concepts_json", "JSONB"),
                         ("epi_params_json", "JSONB"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("similarity_score", "DOUBLE PRECISION"),
                         ("rerank_score", "DOUBLE PRECISION"),
                         ("assigned_at", "TIMESTAMP"), ("screened_at", "TIMESTAMP")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Moves', 'avian influenza', 'boolean', '{}', TRUE)", (SID,))
        cur.execute("""
            INSERT INTO literature_document (id, title, abstract, source, is_duplicate,
                                             project_context)
            VALUES (9771, 'One', 'A cross-sectional study of exposure among poultry cullers.',
                    'pubmed', FALSE, 'literev'),
                   (9772, 'Two', 'A cohort study of exposure at live bird markets, by stall.',
                    'pubmed', FALSE, 'literev')
        """)
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) "
                    "VALUES (%s, 9771, 0.9), (%s, 9772, 0.8)", (SID, SID))
        cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) "
                    "VALUES (%s, 0.45) ON CONFLICT (scenario_id) DO UPDATE "
                    "SET similarity_threshold = 0.45", (SID,))
        sets = ", ".join(
            f"{c} = %(j)s::jsonb" for c in PAID + FREE) + ", " + ", ".join(
            f"{a} = NOW() - INTERVAL '1 day'" for a in (
                "clustering_generated_at", "concept_graph_generated_at", "actions_generated_at",
                "kg_generated_at", "seir_projection_generated_at", "evidence_brief_at"))
        cur.execute(f"UPDATE scenario_settings SET {sets} WHERE scenario_id = %(s)s",
                    {"j": json.dumps({"built": "yesterday"}), "s": SID})
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        if created:
            cur.execute("DROP TABLE document_chunk")


def _columns(conn, *names):
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(names)} FROM scenario_settings WHERE scenario_id = %s",
                    (SID,))
        return dict(zip(names, cur.fetchone()))


# ── The headline ────────────────────────────────────────────────────────────
def test_a_corpus_that_moves_keeps_every_paid_artefact(seeded):
    """THE test. One article receiving its score used to take the clustering with it."""
    with main.engine.begin() as conn:
        mark_corpus_moved(conn, SID)
    got = _columns(seeded, *PAID)
    for col in PAID:
        assert got[col] is not None, f"{col} was destroyed by a corpus move"


def test_a_corpus_that_moves_drops_what_costs_nothing(seeded):
    """And the other half of the rule: the free ones go, because their tab rebuilds them
    on finding them absent. Keeping them would freeze a portrait of the old corpus."""
    with main.engine.begin() as conn:
        mark_corpus_moved(conn, SID)
    got = _columns(seeded, *FREE)
    for col in FREE:
        assert got[col] is None, f"{col} costs nothing: a move should refresh it"


def test_the_move_is_dated_so_a_kept_artefact_can_be_called_old(seeded):
    """Without this date nothing could tell: a scoring pass moves no other date in the
    database, so a clustering built before it would be announced as current."""
    before = _columns(seeded, "corpus_moved_at")["corpus_moved_at"]
    assert before is None
    with main.engine.begin() as conn:
        mark_corpus_moved(conn, SID)
        after = corpus_moved_at(conn, SID)
    assert after is not None
    built_yesterday = _columns(seeded, "clustering_generated_at")["clustering_generated_at"]
    assert is_stale(built_yesterday, after), "the kept clustering does not read as old"


def test_a_scenario_with_no_settings_row_still_records_the_move(seeded):
    """A scenario nobody has configured has no row. Without the INSERT the move would be
    lost, and the first artefact built afterwards would look older than a corpus that had
    apparently never moved."""
    with seeded.cursor() as cur:
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
    with main.engine.begin() as conn:
        mark_corpus_moved(conn, SID)
        assert corpus_moved_at(conn, SID) is not None


# ── Honesty: a kept artefact never passes for a current one ─────────────────
def test_the_clustering_is_served_with_both_dates(seeded):
    """Keeping it is only honest if the tab says what it is. `_load_viz_cache` is the one
    reader of the three visualisations, so the two dates ride on all of them."""
    from api.clustering import _load_viz_cache

    with main.engine.begin() as conn:
        mark_corpus_moved(conn, SID)
    got = _load_viz_cache(SID, "clustering")
    assert got is not None, "the clustering was dropped, not kept"
    assert got["stale"] is True
    assert got["built_at"] and got["corpus_moved_at"]


def test_a_clustering_built_after_the_move_is_not_stale(seeded):
    with main.engine.begin() as conn:
        mark_corpus_moved(conn, SID)
    with seeded.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET clustering_generated_at = NOW() "
                    "WHERE scenario_id = %s", (SID,))
    from api.clustering import _load_viz_cache

    assert _load_viz_cache(SID, "clustering")["stale"] is False


def test_the_actions_are_served_with_both_dates(seeded):
    """Same for the recommended actions, whose cache is keyed by language and was the
    artefact this whole mechanism was written for in the first place."""
    from fastapi.testclient import TestClient

    with seeded.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET recommended_actions_json = %s::jsonb, "
                    "recommended_actions_lang = 'en' WHERE scenario_id = %s",
                    (json.dumps({"en": ["Watch live bird markets."]}), SID))
    with main.engine.begin() as conn:
        mark_corpus_moved(conn, SID)
    r = TestClient(main.app).get(f"/scenarios/{SID}/recommended-actions", params={"lang": "en"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ready" and body["actions"], "the actions were destroyed"
    assert body["stale"] is True and body["corpus_moved_at"]


# ── The readiness panel agrees with what the tabs serve ─────────────────────
def test_the_panel_calls_a_kept_artefact_old_rather_than_missing(seeded):
    """Before, it said "missing" because the payload was gone, which sent a reviewer to
    pay for work that had been done. Now it says how old it is."""
    from api.outputs import STALE, scenario_outputs

    with main.engine.begin() as conn:
        mark_corpus_moved(conn, SID)
    o = {r["key"]: r for r in scenario_outputs(SID, lang="en")["outputs"]}
    assert o["clustering"]["state"] == STALE
    assert o["clustering"]["why"] == "corpus_moved"
    assert o["clustering"]["generated_at"], "and it says when it was built"
    # The free ones are missing, which is correct: they really are gone, and free.
    assert o["knowledge_graph"]["state"] == "missing"


def test_the_free_reset_lists_only_free_outputs():
    """The two lists are the same classification the readiness contract already declares,
    so a result added tomorrow cannot be filed one way here and another way there."""
    from api.outputs import FREE, OUTPUTS

    by_col = {o["column"]: o for o in OUTPUTS if o["shape"] == "artefact"}
    for column in FREE_CACHE_RESET.split():
        col = column.strip(",")
        if col in by_col:
            assert by_col[col]["cost"] == FREE, f"{col} is dropped but is not free"


# ── Every path that moves a corpus dates it, the search included ────────────
def test_the_search_dates_the_corpus_it_changed():
    """The hole this rule had, found live by its own panel on the partners' data.

    A search freezes the corpus with `_set_scenario_corpus` and, alone among the paths
    that move it, did not date it. Measured on the COVID exposure review: the resume
    carried the corpus from 86,716 to 112,182 links while `corpus_moved_at` still read the
    previous day's scoring pass. An artefact rebuilt between that search and the next
    scoring therefore reported itself current while describing a corpus 25,000 articles
    smaller, which is precisely what the dating exists to prevent.

    Read off the source rather than by running a federation: the assertion is that the
    freeze and the dating sit together, which is the thing that was missing."""
    import inspect

    from api import pipeline as P

    src = inspect.getsource(P._run_user_scenario_populate)
    assert "_set_scenario_corpus(" in src, "the fixture reads the right function"
    assert "mark_corpus_moved(" in src, (
        "a search that changes the corpus must date it, like every other path that moves it")
    # And it must be conditional: a replay that changes nothing dates nothing, or it would
    # drop the free caches on every re-run for no reason.
    assert "_previous_links" in src and "!= set(_previous_links)" in src, (
        "the dating must compare the membership before and after, not fire unconditionally")


def test_every_path_that_moves_a_corpus_goes_through_the_one_function():
    """The list, so a path added tomorrow cannot date the corpus one way here and another
    way there. Nine now: the search is the ninth."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    callers = {
        p.name
        for p in list(root.glob("api/*.py")) + [root / "living_review_scheduler.py"]
        if p.exists() and "mark_corpus_moved(" in p.read_text(encoding="utf-8")
        and p.name != "scenario_store.py"            # where it is defined
    }
    assert "pipeline.py" in callers, "the search is one of them"
    for expected in ("relevance.py", "double_blind.py", "subsets.py",
                     "living_review_scheduler.py"):
        assert expected in callers, f"{expected} stopped dating the corpus"
