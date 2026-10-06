"""Narrowing the corpus by cluster or by concept, without inventing a second corpus.

The decision this module makes is a screening decision, so it is written where screening
is written: `screening_status = 'excluded'` on the scenario's own link row. Everything
downstream already reads through that, which is the whole reason for the choice, and these
tests check that it is true rather than assumed: the relevance gate shrinks, the threshold
curve shrinks with it, PRISMA shows the reason, and the same document in ANOTHER scenario
is untouched.

The hazard the pure tests exist for: the clustering is a projection capped at
CLUSTER_MAX_DOCS, and the concept map only covers articles with extracted concepts. "Keep
clusters 3 and 7" therefore says NOTHING about the articles those projections never saw,
and throwing them out on that basis would silently shrink a corpus by thousands. They are
kept unless the caller asks otherwise, and they are counted on their own line either way.
"""
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.subsets import (  # noqa: E402
    SUBSET_REASON_PREFIX,
    describe_selection,
    plan_subset,
    scope_threshold_of,
)
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-subset-test"
OTHER = "usr-subset-other"
IDS = (9601, 9602, 9603, 9604, 9605, 9606)


# ── Pure: who is kept, who goes, and who could not be judged ────────────────
def test_an_article_the_clustering_never_saw_is_kept_not_thrown_out():
    """The clustering stops at CLUSTER_MAX_DOCS. An article past that cap has no cluster,
    which is not the same as "not in the clusters you chose": the projection has no
    opinion about it. Excluding it would shrink a corpus by thousands on the strength of a
    computation that never looked at them."""
    plan = plan_subset([1, 2, 3, 4, 5], cluster_of={1: 0, 2: 0, 3: 1, 4: 1}, clusters_wanted={0})
    assert plan["keep_ids"] == [1, 2, 5]          # 5 was never clustered, and stays
    assert plan["exclude_ids"] == [3, 4]
    assert plan["undecided_ids"] == [5]
    assert plan["by_dimension"]["clusters"] == {"in": 2, "out": 2, "unknown": 1}


def test_the_unjudged_go_only_when_the_caller_says_so():
    plan = plan_subset([1, 2, 3, 4, 5], cluster_of={1: 0, 2: 0, 3: 1, 4: 1},
                       clusters_wanted={0}, unassigned="exclude")
    assert plan["exclude_ids"] == [3, 4, 5]
    assert plan["undecided"] == 1                 # still reported, not absorbed silently


def test_a_dimension_that_cannot_judge_neither_saves_nor_condemns():
    """Article 1 is in the chosen cluster and has no extracted concepts. Under
    `combine='all'` the concept dimension abstains rather than vetoing, so the article
    survives on the cluster's verdict alone."""
    plan = plan_subset([1, 2], cluster_of={1: 0, 2: 0}, clusters_wanted={0},
                       concept_match={2}, concept_known={2}, combine="all")
    assert plan["keep_ids"] == [1, 2]
    assert plan["exclude_ids"] == []
    # ... and an article the concepts DO judge, and reject, still goes.
    plan = plan_subset([1, 2], cluster_of={1: 0, 2: 0}, clusters_wanted={0},
                       concept_match=set(), concept_known={2}, combine="all")
    assert plan["exclude_ids"] == [2] and plan["keep_ids"] == [1]


def test_combine_all_is_an_intersection_and_combine_any_a_union():
    kw = dict(cluster_of={1: 0, 2: 1, 3: 0, 4: 1}, clusters_wanted={0},
              concept_match={1, 2}, concept_known={1, 2, 3, 4})
    assert plan_subset([1, 2, 3, 4], combine="all", **kw)["keep_ids"] == [1]
    assert plan_subset([1, 2, 3, 4], combine="any", **kw)["keep_ids"] == [1, 2, 3]


def test_an_article_no_dimension_can_judge_is_undecided_not_excluded():
    plan = plan_subset([7], cluster_of={}, clusters_wanted={0},
                       concept_match=set(), concept_known=set())
    assert plan["undecided_ids"] == [7] and plan["exclude_ids"] == []
    assert plan["by_dimension"] == {"clusters": {"in": 0, "out": 0, "unknown": 1},
                                    "concepts": {"in": 0, "out": 0, "unknown": 1}}


def test_a_selection_with_no_dimension_is_refused():
    with pytest.raises(ValueError):
        plan_subset([1, 2])
    with pytest.raises(ValueError):
        plan_subset([1], cluster_of={}, clusters_wanted=set(), combine="sometimes")


def test_the_stored_reason_names_the_clusters_rather_than_numbering_them():
    """Cluster ids are positions in one computation and change at the next recompute, so a
    reason reading "scope: not in clusters 3, 7" would be unreadable a week later, and
    wrong. The name goes in, and the string is what PRISMA will show."""
    reason = describe_selection([2], {2: "Vector control"}, ["dengue virus"], "all", "keep")
    assert reason.startswith(SUBSET_REASON_PREFIX)
    assert "Vector control" in reason and "dengue virus" in reason
    assert "cluster 2" not in reason
    assert "hors projection" in describe_selection([2], {2: "V"}, None, "all", "exclude")


def test_the_reason_records_the_threshold_the_narrowing_was_applied_at():
    """A narrowing only ever judged the articles relevant AT THAT MOMENT. Lowering the
    threshold afterwards brings in articles it never saw, and with the threshold absent
    from the record nothing anywhere could say where that boundary is: not PRISMA, not the
    threshold curve, not the person rereading it next month."""
    reason = describe_selection([0], {0: "Vector control"}, None, "all", "keep", threshold=0.6801)
    assert reason.endswith("(seuil 0.6801)")
    assert scope_threshold_of(reason) == 0.6801
    # A reviewer's own free-text reason carries no threshold, and that is not an error.
    assert scope_threshold_of("wrong population") is None
    assert scope_threshold_of(None) is None
    assert scope_threshold_of("scope: hors de clusters X") is None


# ── Integration: the decision lands where the rest of the app reads ─────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


def _clustering_cache(groups: dict[int, list[int]], names: dict[int, str]) -> str:
    return json.dumps({
        "n_docs": sum(len(v) for v in groups.values()),
        "n_docs_total": sum(len(v) for v in groups.values()),
        "n_clusters": len(groups),
        "method": "test",
        "lang": "en",
        "clusters": [{"cluster_id": cid, "cluster_name": names[cid], "is_noise": cid == -1,
                      "n_docs": len(ids), "top_words": [], "summary": "",
                      "points": [{"id": i, "x": 0.0, "y": 0.0} for i in ids]}
                     for cid, ids in groups.items()],
    })


@pytest.fixture()
def seeded(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created_chunk_table = ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("similarity_score", "DOUBLE PRECISION"), ("screening_reason", "TEXT"),
                         ("screening_notes", "TEXT"), ("screened_at", "TIMESTAMP")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        # PRISMA counts embedded chunks. Where pgvector is absent the column is missing
        # entirely, and the query only tests it for NULL, so a plain text column does.
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding TEXT")
        for sid in (SID, OTHER):
            cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        for sid, name in ((SID, "Subset"), (OTHER, "Another scenario")):
            cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                        "VALUES (%s, %s, 'dengue', 'boolean', '{}', TRUE)", (sid, name))
        cur.execute(
            "INSERT INTO literature_document (id, title, abstract, source, is_duplicate, project_context) VALUES "
            "(9601, 'Transmission A', 'a study of transmission', 'pubmed', false, 'literev'),"
            "(9602, 'Transmission B', 'another transmission study', 'pubmed', false, 'literev'),"
            "(9603, 'Vector A', 'a study of vector control', 'pubmed', false, 'literev'),"
            "(9604, 'Vector B', 'another vector study', 'pubmed', false, 'literev'),"
            "(9605, 'Never clustered', 'past the projection cap', 'pubmed', false, 'literev'),"
            "(9606, 'Reviewer excluded', 'thrown out by hand, for its own reason', 'pubmed', false, 'literev')")
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, screening_status, screening_reason) VALUES "
                    "(%s, 9601, 0.90, NULL, NULL), (%s, 9602, 0.85, NULL, NULL),"
                    "(%s, 9603, 0.80, NULL, NULL), (%s, 9604, 0.75, NULL, NULL),"
                    "(%s, 9605, 0.70, NULL, NULL),"
                    "(%s, 9606, 0.95, 'excluded', 'wrong population')", (SID,) * 6)
        # The SAME document also belongs to another scenario, untouched by any of this.
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) "
                    "VALUES (%s, 9603, 0.88)", (OTHER,))
        for sid in (SID, OTHER):
            cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) VALUES (%s, 0.45) "
                        "ON CONFLICT (scenario_id) DO UPDATE SET similarity_threshold = 0.45", (sid,))
        # Clusters: 0 = transmission (9601, 9602), 1 = vectors (9603, 9604).
        # 9605 is in NO cluster: the projection stopped before it.
        cur.execute("UPDATE scenario_settings SET clustering_json = %s::jsonb, "
                    "clustering_generated_at = NOW() WHERE scenario_id = %s",
                    (_clustering_cache({0: [9601, 9602], 1: [9603, 9604]},
                                       {0: "Transmission", 1: "Vector control"}), SID))
    yield db_conn
    with db_conn.cursor() as cur:
        for sid in (SID, OTHER):
            cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


HDR = {"X-API-Key": "test-write-key"}


def test_a_preview_writes_nothing(seeded):
    body = _client().post(f"/user-scenarios/{SID}/subset/preview",
                          json={"clusters": [0]}).json()
    assert body["keep"] == 3 and body["exclude"] == 2 and body["undecided"] == 1
    assert body["exclude_sample"] == [9603, 9604]
    assert body["undecided_sample"] == [9605]     # never clustered, kept
    with seeded.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM article_scenarios WHERE scenario_id = %s "
                    "AND screening_status = 'excluded'", (SID,))
        assert cur.fetchone()[0] == 1             # only the reviewer's own, from the fixture


def test_applying_excludes_exactly_what_the_preview_announced(seeded):
    out = _client().post(f"/user-scenarios/{SID}/subset/apply",
                         json={"clusters": [0]}, headers=HDR).json()
    assert out["status"] == "applied" and out["applied"] == 2
    assert out["reason"].startswith(SUBSET_REASON_PREFIX) and "Transmission" in out["reason"]
    with seeded.cursor() as cur:
        cur.execute("SELECT document_id FROM article_scenarios WHERE scenario_id = %s "
                    "AND screening_status = 'excluded' AND screening_reason LIKE %s "
                    "ORDER BY document_id", (SID, SUBSET_REASON_PREFIX + "%"))
        assert [r[0] for r in cur.fetchall()] == [9603, 9604]


def test_narrowing_one_scenario_leaves_the_same_document_alone_in_another(seeded):
    """THE reason this writes to article_scenarios and never to the document row: 9603 also
    belongs to another scenario. Restricting the scope of one review must not delete an
    article from someone else's."""
    _client().post(f"/user-scenarios/{SID}/subset/apply", json={"clusters": [0]}, headers=HDR)
    with seeded.cursor() as cur:
        cur.execute("SELECT screening_status FROM article_scenarios "
                    "WHERE scenario_id = %s AND document_id = 9603", (OTHER,))
        assert cur.fetchone()[0] is None
        cur.execute("SELECT screening_status FROM literature_document WHERE id = 9603")
        assert cur.fetchone()[0] is None


def test_the_rest_of_the_app_sees_the_narrowed_corpus(seeded):
    """The point of writing it as a screening decision: nothing else had to be taught
    about it. The full-corpus digest and the threshold curve both shrink on their own,
    because both already read through the shared relevance gate."""
    before = main.corpus_digest(SID, 0.45)["n_articles"]
    curve_before = _client().get(f"/scenarios/{SID}/threshold-curve").json()
    _client().post(f"/user-scenarios/{SID}/subset/apply", json={"clusters": [0]}, headers=HDR)
    after = main.corpus_digest(SID, 0.45)["n_articles"]
    curve_after = _client().get(f"/scenarios/{SID}/threshold-curve").json()
    assert before == 5 and after == 3
    assert curve_before["candidates"] == 5 and curve_after["candidates"] == 3


def test_prisma_shows_the_reason_so_a_methods_section_can_state_it(seeded):
    _client().post(f"/user-scenarios/{SID}/subset/apply", json={"clusters": [0]}, headers=HDR)
    prisma = _client().get(f"/user-scenarios/{SID}/prisma").json()
    reasons = {r["reason"]: r["articles"] for r in prisma["manual_curation"]["excluded_by_reason"]}
    assert any(k.startswith(SUBSET_REASON_PREFIX) and v == 2 for k, v in reasons.items())
    assert reasons.get("wrong population") == 1      # the reviewer's own, still its own line
    assert prisma["manual_curation"]["excluded"] == 3


def test_undo_restores_the_scope_exclusions_and_only_those(seeded):
    """A reviewer's per-article exclusions are hours of work and carry their own reasons.
    Undoing a scope narrowing must not touch them."""
    _client().post(f"/user-scenarios/{SID}/subset/apply", json={"clusters": [0]}, headers=HDR)
    out = _client().post(f"/user-scenarios/{SID}/subset/undo", headers=HDR).json()
    assert out["restored"] == 2
    with seeded.cursor() as cur:
        cur.execute("SELECT document_id, screening_reason FROM article_scenarios "
                    "WHERE scenario_id = %s AND screening_status = 'excluded'", (SID,))
        assert cur.fetchall() == [(9606, "wrong population")]
    assert main.corpus_digest(SID, 0.45)["n_articles"] == 5


def test_the_narrowing_in_force_is_readable_instead_of_invisible(seeded):
    assert _client().get(f"/user-scenarios/{SID}/subset").json()["narrowed"] is False
    _client().post(f"/user-scenarios/{SID}/subset/apply", json={"clusters": [0]}, headers=HDR)
    state = _client().get(f"/user-scenarios/{SID}/subset").json()
    assert state["narrowed"] is True and state["excluded_by_scope"] == 2
    assert len(state["steps"]) == 1 and state["steps"][0]["articles"] == 2
    assert "Transmission" in state["steps"][0]["reason"]
    assert state["steps"][0]["applied_at_threshold"] == 0.45
    assert state["judged_above_threshold"] == 0.45


def test_the_corpus_reports_what_the_analyses_read_not_only_what_scores_high(seeded):
    """"N above the threshold" counts SCORES. Once a narrowing can be in force, articles
    above the threshold can be out of scope, and the badge said 291 while the extractions
    read 190.

    The count is taken from the shared gate, never derived as "above the threshold minus
    the excluded": that subtraction drops the articles a reviewer rescued BELOW the
    threshold, which do feed the extractions. On a real corpus it gave 178 for 190."""
    before = _client().get(f"/user-scenarios/{SID}/corpus").json()
    assert before["relevant"] == main.corpus_digest(SID, 0.45)["n_articles"] == 5
    _client().post(f"/user-scenarios/{SID}/subset/apply", json={"clusters": [0]}, headers=HDR)
    after = _client().get(f"/user-scenarios/{SID}/corpus").json()
    assert after["above_threshold"] == before["above_threshold"]      # the scores did not move
    assert after["relevant"] == 3 == main.corpus_digest(SID, 0.45)["n_articles"]
    # Now the case where the subtraction really bites: rescue 9605 by hand, BELOW the
    # threshold. It feeds the extractions and no arithmetic over above-threshold counts
    # can see it.
    with seeded.cursor() as cur:
        cur.execute("UPDATE article_scenarios SET similarity_score = 0.10, "
                    "screening_status = 'included' WHERE scenario_id = %s AND document_id = 9605",
                    (SID,))
    r = _client().get(f"/user-scenarios/{SID}/corpus").json()
    naive = r["above_threshold"] - 3                  # 9603, 9604 narrowed, 9606 by the reviewer
    assert naive == 2
    assert r["relevant"] == 3 == main.corpus_digest(SID, 0.45)["n_articles"] > naive


def test_the_threshold_curve_warns_that_a_narrowing_judged_only_part_of_the_corpus(seeded):
    """The trap a real walkthrough exposed. After narrowing by cluster at threshold 0.45,
    the curve happily offers thresholds BELOW it, which bring back articles the narrowing
    never judged, with nothing to say so. The curve now carries the boundary."""
    assert _client().get(f"/scenarios/{SID}/threshold-curve").json()["scope"] is None
    _client().post(f"/user-scenarios/{SID}/subset/apply", json={"clusters": [0]}, headers=HDR)
    scope = _client().get(f"/scenarios/{SID}/threshold-curve").json()["scope"]
    assert scope == {"excluded_by_scope": 2, "judged_above_threshold": 0.45}
    # Undoing takes the warning away with the narrowing.
    _client().post(f"/user-scenarios/{SID}/subset/undo", headers=HDR)
    assert _client().get(f"/scenarios/{SID}/threshold-curve").json()["scope"] is None


def test_a_selection_that_changes_nothing_says_so_instead_of_writing(seeded):
    out = _client().post(f"/user-scenarios/{SID}/subset/apply",
                         json={"clusters": [0, 1]}, headers=HDR).json()
    assert out["status"] == "noop" and out["applied"] == 0
    assert _client().get(f"/user-scenarios/{SID}/subset").json()["narrowed"] is False


def _recluster(conn, groups, names):
    """What recomputing the clustering does, which a second narrowing by cluster REQUIRES:
    applying drops the cached projection, because it described the previous corpus."""
    with conn.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET clustering_json = %s::jsonb, "
                    "clustering_generated_at = NOW() WHERE scenario_id = %s",
                    (_clustering_cache(groups, names), SID))


def test_a_second_narrowing_by_cluster_needs_the_clustering_recomputed_first(seeded):
    """Applying drops the clustering cache on purpose, so the follow-up narrowing cannot
    be computed from a projection of the corpus that no longer exists. It says so rather
    than reusing it."""
    _client().post(f"/user-scenarios/{SID}/subset/apply",
                   json={"clusters": [0, 1], "unassigned": "exclude"}, headers=HDR)
    again = _client().post(f"/user-scenarios/{SID}/subset/apply",
                           json={"clusters": [0]}, headers=HDR)
    assert again.status_code == 404 and "clustering" in again.json()["detail"].lower()


def test_narrowings_compound_and_the_state_lists_each_step(seeded):
    """Applying twice narrows further rather than replacing, so the state endpoint has to
    carry both steps: undo is how you broaden, and nobody should have to remember what
    they already applied."""
    first = _client().post(f"/user-scenarios/{SID}/subset/apply",
                           json={"clusters": [0, 1], "unassigned": "exclude"}, headers=HDR).json()
    assert first["applied"] == 1                       # the never-clustered 9605
    _recluster(seeded, {0: [9601, 9602], 1: [9603, 9604]},
               {0: "Transmission", 1: "Vector control"})
    second = _client().post(f"/user-scenarios/{SID}/subset/apply",
                            json={"clusters": [0]}, headers=HDR).json()
    assert second["applied"] == 2                      # and now the vector cluster
    state = _client().get(f"/user-scenarios/{SID}/subset").json()
    assert state["excluded_by_scope"] == 3
    assert len(state["steps"]) == 2
    # Undoing without naming a step takes all of them back.
    assert _client().post(f"/user-scenarios/{SID}/subset/undo", headers=HDR).json()["restored"] == 3
    assert main.corpus_digest(SID, 0.45)["n_articles"] == 5


def test_one_step_can_be_undone_by_its_reason(seeded):
    _client().post(f"/user-scenarios/{SID}/subset/apply",
                   json={"clusters": [0, 1], "unassigned": "exclude"}, headers=HDR)
    _recluster(seeded, {0: [9601, 9602], 1: [9603, 9604]},
               {0: "Transmission", 1: "Vector control"})
    _client().post(f"/user-scenarios/{SID}/subset/apply", json={"clusters": [0]}, headers=HDR)
    steps = _client().get(f"/user-scenarios/{SID}/subset").json()["steps"]
    target = next(s for s in steps if s["articles"] == 2)
    out = _client().post(f"/user-scenarios/{SID}/subset/undo",
                         params={"reason": target["reason"]}, headers=HDR).json()
    assert out["restored"] == 2
    assert _client().get(f"/user-scenarios/{SID}/subset").json()["excluded_by_scope"] == 1


def test_a_selection_that_would_empty_the_corpus_is_refused(seeded):
    """A corpus of zero is not a narrowing, it is a mistake, and every extraction
    downstream would then describe nothing while still looking like it worked."""
    with seeded.cursor() as cur:        # a cluster that holds no article of this scenario
        cur.execute("UPDATE scenario_settings SET clustering_json = %s::jsonb WHERE scenario_id = %s",
                    (_clustering_cache({0: [9601, 9602, 9603, 9604, 9605], 7: [999999]},
                                       {0: "Everything", 7: "Elsewhere"}), SID))
    r = _client().post(f"/user-scenarios/{SID}/subset/apply",
                       json={"clusters": [7]}, headers=HDR)
    assert r.status_code == 422 and "aucun article" in r.json()["detail"].lower()
    with seeded.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM article_scenarios WHERE scenario_id = %s "
                    "AND screening_reason LIKE %s", (SID, SUBSET_REASON_PREFIX + "%"))
        assert cur.fetchone()[0] == 0


def test_a_selection_with_no_dimension_at_all_is_refused(seeded):
    r = _client().post(f"/user-scenarios/{SID}/subset/preview", json={})
    assert r.status_code == 422


def _tag(conn, doc_id: int, *concepts: tuple[str, str]) -> None:
    payload = json.dumps({"concepts": [{"t": t, "en": en, "fr": en} for t, en in concepts]})
    with conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET concepts_json = %s::jsonb WHERE id = %s",
                    (payload, doc_id))


def test_narrowing_by_concept_keeps_the_articles_whose_concepts_were_never_extracted(seeded):
    """The concept map's blind spot is different from the clustering's but has the same
    shape: an article with no extracted concepts is not "about something else", it is
    uncharacterised. 9605 has no concepts_json and must survive a concept narrowing."""
    _tag(seeded, 9601, ("pathogen", "dengue virus"))
    _tag(seeded, 9602, ("pathogen", "dengue virus"))
    _tag(seeded, 9603, ("vector", "Aedes albopictus"))
    _tag(seeded, 9604, ("vector", "Aedes albopictus"))
    body = _client().post(f"/user-scenarios/{SID}/subset/preview",
                          json={"concepts": ["pathogen:dengue virus"]}).json()
    assert body["by_dimension"]["concepts"] == {"in": 2, "out": 2, "unknown": 1}
    assert body["exclude_sample"] == [9603, 9604]
    assert body["undecided_sample"] == [9605]
    assert body["meta"]["concepts"]["labels"] == ["dengue virus"]


def test_a_cluster_and_a_concept_narrow_together(seeded):
    """The composition the reviewer asked for: this cluster, and among it, this concept.
    `combine='all'` is an intersection over the dimensions that can judge."""
    _tag(seeded, 9601, ("pathogen", "dengue virus"))
    _tag(seeded, 9602, ("pathogen", "chikungunya virus"))
    _tag(seeded, 9603, ("pathogen", "dengue virus"))
    _tag(seeded, 9604, ("pathogen", "chikungunya virus"))
    body = _client().post(f"/user-scenarios/{SID}/subset/preview",
                          json={"clusters": [0], "concepts": ["pathogen:dengue virus"],
                                "combine": "all"}).json()
    assert body["exclude_sample"] == [9602, 9603, 9604]    # 9602 wrong pathogen, 9603/4 wrong cluster
    assert body["keep"] == 2                               # 9601, plus the unjudged 9605
    union = _client().post(f"/user-scenarios/{SID}/subset/preview",
                           json={"clusters": [0], "concepts": ["pathogen:dengue virus"],
                                 "combine": "any"}).json()
    assert union["exclude_sample"] == [9604]               # only the one both dimensions reject


def test_a_concept_absent_from_the_map_is_a_404_not_an_empty_selection(seeded):
    _tag(seeded, 9601, ("pathogen", "dengue virus"))
    r = _client().post(f"/user-scenarios/{SID}/subset/preview",
                       json={"concepts": ["pathogen:measles virus"]})
    assert r.status_code == 404 and "measles virus" in r.json()["detail"]


def test_a_selection_without_a_clustering_cache_says_so_instead_of_emptying_the_corpus(seeded):
    """Without a cached clustering, EVERY article is unclustered. A naive implementation
    would call them all "not in the clusters you chose" and throw the corpus away."""
    with seeded.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET clustering_json = NULL WHERE scenario_id = %s", (SID,))
    r = _client().post(f"/user-scenarios/{SID}/subset/preview", json={"clusters": [0]})
    assert r.status_code == 404 and "clustering" in r.json()["detail"].lower()


def test_an_unknown_cluster_is_a_404_naming_the_ones_that_exist(seeded):
    r = _client().post(f"/user-scenarios/{SID}/subset/preview", json={"clusters": [42]})
    assert r.status_code == 404 and "42" in r.json()["detail"]


def test_applying_needs_the_write_key(seeded):
    assert _client().post(f"/user-scenarios/{SID}/subset/apply",
                          json={"clusters": [0]}).status_code in (401, 403)
    assert _client().post(f"/user-scenarios/{SID}/subset/undo").status_code in (401, 403)


def test_applying_drops_the_caches_computed_on_the_old_corpus(seeded):
    """The clustering on screen described a corpus that no longer exists. Serving it after
    the narrowing would present the previous corpus as the current one, which is the exact
    failure the threshold slider had."""
    _client().post(f"/user-scenarios/{SID}/subset/apply", json={"clusters": [0]}, headers=HDR)
    with seeded.cursor() as cur:
        cur.execute("SELECT clustering_json, concept_graph_json, recommended_actions_json "
                    "FROM scenario_settings WHERE scenario_id = %s", (SID,))
        assert cur.fetchone() == (None, None, None)


# ── narrowing by study design and by evidence level ──────────────────────────
# The dimension the user asked for, with the same semantics as the clusters: it EXCLUDES
# from the corpus, so every extraction afterwards reads the narrowed set and the PRISMA
# records it as a reviewer decision rather than a display filter.
from api.subsets import describe_selection, plan_subset  # noqa: E402

_DESIGNS = {1: "Essai contrôlé randomisé", 2: "Cohorte", 3: "Essai contrôlé randomisé",
            4: "Devis non précisé"}
_LEVELS = {1: "Élevée", 2: "Faible", 3: "Élevée", 4: "Non évaluée"}


def test_keeping_only_one_design_excludes_the_rest():
    got = plan_subset([1, 2, 3, 4], design_of=_DESIGNS,
                      designs_wanted={"Essai contrôlé randomisé"})
    assert got["keep_ids"] == [1, 3] and got["exclude_ids"] == [2, 4]


def test_keeping_only_strong_evidence():
    got = plan_subset([1, 2, 3, 4], level_of=_LEVELS, levels_wanted={"Élevée"})
    assert got["keep_ids"] == [1, 3] and got["exclude_ids"] == [2, 4]


def test_an_unstated_design_is_a_verdict_not_an_abstention():
    """Unlike a cluster, which can genuinely not know an article (capped projection), the
    design map has READ every relevant article. "Devis non précisé" is what it found, so a
    reviewer can decide to keep or drop those rather than have the dimension abstain."""
    got = plan_subset([1, 2, 3, 4], design_of=_DESIGNS, designs_wanted={"Devis non précisé"})
    assert got["keep_ids"] == [4]
    assert got["undecided_ids"] == [], "nothing abstained: every article was judged"


def test_an_article_missing_from_the_map_still_abstains():
    """A row added after the map was built is genuinely unjudged, and `unassigned` decides
    its fate as it does everywhere else."""
    got = plan_subset([1, 99], design_of=_DESIGNS, designs_wanted={"Cohorte"})
    assert got["undecided_ids"] == [99] and 99 in got["keep_ids"]
    strict = plan_subset([1, 99], design_of=_DESIGNS, designs_wanted={"Cohorte"},
                         unassigned="exclude")
    assert 99 in strict["exclude_ids"]


def test_design_and_level_combine_like_every_other_dimension():
    """`all` keeps what both accept; `any` keeps what either does. A dimension that cannot
    judge still does not vote."""
    both = plan_subset([1, 2, 3, 4], design_of=_DESIGNS, designs_wanted={"Cohorte"},
                       level_of=_LEVELS, levels_wanted={"Élevée"}, combine="all")
    assert both["keep_ids"] == [], "no article is both a cohort and strong evidence"
    either = plan_subset([1, 2, 3, 4], design_of=_DESIGNS, designs_wanted={"Cohorte"},
                         level_of=_LEVELS, levels_wanted={"Élevée"}, combine="any")
    assert either["keep_ids"] == [1, 2, 3]


def test_the_tally_names_the_new_dimensions():
    got = plan_subset([1, 2, 3, 4], design_of=_DESIGNS, designs_wanted={"Cohorte"},
                      level_of=_LEVELS, levels_wanted={"Élevée"})
    assert set(got["by_dimension"]) == {"designs", "levels"}
    assert got["by_dimension"]["designs"] == {"in": 1, "out": 3, "unknown": 0}


def test_selecting_nothing_at_all_is_still_refused():
    with pytest.raises(ValueError, match="aucune dimension"):
        plan_subset([1, 2])


def test_the_reason_names_the_designs_in_words():
    """It lands in `screening_reason` and has to read in a Methods section, so it names
    what was kept rather than an internal code."""
    reason = describe_selection(None, None, None, "all", "keep", threshold=0.45,
                                designs=["Essai contrôlé randomisé"])
    assert "devis Essai contrôlé randomisé" in reason
    assert "seuil 0.45" in reason

    both = describe_selection(None, None, None, "all", "keep",
                              designs=["Essai contrôlé randomisé"], levels=["Élevée"])
    assert "devis" in both and "niveaux de preuve Élevée" in both and " et " in both
