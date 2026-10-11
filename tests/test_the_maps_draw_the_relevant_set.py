"""The theme map, the similarity graph and the narrowing read the relevant set the way
the rest of the application does.

- The theme map's 3,000-article cap ranked by cosine alone, ignoring the rerank, and the
  graph ranked by `COALESCE(rerank, cosine)`, two scales that have nothing in common: an
  unreranked article at 0.50 cosine outranked a reranked one at 0.40. Both now draw in
  `relevance_order_sql` order, the order of the Corpus tab, the export and the extractions.
- HDBSCAN's `min_cluster_size = n // 15` made any theme under 69 articles "unclustered" on
  HPAI_last's 1,043, and narrowing to "clusters A and B" then EXCLUDED every unclustered
  article, as if the noise group were a theme nobody ticked.
- Articles of another `project_context` were counted relevant everywhere but loaded by
  neither the extraction loader nor the theme map, and articles without an abstract were
  left out of the map's total, so "3,000 of N" understated N.
"""
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.subsets import NOISE_CLUSTER, plan_subset  # noqa: E402

SID = "usr-mapsorder-0001"
KEY = {"X-API-Key": "test-write-key"}


# ── Pure ─────────────────────────────────────────────────────────────────────
def test_a_theme_of_twenty_articles_is_a_theme_on_a_thousand():
    assert main._hdbscan_min_cluster_size(1043) == 20      # was 69
    assert main._hdbscan_min_cluster_size(3000) == 60      # was 200
    assert main._hdbscan_min_cluster_size(40) == 5


def test_the_payload_counts_what_the_map_does_not_show():
    docs = [{"id": i, "title": f"T{i}", "year": 2020, "journal": "J"} for i in range(1, 7)]
    cc = {"labels": np.array([0, 0, 0, 1, 1, -1]), "embedding_2d": np.random.RandomState(0).rand(6, 2),
          "method": "test", "feature_names": np.array(["a", "b"]), "X_dense": np.ones((6, 2)),
          "embedding_source": "test"}
    p = main._build_clusters_payload("usr-x", docs, cc, n_docs_total=10, n_docs_no_abstract=1)
    assert p["n_docs"] == 6 and p["n_docs_total"] == 10
    assert p["n_docs_left_out"] == 4 and p["n_docs_no_abstract"] == 1 and p["n_docs_capped"] == 3
    assert p["cluster_max_docs"] == main.CLUSTER_MAX_DOCS


def test_the_unclustered_group_is_unknown_not_out():
    """Keeping clusters 0 used to write `excluded` on every article HDBSCAN left alone."""
    of = {1: 0, 2: 0, 3: 1, 4: NOISE_CLUSTER, 5: NOISE_CLUSTER}
    plan = plan_subset([1, 2, 3, 4, 5, 6], cluster_of=of, clusters_wanted={0})
    assert plan["exclude_ids"] == [3]
    assert plan["undecided_ids"] == [4, 5, 6] and set(plan["keep_ids"]) == {1, 2, 4, 5, 6}
    assert plan["unclustered"] == 2
    assert plan["by_dimension"]["clusters"] == {"in": 2, "out": 1, "unknown": 3}
    # Ticking the unclustered group in so many words keeps them as a verdict...
    ticked = plan_subset([1, 2, 3, 4, 5, 6], cluster_of=of, clusters_wanted={0, NOISE_CLUSTER})
    assert ticked["undecided_ids"] == [6] and {4, 5} <= set(ticked["keep_ids"])
    # ... and only the explicit "exclude what cannot be judged" lets them go.
    strict = plan_subset([1, 2, 3, 4, 5, 6], cluster_of=of, clusters_wanted={0}, unassigned="exclude")
    assert set(strict["exclude_ids"]) == {3, 4, 5, 6}


# ── Against a real database ──────────────────────────────────────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


RERANKED = list(range(3, 13))       # ten articles, rerank 0.30 .. 0.39, cosine 0.50


@pytest.fixture()
def corpus(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    long_abstract = "an abstract long enough to be placed on the theme map of the corpus"
    rows = [  # id, title, abstract, sim, rerank, status, duplicate, context
        (1, "Included by hand below the threshold", long_abstract, 0.10, None, "included", False, "literev"),
        (2, "Unreranked, high cosine", long_abstract, 0.90, None, None, False, "literev"),
        *[(i, f"Reranked {i}", long_abstract, 0.50, 0.27 + i / 100, None, False, "literev")
          for i in RERANKED],
        (13, "Another project context", long_abstract, 0.80, 0.95, None, False, "gesica"),
        (14, "Relevant, no usable abstract", "tiny", 0.70, None, None, False, "literev"),
        (15, "Below the threshold", long_abstract, 0.20, None, None, False, "literev"),
        (16, "Excluded by a reviewer", long_abstract, 0.99, 0.99, "excluded", False, "literev"),
        (17, "A duplicate", long_abstract, 0.99, 0.99, None, True, "literev"),
    ]
    with db_conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS document_search")
        cur.execute("DROP TABLE IF EXISTS document_chunk, article_scenarios, literature_document CASCADE")
        cur.execute(
            "CREATE TABLE literature_document ("
            "id bigint PRIMARY KEY, title text NOT NULL, source text NOT NULL DEFAULT 'pubmed',"
            "abstract text, year int DEFAULT 2024, journal text, authors text, doi text,"
            "study_design text, quality_score double precision, citation_count int,"
            "pico_json jsonb, is_duplicate boolean DEFAULT false, project_context text DEFAULT 'literev',"
            "created_at timestamptz DEFAULT now())")
        cur.execute(
            "CREATE TABLE document_chunk ("
            "id bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,"
            "document_id bigint NOT NULL REFERENCES literature_document(id) ON DELETE CASCADE,"
            "chunk_index int DEFAULT 0, content text, chunk_type text, embedding text,"
            "created_at timestamp DEFAULT now())")
        cur.execute(
            "CREATE TABLE article_scenarios (scenario_id text, document_id bigint,"
            "similarity_score double precision, rerank_score double precision,"
            "screening_status text, screening_reason text, screening_notes text,"
            "screened_at timestamp, prior_screening_status text, prior_screening_reason text,"
            "reviewer_1_status text, reviewer_2_status text,"
            "PRIMARY KEY (scenario_id, document_id))")
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters) "
                    "VALUES (%s, 'Maps', 'dengue', 'boolean', '{}')", (SID,))
        for (i, title, abstract, sim, rr, status, dup, ctx) in rows:
            cur.execute("INSERT INTO literature_document (id, title, abstract, is_duplicate, project_context)"
                        " VALUES (%s, %s, %s, %s, %s)", (i, title, abstract, dup, ctx))
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score,"
                        " rerank_score, screening_status) VALUES (%s, %s, %s, %s, %s)",
                        (SID, i, sim, rr, status))
            cur.execute("INSERT INTO document_chunk (document_id, content, chunk_type, embedding)"
                        " VALUES (%s, %s, 'title_abstract', %s)",
                        (i, f"{title}\n\n{abstract}", f"[{0.1 + i / 100:.3f}, 0.2, 0.3]"))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))


def test_the_theme_map_draws_in_the_apps_relevance_order(corpus):
    counts: dict = {}
    docs, total = main._clustering_docs(SID, 0.45, cap=5, counts=counts)
    # Hand inclusions first, then the reranked by rerank score; the unreranked 0.90 cosine
    # no longer jumps the queue.
    assert [d["id"] for d in docs] == [1, 13, 12, 11, 10]
    # The total is the WHOLE relevant set, the article without an abstract included.
    assert total == 14
    assert counts == {"relevant": 14, "clusterable": 13, "no_abstract": 1}


def test_relevant_means_the_same_everywhere(corpus):
    """Article 13 belongs to another project context: counted relevant by the counters,
    it must also be loaded for the exports and the extractions, and drawn on the map."""
    loaded = {a["id"] for a in main._get_above_threshold_articles(SID)}
    assert 13 in loaded
    assert len(loaded) == main.scenario_counts(SID)["relevant"] == 14
    drawn, _ = main._clustering_docs(SID, 0.45)
    assert 13 in {d["id"] for d in drawn}


def test_the_similarity_graph_ranks_like_the_rest_of_the_app(corpus):
    kg = main._compute_user_kg(SID, max_nodes=10, min_similarity=0.35)
    ids = [n["id"] for n in kg["nodes"]]
    assert ids == [1, 13, 12, 11, 10, 9, 8, 7, 6, 5], (
        "the graph mixes the rerank and the cosine scales again")
    assert 2 not in ids          # unreranked at 0.90 cosine: after every reranked article
    assert kg["n_total"] == 14


def _cache(groups: dict) -> dict:
    return {"n_docs": sum(len(v) for v in groups.values()), "n_docs_total": 14, "lang": "en",
            "n_clusters": len([g for g in groups if g != NOISE_CLUSTER]), "method": "test",
            "clusters": [{"cluster_id": cid, "cluster_name": f"Group {cid}",
                          "is_noise": cid == NOISE_CLUSTER, "n_docs": len(ids), "top_words": [],
                          "summary": "", "points": [{"id": i, "x": 0.0, "y": 0.0} for i in ids]}
                         for cid, ids in groups.items()]}


def test_narrowing_to_some_themes_keeps_what_the_clustering_left_unclustered(corpus):
    main._save_viz_cache(SID, "clustering", _cache({0: [1, 13, 2], 1: [3, 4], NOISE_CLUSTER: [5, 6]}))
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    body = client.post(f"/user-scenarios/{SID}/subset/preview", json={"clusters": [0]}).json()
    assert body["exclude_sample"] == [3, 4]
    assert {5, 6} <= set(body["undecided_sample"]) and body["unclustered"] == 2
    ticked = client.post(f"/user-scenarios/{SID}/subset/preview",
                         json={"clusters": [0, NOISE_CLUSTER]}).json()
    assert not ({5, 6} & set(ticked["undecided_sample"])) and ticked["exclude_sample"] == [3, 4]
    out = client.post(f"/user-scenarios/{SID}/subset/apply", json={"clusters": [0]}, headers=KEY).json()
    assert out["applied"] == 2
    with corpus.cursor() as cur:
        cur.execute("SELECT document_id FROM article_scenarios WHERE scenario_id = %s"
                    " AND screening_status = 'excluded' AND document_id IN (5, 6)", (SID,))
        assert cur.fetchall() == [], "the unclustered articles were thrown out"
    assert json.loads(json.dumps(out))["status"] == "applied"
