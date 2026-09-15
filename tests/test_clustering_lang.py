"""Cluster summaries follow the language toggle.

The clustering cache used to be served regardless of the requested language, so
summaries generated in French came back under the English toggle. Now a cached
payload is served only when every dense cluster carries a summary in that language;
otherwise the structure is kept and only the summaries are regenerated (LLM-only
background job), and the page polls /clustering/status in the same language.
"""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from conftest import patch_app  # noqa: E402

SID = "usr-lang-cluster-test"


def _payload(summaries_lang: str | None, with_noise: bool = True) -> dict:
    clusters = [
        {"cluster_id": 0, "cluster_name": "Cluster 1", "is_noise": False, "n_docs": 3,
         "center_x": 0.0, "center_y": 0.0, "top_words": ["flu"],
         "summary": "Résumé en français.",
         "summaries": ({summaries_lang: "Résumé en français."} if summaries_lang else {}),
         "representative_doc": {"id": 1, "title": "t", "year": 2020, "journal": "j"},
         "points": [{"id": 1, "title": "t", "year": 2020, "x": 0.1, "y": 0.1},
                    {"id": 2, "title": "u", "year": 2021, "x": 0.9, "y": 0.9}]},
    ]
    if with_noise:
        clusters.append({"cluster_id": -1, "cluster_name": "Non-classés", "is_noise": True, "n_docs": 1,
                         "center_x": 1.0, "center_y": 1.0, "top_words": [],
                         "summary": "Bruit de fond (articles non regroupés).", "summaries": {},
                         "representative_doc": {"id": 2, "title": "u", "year": 2021, "journal": ""},
                         "points": []})
    return {"scenario_id": SID, "n_docs": 4, "n_clusters": 1, "method": "hdbscan",
            "embedding_source": "openai", "clusters": clusters, "from_cache": True, "lang": summaries_lang}


# ── pure helpers ──────────────────────────────────────────────────────────────
def test_clusters_have_lang_checks_every_dense_cluster():
    assert main._clusters_have_lang(_payload("fr"), "fr") is True
    assert main._clusters_have_lang(_payload("fr"), "en") is False
    assert main._clusters_have_lang(_payload(None), "fr") is False     # pipeline cache: no summaries
    assert main._clusters_have_lang({"clusters": []}, "en") is True     # nothing to translate
    assert main._clusters_have_lang(None, "en") is False


def test_localize_picks_the_requested_language_and_translates_noise_labels():
    p = _payload("fr")
    p["clusters"][0]["summaries"]["en"] = "Summary in English."
    en = main._localize_clusters_payload(p, "en")
    assert en["lang"] == "en"
    assert en["clusters"][0]["summary"] == "Summary in English."
    assert en["clusters"][1]["cluster_name"] == "Unclassified"
    assert en["clusters"][1]["summary"] == "Background noise (articles not grouped)."
    fr = main._localize_clusters_payload(p, "fr")
    assert fr["clusters"][0]["summary"] == "Résumé en français."
    assert fr["clusters"][1]["cluster_name"] == "Non-classés"
    # the cached object itself is untouched
    assert p["clusters"][0]["summary"] == "Résumé en français."
    assert p["clusters"][1]["cluster_name"] == "Non-classés"


def test_localize_caps_the_points_evenly_and_reports_totals():
    # 25,000 points were 2.5 MB of JSON and 25,000 SVG circles; the served view is a
    # deterministic, evenly spaced sample with the true totals alongside.
    p = _payload("fr", with_noise=False)
    big = [{"id": i, "title": "t", "year": 2020, "x": float(i), "y": 0.0} for i in range(1000)]
    small = [{"id": 5000 + i, "title": "u", "year": 2020, "x": 0.0, "y": float(i)} for i in range(10)]
    p["clusters"][0]["points"] = big
    p["clusters"].append({**p["clusters"][0], "cluster_id": 1, "points": small, "summaries": {"fr": "s"}})
    out = main._localize_clusters_payload(p, "fr", max_points=100)
    assert out["points_total"] == 1010 and out["points_shown"] <= 110
    c0, c1 = out["clusters"]
    assert c0["points_total"] == 1000 and 90 <= len(c0["points"]) <= 100
    assert c1["points_total"] == 10 and len(c1["points"]) == 10           # small clusters keep ≥ 5
    xs = [pt["x"] for pt in c0["points"]]
    assert xs == sorted(xs) and xs[0] == 0.0 and xs[-1] >= 900.0            # spread over the whole cloud
    assert main._localize_clusters_payload(p, "fr", max_points=100) == out  # deterministic
    # under the cap: everything is served, totals still reported
    full = main._localize_clusters_payload(p, "fr", max_points=5000)
    assert full["points_shown"] == full["points_total"] == 1010
    assert len(p["clusters"][0]["points"]) == 1000                          # cache untouched


def test_localize_falls_back_to_existing_summary_and_localizes_messages():
    legacy = _payload(None)                       # pre-language cache: single `summary`
    out = main._localize_clusters_payload(legacy, "en")
    assert out["clusters"][0]["summary"] == "Résumé en français."   # nothing better available
    msg = {"scenario_id": SID, "clusters": [], "message_code": "insufficient_corpus", "message": "x"}
    assert "at least 5 articles" in main._localize_clusters_payload(msg, "en")["message"]
    assert "minimum 5 articles" in main._localize_clusters_payload(msg, "fr")["message"]


# ── endpoint: cache in the other language → summaries-only background job ─────
@pytest.fixture()
def no_db(monkeypatch):
    patch_app(monkeypatch, "_get_user_scenario_or_404", lambda sid: {"id": sid})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    main._clustering_jobs.pop(SID, None)
    yield
    main._clustering_jobs.pop(SID, None)


class _FakeThread:
    started: list = []

    def __init__(self, target=None, args=(), daemon=None):
        self.target, self.args = target, args

    def start(self):
        _FakeThread.started.append((self.target, self.args))


def test_cached_summaries_in_requested_language_are_served_directly(no_db, monkeypatch):
    patch_app(monkeypatch, "_load_viz_cache", lambda sid, col, ttl=86400: _payload("fr"))
    res = main.get_user_scenario_clustering(SID, False, "fr")
    assert res.get("status") != "running"
    assert res["lang"] == "fr" and res["clusters"][0]["summary"] == "Résumé en français."


def test_cached_summaries_in_other_language_trigger_relocalization(no_db, monkeypatch):
    import threading
    _FakeThread.started.clear()
    monkeypatch.setattr(threading, "Thread", _FakeThread)
    patch_app(monkeypatch, "_load_viz_cache", lambda sid, col, ttl=86400: _payload("fr"))
    res = main.get_user_scenario_clustering(SID, False, "en")
    assert res["status"] == "running" and res["lang"] == "en"
    assert len(_FakeThread.started) == 1
    target, args = _FakeThread.started[0]
    assert target is main._relocalize_clustering_background       # summaries only, no re-clustering
    assert args[0] == SID and args[2] == "en"
    assert main._clustering_jobs[SID]["status"] == "running"
    # a second call while the job runs does not start another thread
    res2 = main.get_user_scenario_clustering(SID, False, "en")
    assert res2["status"] == "running" and len(_FakeThread.started) == 1


def test_without_openai_key_the_cache_is_served_as_is(no_db, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    patch_app(monkeypatch, "_load_viz_cache", lambda sid, col, ttl=86400: _payload("fr"))
    res = main.get_user_scenario_clustering(SID, False, "en")
    assert res.get("status") != "running"
    assert res["clusters"][1]["cluster_name"] == "Unclassified"    # labels still localized


def test_status_endpoint_localizes_the_finished_job(no_db):
    p = _payload("fr")
    p["clusters"][0]["summaries"]["en"] = "Summary in English."
    main._clustering_jobs[SID] = {"status": "done", "result": p}
    assert main.get_user_scenario_clustering_status(SID, "en")["clusters"][0]["summary"] == "Summary in English."
    assert main.get_user_scenario_clustering_status(SID, "fr")["clusters"][0]["summary"] == "Résumé en français."
    main._clustering_jobs[SID] = {"status": "running"}
    assert main.get_user_scenario_clustering_status(SID, "en")["message"].startswith("Computing")


def test_summarize_in_lang_keeps_structure_and_stores_per_language(no_db, monkeypatch):
    """LLM-only relocalization: same clusters/points, summaries added under the new
    language, the other language's summary preserved."""
    calls: list = []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            class _R:
                def mappings(self_inner):
                    return [{"id": 1, "title": "t", "abstract": "a"}, {"id": 2, "title": "u", "abstract": "b"}]
            return _R()

    monkeypatch.setattr(main.engine, "connect", lambda: _Conn())
    patch_app(monkeypatch, "_get_scenario_name", lambda sid: "Flu scenario")

    class _OAI:
        def __init__(self, **kw):
            pass

    import llm_usage
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", _OAI)

    def _fake_summary(client, title, docs, lang):
        calls.append((title, [d["id"] for d in docs], lang))
        return "Summary in English."
    patch_app(monkeypatch, "_cluster_summary_llm", _fake_summary)

    out = main._summarize_clusters_in_lang(SID, _payload("fr"), "en")
    assert calls == [("Flu scenario", [1, 2], "en")]           # nearest-to-centre docs, target lang
    dense = out["clusters"][0]
    assert dense["summaries"] == {"fr": "Résumé en français.", "en": "Summary in English."}
    assert dense["summary"] == "Summary in English." and out["lang"] == "en"
    assert len(dense["points"]) == 2 and out["n_clusters"] == 1   # structure untouched
