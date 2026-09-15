"""Clustering at scale: the shared core parses pgvector embeddings with numpy
(float32), keeps the TF-IDF matrix sparse, and the payload builder works on it; the
document selection is capped to the most relevant CLUSTER_MAX_DOCS articles and the
payload reports the eligible total."""
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sklearn")

import main  # noqa: E402

WORDS = "influenza surveillance wastewater hospital vaccination outbreak model cohort trial".split()


def _docs(n: int, dims: int = 8):
    rnd = np.random.RandomState(0)
    out = []
    for i in range(n):
        vec = rnd.rand(dims)
        out.append({"id": i + 1, "title": f"Doc {i} " + " ".join(WORDS[(i + k) % len(WORDS)] for k in range(3)),
                    "abstract": " ".join(WORDS[(i * 3 + k) % len(WORDS)] for k in range(40)),
                    "year": 2010 + i % 10, "journal": "J",
                    "embedding_str": "[" + ",".join(f"{x:.4f}" for x in vec) + "]" if i % 5 else None})
    return out


def test_cluster_core_parses_embeddings_as_float32_and_keeps_tfidf_sparse():
    docs = _docs(30)
    texts = [f"{d['title']} {d['abstract']}" for d in docs]
    cc = main._cluster_core(docs, texts, openai_key=None, allow_openai_embeddings=False, tfidf_min_df=1)
    assert cc["embedding_source"] == "db_pgvector"
    assert len(cc["labels"]) == 30 and cc["embedding_2d"].shape == (30, 2)
    assert hasattr(cc["X_dense"], "toarray")                     # sparse, not 25,000 × 800 floats
    payload = main._build_clusters_payload("usr-x", docs, cc, n_docs_total=25000)
    assert payload["n_docs"] == 30 and payload["n_docs_total"] == 25000
    assert sum(c["n_docs"] for c in payload["clusters"]) == 30
    assert all(isinstance(w, str) for c in payload["clusters"] for w in c["top_words"])


def test_cluster_core_falls_back_to_tfidf_without_embeddings():
    docs = [{**d, "embedding_str": None} for d in _docs(20)]
    texts = [f"{d['title']} {d['abstract']}" for d in docs]
    cc = main._cluster_core(docs, texts, openai_key=None, allow_openai_embeddings=False, tfidf_min_df=1)
    assert cc["embedding_source"] == "tfidf" and len(cc["labels"]) == 20


def test_clustering_docs_are_capped_to_the_most_relevant(monkeypatch):
    captured = {}

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, stmt, params=None):
            sql = str(stmt)
            captured.setdefault("sql", []).append(sql)
            captured["params"] = params

            class _R:
                def scalar(self_inner):
                    return 25000

                def mappings(self_inner):
                    class _M:
                        def all(self_m):
                            return []
                    return _M()
            return _R()

    monkeypatch.setattr(main.engine, "connect", lambda: _Conn())
    docs, total = main._clustering_docs("usr-x", 0.45, cap=3000)
    assert docs == [] and total == 25000
    select_sql = captured["sql"][-1]
    assert "LIMIT :cap" in select_sql and captured["params"]["cap"] == 3000
    assert "= 'included') DESC" in select_sql and "similarity_score DESC" in select_sql   # relevance, not year
    assert main.CLUSTER_MAX_DOCS >= 200
