"""What the cross-encoder is asked to judge, and with which question.

Each article went to Cohere as `title + abstract[:1500]`: a structured abstract lost its
results and conclusions, the very part that says whether it answers the question, before
being judged. The query went as `query[:4000]` with its PubMed field tags still in, so a
long Boolean query lost its tail, the failure EMBED_QUERY_MAX_CHARS documents for the old
embedding cut, and nothing said so.
"""
import logging

import pytest

pytest.importorskip("fastapi")

from api import relevance as R  # noqa: E402

LONG_QUERY = (" OR ".join(f'"exposure concept number {i:03d}"[tiab]' for i in range(160))
              + ' AND "Influenza in Birds"[mh]')


def test_a_document_keeps_up_to_4000_characters_by_default():
    assert R.RERANK_DOC_MAX_CHARS == 4000
    abstract = "Background. " + "Methods and results, in detail. " * 200 + "CONCLUSION-MARKER"
    doc = R.rerank_document_text("A trial", abstract)
    assert len(doc) == 4000
    assert doc.startswith("A trial\n\nBackground.")
    # Well past the old 1,500-character cut.
    assert len(R.rerank_document_text("T", "x" * 3000)) == len("T\n\n") + 3000


def test_the_document_cap_is_an_environment_setting(monkeypatch):
    from api.core import _env_int
    monkeypatch.setenv("RERANK_DOC_MAX_CHARS", "6000")
    assert _env_int("RERANK_DOC_MAX_CHARS", 4000, 200) == 6000
    monkeypatch.setattr(R, "RERANK_DOC_MAX_CHARS", 1000)
    assert len(R.rerank_document_text("T", "y" * 5000)) == 1000


def test_the_query_loses_its_field_tags_before_any_cut():
    q = R.rerank_query_text('"avian influenza"[tiab] AND (H5N1[mh] OR poultry[tw])')
    assert "[" not in q and "avian influenza" in q and "H5N1" in q
    assert "  " not in q


def test_a_cut_query_is_logged_once_with_its_scenario(caplog):
    caplog.set_level(logging.WARNING, logger="literev-api")
    q = R.rerank_query_text(LONG_QUERY, "usr-cut-0001")
    assert len(q) == R.RERANK_QUERY_MAX_CHARS
    assert any("usr-cut-0001" in r.getMessage() and "coupée" in r.getMessage() for r in caplog.records)
    # Without a scenario (each batch of a sweep), the same cut is applied silently.
    caplog.clear()
    assert R.rerank_query_text(LONG_QUERY) == q and not caplog.records


def test_what_reaches_cohere_has_no_field_tag(monkeypatch):
    sent: list = []

    class _Resp:
        status_code = 200
        headers: dict = {}

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.5}]}

    def _post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        return _Resp()

    import requests as _rq
    monkeypatch.setenv("COHERE_API_KEY", "k")
    monkeypatch.setattr(_rq, "post", _post)
    assert R._cohere_rerank('"avian influenza"[tiab] AND H5N1[mh]', ["doc"]) == [0.5]
    assert sent[0]["query"] == '"avian influenza" AND H5N1'


def test_a_sweep_sends_the_long_abstract_and_the_stripped_query(monkeypatch):
    """Through `_run_cross_encoder_rerank`, with the database replaced by two rows."""
    seen: dict = {}

    def _fake_cohere(query, docs, model="rerank-v3.5", failure=None, **kw):
        seen["query"], seen["docs"] = query, docs
        return [0.5] * len(docs)

    class _Rows:
        def __init__(self, rows):
            self._rows = rows

        def mappings(self):
            return self

        def all(self):
            return self._rows

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, stmt, params=None):
            return _Rows([{"id": 1, "title": "Long", "abstract": "z" * 3500},
                          {"id": 2, "title": "Short", "abstract": "a short abstract here"}])

    class _Engine:
        def connect(self):
            return _Conn()

        def begin(self):
            return _Conn()

    monkeypatch.setenv("COHERE_API_KEY", "k")
    monkeypatch.setattr(R, "engine", _Engine())
    monkeypatch.setattr(R, "_cohere_rerank", _fake_cohere)
    out = R._run_cross_encoder_rerank("usr-x", '"avian influenza"[tiab]')
    assert out["scored"] == 2
    assert seen["query"] == '"avian influenza"'
    assert len(seen["docs"][0]) == len("Long\n\n") + 3500, "the abstract was cut at 1,500 again"
