"""The relevance gate scores what every article has: its title and abstract.

`similarity_score`, the number the gate cuts, was the MAXIMUM cosine over every chunk of
an article, abstract and full-text sections alike. A full-text article drew thirty times,
an abstract-only one once, so at a fixed threshold full-text articles passed more easily
and fetching a full text could only raise a score. On HPAI_last 869 of the 1,043 relevant
articles have a full text, against 4,176 of the 15,660 in the corpus. The full-text
chunks keep serving retrieval (assistant, excerpts); they no longer decide relevance.

The OpenAI fallback (articles whose chunk is not embedded yet) embedded
`title + abstract[:1500]` while the worker embeds the whole chunk: the same article could
get two cosines depending on the path taken, against a hard threshold. Both paths now
embed the same text.
"""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api import relevance as R  # noqa: E402
from api.documents import _truncate_to_tokens  # noqa: E402

SID = "usr-tascore-0001"
DIM = 1536


# ── The text, pure ───────────────────────────────────────────────────────────
def test_the_fallback_embeds_exactly_what_the_worker_embeds():
    """The worker embeds `_truncate_to_tokens(chunk.content)` (documents._embed_one_call)."""
    content = "A title\n\n" + "A structured abstract. Results: it worked. " * 120
    assert R._fallback_text({"chunk_text": content, "title": "ignored", "abstract": "ignored"}) \
        == _truncate_to_tokens(content)
    # The abstract is no longer cut at 1,500 characters.
    assert len(R._fallback_text({"chunk_text": content})) > 1500


def test_without_a_chunk_the_fallback_rebuilds_the_chunk_the_ingestion_writes():
    """api/sources.py writes `f"{title}\\n\\n{abstract or ''}".strip()`."""
    assert R.title_abstract_text("A title", "An abstract") == "A title\n\nAn abstract"
    assert R.title_abstract_text("Only a title", None) == "Only a title"
    assert R._fallback_text({"chunk_text": None, "title": "A title", "abstract": "An abstract"}) \
        == _truncate_to_tokens("A title\n\nAn abstract")


# ── Fakes for the embedding model ────────────────────────────────────────────
def _unit(*head):
    vec = [0.0] * DIM
    for i, v in enumerate(head):
        vec[i] = v
    return vec


class _D:
    def __init__(self, embedding, index):
        self.embedding, self.index = embedding, index


class _R:
    def __init__(self, data):
        self.data = data


def _fake_openai(query_vec, doc_vec, sink):
    class _Embeddings:
        def create(self, model=None, input=None, **kw):
            if isinstance(input, str):                  # the query
                return _R([_D(query_vec, 0)])
            sink.extend(input)                          # a fallback batch
            return _R([_D(doc_vec, i) for i in range(len(input))])

    class _Client:
        def __init__(self, *a, **k):
            self.embeddings = _Embeddings()
    return _Client


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


# ── The fast path, over a real pgvector column ───────────────────────────────
def _vector_ready(conn) -> bool:
    """pgvector INSTALLED and a vector `document_chunk.embedding` (a deployment's schema).
    Not created here: a test that installs an extension changes the database for every
    test after it (cf. test_rag_whole_corpus)."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if cur.fetchone() is None:
            return False
        cur.execute("SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a"
                    " WHERE a.attrelid = to_regclass('document_chunk') AND a.attname = 'embedding'")
        row = cur.fetchone()
        return bool(row) and str(row[0]).startswith("vector")


IDS = (97101, 97102, 97103)


@pytest.fixture()
def vector_corpus(db_conn, monkeypatch):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    if not _vector_ready(db_conn):
        pytest.skip("pgvector is not installed with a vector document_chunk.embedding here")
    monkeypatch.setattr(R, "_schedule_autoscore", lambda sid, delay: None)
    vec = lambda v: "[" + ",".join(str(x) for x in v) + "]"  # noqa: E731
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM document_chunk WHERE document_id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.executemany(
            "INSERT INTO literature_document (id, title, source, abstract, project_context)"
            " VALUES (%s, %s, 'pubmed', %s, 'literev')",
            [(97101, "Full text close to the query, abstract not", "an abstract about something else"),
             (97102, "Abstract close to the query", "an abstract about exactly the question"),
             (97103, "Not embedded yet, full text close", "an abstract that the fallback will embed")])
        cur.executemany("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score)"
                        " VALUES (%s, %s, %s)", [(SID, i, None) for i in IDS])
        # 97101: title+abstract at cosine 0.6, a full-text section at cosine 1.0.
        # 97102: title+abstract at cosine 1.0.
        # 97103: a full-text section at cosine 1.0, its title+abstract chunk not embedded.
        cur.executemany(
            "INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type, embedding)"
            " VALUES (%s, %s, %s, %s, %s::vector)",
            [(97101, 0, "ta 97101", "title_abstract", vec(_unit(0.6, 0.8))),
             (97101, 1, "ft 97101", "fulltext_section", vec(_unit(1.0))),
             (97102, 0, "ta 97102", "title_abstract", vec(_unit(1.0))),
             (97103, 1, "ft 97103", "fulltext_section", vec(_unit(1.0)))])
        cur.execute("INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type)"
                    " VALUES (97103, 0, 'Not embedded yet, full text close\n\nan abstract that "
                    "the fallback will embed', 'title_abstract')")
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM document_chunk WHERE document_id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))


def _scores(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT document_id, similarity_score FROM article_scenarios"
                    " WHERE scenario_id = %s", (SID,))
        return {r[0]: r[1] for r in cur.fetchall()}


def test_a_close_full_text_does_not_lift_a_distant_abstract(vector_corpus, monkeypatch):
    sent: list = []
    import llm_usage
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", _fake_openai(_unit(1.0), _unit(0.6, 0.8), sent))
    rep: dict = {}
    R._run_semantic_rerank_inline(SID, "the question", report=rep)
    scores = _scores(vector_corpus)
    assert scores[97101] == pytest.approx(0.6, abs=1e-4), (
        "the score is still the best chunk of the whole paper, full text included")
    assert scores[97102] == pytest.approx(1.0, abs=1e-4)
    # The not-yet-embedded article goes through the fallback, on its title and abstract,
    # whatever its full text says.
    assert scores[97103] == pytest.approx(0.6, abs=1e-4)
    assert sent == [_truncate_to_tokens("Not embedded yet, full text close\n\nan abstract that "
                                        "the fallback will embed")]
    assert rep["fast"] == 2 and rep["fallback"] == 1 and rep["newly_scored"] == 3


def test_a_rescoring_says_how_many_scores_moved(vector_corpus, monkeypatch):
    """Existing scenarios keep the scores the old rule wrote until they are rescored; the
    rescoring then says what it changed."""
    with vector_corpus.cursor() as cur:
        # What the old rule stored: the full-text section's 1.0 for 97101.
        cur.execute("UPDATE article_scenarios SET similarity_score = 1.0"
                    " WHERE scenario_id = %s AND document_id IN (97101, 97102)", (SID,))
    import llm_usage
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", _fake_openai(_unit(1.0), _unit(0.6, 0.8), []))
    rep: dict = {}
    R._run_semantic_rerank_inline(SID, "the question", report=rep)
    assert rep["changed"] == 1          # 97101 fell from 1.0 to 0.6; 97102 stayed at 1.0
    assert rep["newly_scored"] == 1     # 97103
    # Only 97103 changed side: unscored (0 for the gate) before, 0.6 now. 97101 moved but
    # stayed above the 0.45 starting threshold.
    assert rep["crossed_threshold"] == 1
