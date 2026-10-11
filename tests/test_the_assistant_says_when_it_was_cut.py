"""An assistant answer cut at its length limit says so, on both paths.

The stream the interface reads had a ceiling of 1,500 tokens and never read why the model
stopped: a cut answer was followed by `done`, and the client archived it as complete. The
non-stream path (1,200 tokens) had no check either. Both now allow 3,000 tokens; the
stream sends a distinct `cut` event before `done`, and the non-stream answer carries
`cut: true`. The interface no longer sends `top_k: 12`, half the server's own default."""
import asyncio
import inspect
import json
import types

import pytest

import llm_usage
import main
from api import assistant
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-assistant-cut-test"
IDS = (8801, 8802)


class _Stream:
    """An async stream of chat chunks, the last one carrying `finish_reason`."""

    def __init__(self, tokens, finish):
        self.chunks = [types.SimpleNamespace(choices=[types.SimpleNamespace(
            delta=types.SimpleNamespace(content=t), finish_reason=None)]) for t in tokens]
        self.chunks.append(types.SimpleNamespace(choices=[types.SimpleNamespace(
            delta=types.SimpleNamespace(content=None), finish_reason=finish)]))
        self.chunks.insert(0, types.SimpleNamespace(choices=[]))          # a chunk with no choice

    def __aiter__(self):
        async def _gen():
            for c in self.chunks:
                yield c
        return _gen()


def _events(stream) -> list[str]:
    async def _collect():
        return [e async for e in assistant._answer_events(stream, 3000, "test")]
    return asyncio.run(_collect())


def test_a_cut_stream_ends_with_a_cut_event():
    events = _events(_Stream(["The answer ", "stops mid"], "length"))
    assert [json.loads(e[len("data: "):])["token"] for e in events[:2]] == ["The answer ", "stops mid"]
    assert events[-1].startswith("event: cut\n")
    payload = json.loads(events[-1].split("data: ", 1)[1])
    assert payload == {"cut": True, "reason": "length", "max_tokens": 3000}


def test_a_complete_stream_has_no_cut_event():
    events = _events(_Stream(["A whole answer."], "stop"))
    assert len(events) == 1 and "event: cut" not in events[0]


def test_the_filtered_stream_allows_3000_tokens_and_says_cut_before_done():
    assert assistant._ANSWER_MAX_TOKENS == 3000
    src = inspect.getsource(assistant.ask_stream_filtered)
    assert "max_tokens=_ANSWER_MAX_TOKENS" in src and "max_tokens=1500" not in src
    assert src.index("_answer_events(") < src.rindex("event: done")
    # The server's default applies: 24 articles, one excerpt each.
    assert 'payload.get("top_k", 24)' in src


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


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
        ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("is_duplicate", "BOOLEAN DEFAULT FALSE"),
                         ("authors", "TEXT"), ("doi", "TEXT"), ("journal", "TEXT")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS similarity_score DOUBLE PRECISION")
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding TEXT")
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS metadata_json JSONB")
        main._ensure_extraction_columns()
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM document_chunk WHERE document_id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters) "
                    "VALUES (%s, 'Avian influenza', 'h5n1', 'boolean', '{}')", (SID,))
        for doc_id in IDS:
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                        "is_duplicate) VALUES (%s, 'Poultry workers and H5N1', 'Poultry workers were tested for H5N1 antibodies.', 2025, "
                        "'pubmed', 'literev', false)", (doc_id,))
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) "
                        "VALUES (%s, %s, 0.9)", (SID, doc_id))
            cur.execute("INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type) "
                        "VALUES (%s, 0, 'Poultry workers exposed to sick birds were seropositive.', "
                        "'title_abstract')", (doc_id,))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM document_chunk WHERE document_id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))


@pytest.mark.parametrize("finish, cut", [("length", True), ("stop", False)])
def test_the_non_stream_answer_says_whether_it_was_cut(seeded, monkeypatch, finish, cut):
    from fastapi.testclient import TestClient
    seen: list = []

    class _Fake:
        def __init__(self, *a, **kw):
            def _no_embedding(**_):
                raise RuntimeError("no embeddings here")        # the lexical branch answers
            def _chat(**kw):
                seen.append(kw)
                msg = types.SimpleNamespace(content="An answer")
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish)])
            self.embeddings = types.SimpleNamespace(create=_no_embedding)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_chat))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", _Fake)
    out = TestClient(main.app).post(f"/user-scenarios/{SID}/rag", json={"question": "poultry workers"}).json()
    assert out["answer"] == "An answer" and out["cut"] is cut
    assert seen[0]["max_tokens"] == 3000
    # The excerpts are described as what they are: the closest to the question asked.
    assert "les plus proches de la question posee" in seen[0]["messages"][1]["content"]
