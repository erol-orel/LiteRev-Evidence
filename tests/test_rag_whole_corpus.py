"""The assistant answers over the whole relevant corpus, and says what it really read.

Three things had drifted apart here, and each one made the RAG tab claim something the
code did not do.

1. THE GATE. "The relevant articles of a scenario" was hand-copied into every module and
   the copies diverged. The one in `/ask/stream/filtered` had lost the duplicate filter
   AND the excluded filter, and admitted unscored articles at any threshold, so the
   assistant could quote an article a reviewer had just thrown out, under a counter that
   was computed on the correct subset. `relevant_gate_sql` is now the single source.

2. BREADTH. The retrieval took the top k CHUNKS, which on a heavily chunked corpus can
   be three articles cut twelve ways. One excerpt per article (DISTINCT ON) spends the
   same context budget on the number of articles the answer touches.

3. THE FIGURES. The other generators (brief, variables, actions) put the whole-corpus
   digest in their prompt, so their counts come from SQL over every relevant article.
   The RAG did not, and answered "most studies report..." on the strength of twelve
   excerpts. It now carries the same digest and the same coverage sentence.
"""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.scenario_store import relevant_gate_sql  # noqa: E402
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-rag-corpus-test"
IDS = (9501, 9502, 9503, 9504, 9505)


# ── The gate itself ──────────────────────────────────────────────────────────
def test_the_gate_carries_all_three_conditions():
    """Duplicates out, reviewer exclusions out, and in only when included by hand or at
    or above the threshold. An unscored article reads as 0, so it stays out unless the
    threshold is 0: the same convention as every other count in the app."""
    sql = relevant_gate_sql(doc="d", link="ars", thr=":thr")
    assert "d.is_duplicate IS NOT TRUE" in sql
    # Le statut lu est celui DE CETTE REVUE : il se lisait
    # COALESCE(ars.screening_status, d.screening_status), donc « à défaut de décision
    # ici, la décision prise dans une autre revue », sur une ligne partagée.
    assert "ars.screening_status IS DISTINCT FROM 'excluded'" in sql
    assert "d.screening_status" not in sql
    assert "COALESCE(ars.similarity_score, 0) >= :thr" in sql
    assert "= 'included'" in sql


def test_the_gate_follows_the_aliases_it_is_given():
    sql = relevant_gate_sql(doc="doc", link="asn", thr=":threshold")
    assert "doc.is_duplicate" in sql and "asn.screening_status" in sql
    assert "COALESCE(asn.similarity_score, 0) >= :threshold" in sql
    assert " d." not in sql and "ars." not in sql


# ── No RAG path writes its own gate any more ─────────────────────────────────
def _rag_sources():
    import inspect

    from api import assistant

    return {
        "/ask/stream": inspect.getsource(assistant.ask_stream),
        "/ask/stream/filtered": inspect.getsource(assistant.ask_stream_filtered),
        "/user-scenarios/{id}/rag": inspect.getsource(assistant.user_scenario_rag_assistant),
    }


def test_every_rag_path_uses_the_shared_gate():
    for name, src in _rag_sources().items():
        assert "relevant_gate_sql" in src, f"{name} does not use the shared relevance gate"


def test_no_rag_path_keeps_a_hand_written_copy_of_the_gate():
    """The failure mode was not a missing gate, it was a SECOND one. Any local rewrite of
    the threshold or screening condition is what drifted, so none may come back."""
    for name, src in _rag_sources().items():
        assert "similarity_score >= :threshold" not in src, f"{name} rewrote the threshold test"
        assert "similarity_score IS NULL" not in src, (
            f"{name} lets unscored articles through at any threshold again")
        assert "COALESCE(ars.similarity_score, 0) >= :thr OR" not in src, (
            f"{name} rewrote the gate inline")


def test_every_rag_path_retrieves_one_excerpt_per_article():
    """Without DISTINCT ON, `top_k` chunks can be a handful of articles cut many ways,
    and the answer rests on three papers while the interface announces the whole relevant
    subset."""
    for name, src in _rag_sources().items():
        assert "DISTINCT ON (d.id)" in src, f"{name} still ranks chunks, not articles"


def test_the_scenario_rag_prompts_carry_the_whole_corpus_digest():
    """Same shape as the brief, the variables and the recommended actions: the digest
    (SQL over EVERY relevant article, no LLM, no sampling) carries the figures, the
    retrieved excerpts carry the quotations, and the coverage sentence says which is
    which. Without it the answer generalises from whatever came back."""
    src = _rag_sources()
    for name in ("/ask/stream/filtered", "/user-scenarios/{id}/rag"):
        assert "corpus_digest" in src[name], f"{name} does not read the corpus digest"
        assert "digest_to_prompt" in src[name], f"{name} does not put the digest in the prompt"
        assert "digest_coverage_note" in src[name], f"{name} omits the coverage sentence"
    # ... and the system prompt forbids counting the excerpts instead.
    assert "CORPUS COMPLET" in src["/ask/stream/filtered"]
    assert "CORPUS COMPLET" in src["/user-scenarios/{id}/rag"]


def test_the_stream_reports_what_it_quoted_not_only_what_it_searched():
    """`papers_used` is the subset SEARCHED. Alone under the answer it read as "this
    answer is built on 2,170 papers", which was never true of a generated paragraph."""
    src = _rag_sources()["/ask/stream/filtered"]
    assert '"papers_quoted"' in src
    assert '"digest_complete"' in src


# ── The gate, exercised end to end against a real database ───────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def seeded(db_conn, monkeypatch):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    # No key: the endpoint takes its lexical branch, which needs no pgvector, and returns
    # the sources in degraded mode. The gate under test is the same in both branches.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created_chunk_table = ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("journal", "TEXT"),
                         ("authors", "TEXT"), ("doi", "TEXT"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS "
                    "similarity_score DOUBLE PRECISION")
        # production's document_chunk carries it (schema.sql); the suite's minimal one
        # does not, and the RAG selects it.
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS "
                    "metadata_json JSONB DEFAULT '{}'::jsonb")
        cur.execute("DELETE FROM document_chunk WHERE document_id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'RAG corpus', 'dengue', 'boolean', '{}', TRUE)", (SID,))
        # Five articles, all matching the question's words, differing only in the one
        # thing the gate looks at.
        cur.execute(
            "INSERT INTO literature_document (id, title, abstract, source, is_duplicate, project_context) VALUES "
            "(9501, 'Above the threshold', 'dengue transmission in the city', 'pubmed', false, 'literev'),"
            "(9502, 'Below the threshold', 'dengue transmission in the countryside', 'pubmed', false, 'literev'),"
            "(9503, 'Excluded by a reviewer', 'dengue transmission, wrong population', 'pubmed', false, 'literev'),"
            "(9504, 'A duplicate record', 'dengue transmission in the city', 'pubmed', true, 'literev'),"
            "(9505, 'Included by a reviewer', 'dengue transmission, hand picked', 'pubmed', false, 'literev')")
        cur.execute(
            "INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, screening_status) VALUES "
            "(%s, 9501, 0.80, NULL), (%s, 9502, 0.10, NULL), (%s, 9503, 0.95, 'excluded'),"
            "(%s, 9504, 0.90, NULL), (%s, 9505, 0.02, 'included')", (SID,) * 5)
        cur.execute("INSERT INTO document_chunk (document_id, chunk_index, content) "
                    "SELECT id, 0, abstract FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) VALUES (%s, 0.45) "
                    "ON CONFLICT (scenario_id) DO UPDATE SET similarity_threshold = 0.45", (SID,))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunk WHERE document_id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def test_the_scenario_rag_never_quotes_an_excluded_article_or_a_duplicate(seeded):
    """Checked through the endpoint rather than through the SQL string: a reviewer's
    exclusion must reach the assistant, and a duplicate must not be quoted."""
    from fastapi.testclient import TestClient

    out = TestClient(main.app).post(f"/user-scenarios/{SID}/rag",
                                    json={"question": "dengue transmission"}).json()
    quoted = {s["document_id"] for s in out["sources"]}
    assert 9503 not in quoted, "an article the reviewer excluded was quoted"
    assert 9504 not in quoted, "a duplicate was quoted"
    assert 9502 not in quoted, "an article below the threshold was quoted"
    assert quoted == {9501, 9505}, "the relevant subset is above the threshold plus the hand-included"


# ── The path where the bug actually lived, over pgvector ─────────────────────
def _has_pgvector(conn) -> bool:
    """Is the extension ALREADY installed? Deliberately not `CREATE EXTENSION`: a test
    that installs an extension changes the database for every test after it."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        return cur.fetchone() is not None


def test_the_streaming_rag_applies_the_same_gate_over_pgvector(seeded, monkeypatch):
    """`/ask/stream/filtered` is the endpoint the RAG tab calls, and the one whose copy of
    the gate had lost the duplicate and excluded filters. Its retrieval needs pgvector, so
    this runs only where the extension is available (CI's plain Postgres skips it), but it
    is the only place the defect was reachable: over a real vector search, an article the
    reviewer excluded came back and was quoted."""
    if not _has_pgvector(seeded):
        pytest.skip("pgvector is not available on this database")

    import json

    with seeded.cursor() as cur:
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding vector(1536)")
        # Every chunk gets the SAME vector, so the ranking cannot be what keeps an
        # article out: only the gate can.
        cur.execute("UPDATE document_chunk SET embedding = %s WHERE document_id = ANY(%s)",
                    ("[" + ",".join(["0.01"] * 1536) + "]", list(IDS)))

    class _FakeEmbeddings:
        def create(self, **kw):
            class _D:
                embedding = [0.01] * 1536
            return type("R", (), {"data": [_D()]})()

    class _FakeOpenAI:
        def __init__(self, *a, **kw):
            self.embeddings = _FakeEmbeddings()

    import llm_usage

    monkeypatch.setattr(llm_usage, "MeteredOpenAI", _FakeOpenAI)

    from fastapi.testclient import TestClient

    body = TestClient(main.app).post(
        "/ask/stream/filtered",
        json={"question": "dengue transmission", "scenario_id": SID}).text
    sources = next((json.loads(line[6:]) for line in body.splitlines()
                    if line.startswith("data: [")), None)
    assert sources is not None, f"no sources event in the stream: {body[:400]}"
    quoted = {s["document_id"] for s in sources}
    assert 9503 not in quoted, "the streaming RAG quoted an article the reviewer excluded"
    assert 9504 not in quoted, "the streaming RAG quoted a duplicate"
    assert 9502 not in quoted, "the streaming RAG quoted an article below the threshold"
    assert quoted == {9501, 9505}


def test_the_streaming_rag_reports_the_two_counts_separately(seeded, monkeypatch):
    """The meta event must distinguish the subset SEARCHED from the articles QUOTED."""
    if not _has_pgvector(seeded):
        pytest.skip("pgvector is not available on this database")

    import json

    with seeded.cursor() as cur:
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding vector(1536)")
        cur.execute("UPDATE document_chunk SET embedding = %s WHERE document_id = ANY(%s)",
                    ("[" + ",".join(["0.01"] * 1536) + "]", list(IDS)))

    class _FakeOpenAI:
        def __init__(self, *a, **kw):
            self.embeddings = type("E", (), {
                "create": lambda _s, **kw: type("R", (), {
                    "data": [type("D", (), {"embedding": [0.01] * 1536})()]})()})()

    import llm_usage

    monkeypatch.setattr(llm_usage, "MeteredOpenAI", _FakeOpenAI)

    from fastapi.testclient import TestClient

    body = TestClient(main.app).post(
        "/ask/stream/filtered",
        json={"question": "dengue transmission", "scenario_id": SID}).text
    meta = next((json.loads(line[6:]) for line in body.splitlines()
                 if line.startswith("data: {") and "papers_used" in line), None)
    assert meta is not None, f"no meta event in the stream: {body[:400]}"
    assert meta["papers_used"] == 2          # the relevant subset: 9501 and 9505
    assert meta["papers_quoted"] == 2        # and the answer reproduces both
    assert meta["digest_complete"] is True   # the figures rest on the whole-corpus digest
