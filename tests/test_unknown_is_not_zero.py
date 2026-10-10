"""An unknown is not a zero: the quality score and the citation count, as the brief reads them.

Found on the production scenario HPAI_last, prepared for a team that will act on it: the
English brief concluded "overall methodological quality is low, consistent with the
reported mean quality score of 0.08 and cumulative citations of zero". Both numbers were
artefacts of the pipeline, not facts about the literature.

- 0.08 was a mean over 1,000 relevant articles of which 861 had never been scored. The
  column carries a legacy 0.0 default, and the batch metadata endpoint, unlike the
  pipeline step, wrote `metadata_json` and nothing else: no design in its column, no
  sample size, no score. The 139 articles that had one averaged 0.57.
- zero was `SUM(COALESCE(citation_count, 0))` over 1,000 articles whose citation counts
  nobody had collected.

These tests pin the repairs: one metadata write for both paths, a digest whose mean
covers the scored articles only and says how many, citation counts summed over the known
ones or declared unknown, and a backfill that can be held to one scenario.
"""
import json

import pytest

import main
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-unknown-not-zero"
OTHER = "usr-unknown-not-zero-other"
IDS = tuple(range(9300, 9310))


# ── the metadata write (pure) ────────────────────────────────────────────────

def test_the_metadata_write_fills_the_columns_and_the_score():
    p = main._metadata_update_params(
        7, {"study_type": "Cohort", "sample_size": "1,200", "bias_risk": "low"},
        {"year": 2023, "citation_count": None, "open_access": True})
    assert p["article_id"] == 7
    assert p["study_design"] == "Cohort" and p["sample_size"] == 1200
    assert p["quality_score"] == main._compute_quality_score(
        study_design="Cohort", year=2023, sample_size=1200, citation_count=None,
        open_access=True, bias_risk="low")
    assert p["quality_score"] > 0
    assert json.loads(p["meta"])["study_type"] == "Cohort"


def test_what_the_row_already_knew_survives_an_extraction_that_says_nothing():
    p = main._metadata_update_params(8, {"study_type": None, "sample_size": None},
                                     {"study_design": "Case-control", "sample_size": 40, "year": 2020})
    assert p["study_design"] == "Case-control" and p["sample_size"] == 40
    assert p["quality_score"] > 0


def test_both_metadata_paths_share_the_one_write():
    """The batch endpoint and the pipeline step wrote two different UPDATEs, and only one
    of them computed the score. One write, called by both, cannot drift."""
    import inspect
    import pathlib

    import api.enrichment
    import api.pipeline

    for fn in (api.enrichment.extract_metadata_batch, api.pipeline._run_user_scenario_full_pipeline):
        src = inspect.getsource(fn)
        assert "METADATA_UPDATE_SQL" in src and "_metadata_update_params(" in src, fn.__name__
    api_dir = pathlib.Path(api.enrichment.__file__).resolve().parent
    writers = sorted(p.name for p in api_dir.glob("*.py")
                     if "SET metadata_json" in p.read_text(encoding="utf-8"))
    assert writers == ["documents.py"]


# ── the digest block (pure) ──────────────────────────────────────────────────

def _digest(**over):
    d = {"n_articles": 1000, "complete": True, "threshold": 0.58, "n_included": 0,
         "n_with_pico": 1000, "n_with_fulltext": 931, "year_min": 2000, "year_max": 2026,
         "mean_quality": 0.572, "n_with_quality": 139,
         "total_citations": None, "n_with_citations": 0}
    d.update(over)
    return d


def test_unknown_citations_are_declared_unknown_not_summed_to_zero():
    block = main.digest_to_prompt(_digest())
    assert "Citations: inconnues pour tous les articles" in block
    assert "pas nulles" in block
    assert "citations cumulees: 0" not in block and "0 au total" not in block


def test_known_citations_carry_their_denominator():
    block = main.digest_to_prompt(_digest(total_citations=1234, n_with_citations=250))
    assert "Citations: 1234 au total sur 250 articles dont le nombre est connu / 1000." in block


def test_the_quality_mean_says_what_it_is_and_how_many_articles_it_covers():
    block = main.digest_to_prompt(_digest())
    assert "moyenne 0.572 sur 139 articles notes / 1000" in block
    assert "pas une evaluation GRADE" in block


def test_a_corpus_without_scores_says_so_instead_of_staying_silent():
    block = main.digest_to_prompt(_digest(mean_quality=None, n_with_quality=0))
    assert "Score de qualite composite: non calcule pour ce corpus." in block


def test_the_block_never_carries_an_em_dash():
    assert chr(0x2014) not in main.digest_to_prompt(_digest(total_citations=5, n_with_citations=1))


# ── the database: digest, batch endpoint, scoped backfill ───────────────────

def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def corpus(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        # Runs alone on an empty database too: the base tables, then the columns
        # production has (same bootstrap as tests/test_rag_whole_corpus.py).
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created_chunk_table = ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("citation_count", "INTEGER"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS "
                    "similarity_score DOUBLE PRECISION")
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        _clean(cur)
        for sid in (SID, OTHER):
            cur.execute("INSERT INTO user_scenarios (id, name, query, created_at, updated_at) "
                        "VALUES (%s, 'HPAI', 'avian influenza', NOW(), NOW()) "
                        "ON CONFLICT (id) DO NOTHING", (sid,))
        # 9300 scored, 10 citations; 9301 the legacy 0.0 default and no citation count;
        # 9302 never written at all; 9303 scored, and a citation count that is a real 0.
        for doc_id, quality, cites in ((9300, 0.8, 10), (9301, 0.0, None),
                                       (9302, None, None), (9303, 0.6, 0)):
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, "
                        "project_context, quality_score, citation_count, open_access) "
                        "VALUES (%s, %s, 'an abstract long enough to be read', 2022, 'pubmed', "
                        "'literev', %s, %s, TRUE)", (doc_id, f"Article {doc_id}", quality, cites))
            cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                        "VALUES (%s, %s, 0.9)", (doc_id, SID))
    yield db_conn
    with db_conn.cursor() as cur:
        _clean(cur)
        if created_chunk_table:
            cur.execute("DROP TABLE IF EXISTS document_chunk")


def _clean(cur):
    cur.execute("DELETE FROM article_scenarios WHERE scenario_id IN (%s, %s)", (SID, OTHER))
    cur.execute("DELETE FROM literature_document WHERE id BETWEEN %s AND %s", (IDS[0], IDS[-1]))
    cur.execute("DELETE FROM scenario_settings WHERE scenario_id IN (%s, %s)", (SID, OTHER))
    cur.execute("DELETE FROM user_scenarios WHERE id IN (%s, %s)", (SID, OTHER))


def test_the_digest_averages_the_scored_and_sums_the_known(corpus):
    d = main.corpus_digest(SID, threshold=0.45)
    assert d["complete"] is True and d["n_articles"] == 4
    # 0.8 and 0.6: the legacy 0.0 and the NULL are "never scored", not "scored zero".
    assert d["mean_quality"] == 0.7 and d["n_with_quality"] == 2
    # 10 and a known 0; the two unknowns are left out of the sum AND of the count.
    assert d["total_citations"] == 10 and d["n_with_citations"] == 2
    block = main.digest_to_prompt(d)
    assert "moyenne 0.7 sur 2 articles notes / 4" in block
    assert "Citations: 10 au total sur 2 articles dont le nombre est connu / 4." in block
    json.dumps(d)                                   # serialisable: no Decimal left behind


def test_a_corpus_with_no_citation_count_has_no_citation_total(corpus, db_conn):
    with db_conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET citation_count = NULL WHERE id BETWEEN %s AND %s",
                    (IDS[0], IDS[-1]))
    d = main.corpus_digest(SID, threshold=0.45)
    assert d["total_citations"] is None and d["n_with_citations"] == 0
    assert "inconnues pour tous les articles" in main.digest_to_prompt(d)


class _MetadataLLM:
    """Stands in for the OpenAI client: every article is a cross-sectional study of 450."""

    def __init__(self):
        payload = {"study_type": "Cross-sectional", "sample_size": 450, "country": "IT",
                   "bias_risk": "moderate", "metadata_confidence": 0.9}
        msg = type("M", (), {"content": json.dumps(payload)})

        class _Completions:
            def create(self, **kw):
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})
        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_the_batch_endpoint_scores_what_it_extracts(corpus, db_conn, monkeypatch):
    import llm_usage

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", lambda **kw: _MetadataLLM())
    out = main.extract_metadata_batch(scenario_id=SID, limit=100000, scope="all")
    assert out["extracted"] == 4 and out["errors"] == 0
    with db_conn.cursor() as cur:
        cur.execute("SELECT id, study_design, sample_size, quality_score, metadata_json "
                    "FROM literature_document WHERE id BETWEEN %s AND %s ORDER BY id", (IDS[0], IDS[-1]))
        rows = cur.fetchall()
    assert len(rows) == 4
    for doc_id, design, sample, quality, meta in rows:
        assert design == "Cross-sectional" and sample == 450, doc_id
        assert meta["country"] == "IT"
        assert quality is not None and quality > 0, doc_id


def test_the_backfill_can_be_held_to_one_scenario(corpus, db_conn):
    with db_conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET metadata_json = %s WHERE id = 9301",
                    (json.dumps({"study_type": "Cohort", "bias_risk": "low"}),))
        # An unscored article of ANOTHER scenario: the scoped backfill must leave it alone.
        cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, "
                    "project_context, quality_score, open_access) VALUES (9309, 'Elsewhere', "
                    "'an abstract long enough to be read', 2022, 'pubmed', 'literev', 0.0, TRUE)")
        cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                    "VALUES (9309, %s, 0.9)", (OTHER,))
    out = main.recompute_quality_scores(limit=5000, only_missing=True, scenario_id=SID)
    assert out["scanned"] == 2                      # 9301 (0.0) and 9302 (NULL), nothing else
    with db_conn.cursor() as cur:
        cur.execute("SELECT id, quality_score, study_design FROM literature_document "
                    "WHERE id IN (9301, 9302, 9309) ORDER BY id")
        got = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    assert got[9301][0] > 0 and got[9301][1] == "Cohort"
    assert got[9302][0] > 0                         # year and open access are signals too
    assert got[9309][0] == 0.0                      # the other scenario is untouched
