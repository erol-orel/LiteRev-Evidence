"""Every extraction reads ALL the relevant articles, never a sample (api/digest.py).

The house rule: relevant means above the similarity threshold OR included by a reviewer,
never the excluded. A corpus of thousands cannot be pasted into one prompt, so the rule is
kept by map then reduce: per-article facts (PICO, concepts) are extracted once and cached
on the article row, and the digest aggregates them over the WHOLE relevant subset in SQL.
The generators write over that digest, with a handful of articles reproduced only to quote.

These tests pin the two halves: the digest counts the whole subset (integration), and the
generators' prompts actually carry it (the LLM client is stubbed)."""
import json

import main
from conftest import ensure_document_columns, patch_app  # noqa: E402

SID = "usr-digest-test"


def _seed(db_conn, n_relevant=6):
    """A corpus with every case: above threshold, below, excluded, and a below-threshold
    article included by hand (which must count as relevant)."""
    ensure_document_columns(db_conn.cursor())
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 8100 AND 8199")
        cur.execute("INSERT INTO user_scenarios (id, name, query, created_at, updated_at) "
                    "VALUES (%s, 'Dengue', 'dengue', NOW(), NOW()) ON CONFLICT (id) DO NOTHING", (SID,))
        pico = json.dumps({"P": "residents", "O": "cases", "study_design": "Cohort"})
        concepts = json.dumps({"v": 1, "concepts": [{"t": "pathogen", "en": "dengue virus", "fr": "virus de la dengue"}]})
        # 4 above threshold, one of them with concepts and PICO
        for i in range(8100, 8104):
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, country, journal, "
                        "study_design, quality_score, citation_count, source, project_context) VALUES "
                        "(%s,%s,'abstract',2024,'IT','Eurosurveillance','Cohort',0.8,10,'pubmed','literev')",
                        (i, f"Relevant article {i}"))
            cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                        "VALUES (%s,%s,0.80)", (i, SID))
        cur.execute("UPDATE literature_document SET pico_json = %s, concepts_json = %s WHERE id IN (8100, 8101)",
                    (pico, concepts))
        # below threshold, NOT relevant
        cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                    "quality_score) VALUES (8110,'Below threshold','a',2020,'pubmed','literev',0.5)")
        cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                    "VALUES (8110,%s,0.10)", (SID,))
        # below threshold but INCLUDED by a reviewer: relevant
        cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                    "quality_score) VALUES (8111,'Hand picked','a',2019,'pubmed','literev',0.5)")
        cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score, screening_status) "
                    "VALUES (8111,%s,0.10,'included')", (SID,))
        # above threshold but EXCLUDED by a reviewer: not relevant
        cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                    "quality_score) VALUES (8112,'Thrown out','a',2024,'pubmed','literev',0.9)")
        cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score, screening_status) "
                    "VALUES (8112,%s,0.95,'excluded')", (SID,))


def _cleanup(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 8100 AND 8199")
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def test_the_digest_covers_the_whole_relevant_subset(db_conn):
    _seed(db_conn)
    try:
        d = main.corpus_digest(SID, threshold=0.45)
        # 4 above threshold + 1 included by hand; the excluded one and the below-threshold
        # one are out. This is the count every generator must speak for.
        assert d["n_articles"] == 5
        assert d["n_included"] == 1
        assert d["n_with_pico"] == 2 and d["n_with_concepts"] == 2
        assert d["year_min"] == 2019 and d["year_max"] == 2024
        assert d["complete"] is True
        assert isinstance(d["mean_quality"], float)          # JSON serialisable, not Decimal
        assert ("IT", 4) in [(r["value"], r["n"]) for r in d["by_country"]]
        assert d["concepts"]["pathogen"][0] == {"label": "dengue virus", "n": 2}

        block = main.digest_to_prompt(d)
        assert "CORPUS COMPLET: 5 articles pertinents" in block
        assert "dengue virus (2)" in block
        assert "Eurosurveillance (4)" in block

        note = main.digest_coverage_note(d, 30)
        assert "TOTALITE des 5 articles" in note and "30 articles reproduits" in note
    finally:
        _cleanup(db_conn)


def test_an_empty_corpus_yields_an_empty_block():
    assert main.digest_to_prompt({}) == ""
    assert main.digest_to_prompt({"n_articles": 0}) == ""


class _CapturedLLM:
    """Stands in for the OpenAI client and keeps the prompt it was handed."""

    def __init__(self, payload: dict):
        self.payload, self.prompts = payload, []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.prompts.append("\n".join(m["content"] for m in kw["messages"]))
                msg = type("M", (), {"content": json.dumps(outer.payload)})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})
        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_the_actions_prompt_carries_the_whole_corpus_not_the_sample(monkeypatch):
    """The generator reproduces 20 articles at most; the prompt must still state the real
    total and forbid concluding from those 20 alone."""
    import llm_usage

    sample = [{"id": i, "title": f"Article {i}", "pico_json": {"O": "cases"}} for i in range(1, 21)]
    patch_app(monkeypatch, "_get_above_threshold_articles", lambda sid, *a, **k: sample)
    patch_app(monkeypatch, "_get_scenario_name", lambda sid: "Dengue in Europe")
    patch_app(monkeypatch, "corpus_digest", lambda sid, thr=None: {
        "n_articles": 2732, "n_included": 12, "n_with_pico": 2700, "n_with_fulltext": 400,
        "threshold": 0.45, "year_min": 1998, "year_max": 2026, "mean_quality": 0.61,
        "total_citations": 90000, "by_design": [{"value": "cohort", "n": 800}],
        "by_country": [], "by_journal": [], "by_year": [], "concepts": {}, "complete": True})
    captured = _CapturedLLM({"recommended_actions": ["a", "b", "c", "d"]})
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", lambda **kw: captured)
    written = {}
    patch_app(monkeypatch, "engine", main.engine)            # unchanged, kept explicit
    monkeypatch.setattr(main, "_maybe_generate_actions", lambda *a, **k: True, raising=False)

    class _Conn:
        def execute(self, *a, **k): written["saved"] = True
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(main.engine, "begin", lambda: _Conn())

    actions = main._generate_recommended_actions("usr-x", lang="en")
    assert actions == ["a", "b", "c", "d"] and written.get("saved")
    prompt = captured.prompts[0]
    assert "CORPUS COMPLET: 2732 articles pertinents" in prompt
    assert "TOTALITE des 2732 articles" in prompt
    assert "20 articles reproduits" in prompt
    assert "corpus complet" in prompt.lower()


def test_no_extraction_path_keeps_a_default_article_cap():
    """The caps that used to sample are off by default: the extractions read everything.
    A positive value stays available as an operational fallback."""
    assert main.EPI_PARAM_MAX_ARTICLES == 0
    assert main.CONCEPT_MAX_ARTICLES == 0
    assert main.CONCEPT_GRAPH_MAX_ARTICLES == 0
