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

    class _Result:
        """La génération LIT le cache des actions avant d'y écrire sa langue : un
        emplacement par langue, au lieu d'un seul que l'autre langue écrasait."""
        def mappings(self): return self
        def first(self): return None

    class _Conn:
        def execute(self, *a, **k):
            written["saved"] = True
            return _Result()
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
    assert main.EXTRACTION_MAX_ARTICLES == 0
    assert main.FINDINGS_MAX_ARTICLES == 0


# ── What the studies FOUND, counted over the whole relevant subset ───────────
# The digest counted designs, countries and concepts, never results: whatever a generator
# said about findings came from the 20 to 30 articles it reproduces. Each article now
# carries its main finding and its direction (api/pico_map), and the digest counts the
# directions over every relevant article, in SQL.

def _finding(direction: str, outcome: str = "cases") -> str:
    return json.dumps({"P": "residents", "O": outcome, "study_design": "Cohort",
                       "key_finding": f"A finding ({direction}).", "finding_direction": direction})


def test_the_digest_counts_the_findings_of_every_relevant_article(db_conn):
    _seed(db_conn)
    try:
        outcome = json.dumps({"v": 1, "concepts": [{"t": "outcome", "en": "mortality", "fr": "mortalite"}]})
        with db_conn.cursor() as cur:
            for doc_id, direction in ((8100, "increase"), (8101, "increase"), (8102, "decrease"),
                                      (8111, "no_difference"),
                                      # below the threshold and excluded: never counted
                                      (8110, "increase"), (8112, "increase")):
                cur.execute("UPDATE literature_document SET pico_json = %s, concepts_json = %s WHERE id = %s",
                            (_finding(direction), outcome, doc_id))
            # 8103 is relevant and was never read for its finding.
        d = main.corpus_digest(SID, threshold=0.45)
        f = d["findings"]
        assert (f["n_read"], f["n_with_finding"]) == (4, 4)
        assert f["by_direction"] == {"increase": 2, "decrease": 1, "no_difference": 1,
                                     "mixed": 0, "descriptive": 0, "none": 0}
        assert f["outcome_basis"] == "concepts" and f["n_outcome_labels"] == 1
        assert f["by_outcome"] == [{"label": "mortality", "n": 4, "n_read": 4, "by_direction": {
            "increase": 2, "decrease": 1, "no_difference": 1, "mixed": 0, "descriptive": 0, "none": 0}}]

        block = main.digest_to_prompt(d)
        assert "RESULTATS sur les 5 articles pertinents" in block
        assert "4 lus, dont 4 rapportent un resultat" in block
        assert "hausse 2, baisse 1, sans difference 1, mixte 0" in block
        assert "1 articles ne sont pas encore lus pour leur resultat" in block
        assert "mortality (4 articles: hausse 2, baisse 1, sans difference 1)" in block
    finally:
        _cleanup(db_conn)


def test_without_outcome_concepts_the_pico_outcome_is_grouped(db_conn):
    _seed(db_conn)
    try:
        with db_conn.cursor() as cur:
            cur.execute("UPDATE literature_document SET pico_json = %s WHERE id = 8100", (_finding("increase", "Mortality; ICU"),))
            cur.execute("UPDATE literature_document SET pico_json = %s WHERE id = 8101", (_finding("decrease", "mortality (28 days)"),))
        f = main.corpus_digest(SID, threshold=0.45)["findings"]
        assert f["outcome_basis"] == "pico_outcome"
        assert f["by_outcome"][0]["label"] == "mortality" and f["by_outcome"][0]["n"] == 2
        assert f["by_outcome"][0]["by_direction"]["increase"] == 1
    finally:
        _cleanup(db_conn)


def test_a_corpus_read_for_nothing_forbids_a_trend_instead_of_staying_silent():
    d = {"n_articles": 40, "complete": True, "findings": {"n_read": 0, "n_with_finding": 0,
                                                          "by_direction": {}, "by_outcome": []}}
    assert "aucun des 40 articles pertinents n'a encore ete lu" in main.digest_to_prompt(d)
    d["findings"] = {"error": "boom"}
    assert "le decompte des resultats sur le corpus complet a echoue" in main.digest_to_prompt(d)


def _fake_engine_begin(monkeypatch):
    """The generators write their result to scenario_settings: a connection that accepts
    the statements and returns nothing."""
    class _Result:
        def mappings(self): return self
        def first(self): return None

    class _Conn:
        def execute(self, *a, **k): return _Result()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(main.engine, "begin", lambda: _Conn())


_DIGEST = {
    "n_articles": 2732, "n_included": 12, "n_with_pico": 2700, "n_with_fulltext": 400,
    "threshold": 0.45, "year_min": 1998, "year_max": 2026, "mean_quality": 0.61,
    "n_with_quality": 2000, "total_citations": 90000, "n_with_citations": 2500,
    "by_design": [{"value": "cohort", "n": 800}], "by_country": [], "by_journal": [],
    "by_year": [], "concepts": {}, "complete": True,
    "findings": {"n_read": 2600, "n_with_finding": 2400, "outcome_basis": "concepts",
                 "by_direction": {"increase": 900, "decrease": 500, "no_difference": 400,
                                  "mixed": 100, "descriptive": 500, "none": 200},
                 "by_outcome": [], "n_outcome_labels": 0},
}


def test_the_brief_prompt_carries_the_findings_and_the_extraction_block(monkeypatch):
    """The brief writes over the digest, now with the findings counted over the whole
    corpus, and over the structured extraction counts the assistant already received.
    The reproduced articles carry their key finding, which no writer used to store."""
    import llm_usage

    sample = [{"id": i, "title": f"Article {i}", "year": 2020, "study_design": "Cohort study",
               "abstract": "a", "has_pico": True,
               "pico_json": {"O": "cases", "key_finding": f"Finding number {i}.", "finding_direction": "increase"}}
              for i in range(1, 41)]
    patch_app(monkeypatch, "_get_above_threshold_articles", lambda sid, *a, **k: sample)
    patch_app(monkeypatch, "_get_scenario_name", lambda sid: "Dengue in Europe")
    patch_app(monkeypatch, "_get_scenario_threshold", lambda sid: 0.45)
    patch_app(monkeypatch, "corpus_digest", lambda sid, thr=None: dict(_DIGEST))
    patch_app(monkeypatch, "extraction_digest", lambda sid, thr=None: {
        "complete": True, "n_extracted": 12, "n_relevant": 2732, "n_fulltext": 5, "n_abstract": 7,
        "n_unread": 2720, "coverage": {"sex_gender": 3}, "coverage_fulltext": {"sex_gender": 2}})
    captured = _CapturedLLM({"executive_summary": "x", "claims": []})
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", lambda **kw: captured)
    _fake_engine_begin(monkeypatch)

    brief = main._generate_evidence_brief_llm("usr-x", force=True, lang="en")
    assert "error" not in brief, brief
    prompt = captured.prompts[0]
    assert "RESULTATS sur les 2732 articles pertinents" in prompt
    assert "hausse 900, baisse 500, sans difference 400" in prompt
    assert "EXTRACTION STRUCTUREE: 12 des 2732 articles pertinents" in prompt
    assert "denominateur (12 articles extraits sur 2732)" in prompt
    assert "Finding number 7." in prompt and "Finding number 31." not in prompt   # 30 reproduced
    assert "TOTALITE des 2732 articles" in prompt and "30 articles reproduits" in prompt
    assert "mieux" not in prompt and "les plus proches de la question" in prompt


def test_the_actions_prompt_reproduces_each_article_whole_with_its_finding(monkeypatch):
    """Twenty articles were promised and a JSON cut at 6,000 characters delivered about
    seventeen, the last one torn mid-string, each with an empty key finding."""
    import llm_usage

    long_title = "A long title about vector control and dengue in southern Europe " * 4
    sample = [{"id": i, "title": f"{long_title}{i}", "year": 2021, "study_design": "Cohort study",
               "pico_json": {"O": "dengue incidence", "key_finding": f"Incidence fell by {i}%.",
                             "finding_direction": "decrease"}} for i in range(1, 21)]
    patch_app(monkeypatch, "_get_above_threshold_articles", lambda sid, *a, **k: sample)
    patch_app(monkeypatch, "_get_scenario_name", lambda sid: "Dengue in Europe")
    patch_app(monkeypatch, "corpus_digest", lambda sid, thr=None: dict(_DIGEST))
    captured = _CapturedLLM({"recommended_actions": ["a", "b", "c", "d"]})
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", lambda **kw: captured)
    _fake_engine_begin(monkeypatch)

    assert main._generate_recommended_actions("usr-x", lang="en") == ["a", "b", "c", "d"]
    prompt = captured.prompts[0]
    for i in range(1, 21):
        assert f"- [{i}] {long_title}{i} (2021; Cohort study) | outcome: dengue incidence | " \
               f"résultat: Incidence fell by {i}%. [decrease]" in prompt
    assert "RESULTATS sur les 2732 articles pertinents" in prompt
    assert "mieux" not in prompt


def test_the_actions_are_asked_again_shorter_when_the_answer_is_cut(monkeypatch):
    import types

    import llm_usage

    calls: list = []

    def _create(**kw):
        calls.append(kw)
        content, finish = (('{"recommended_actions": ["a very long', "length") if len(calls) == 1
                           else ('{"recommended_actions": ["a", "b"]}', "stop"))
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish)])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_create)))
    patch_app(monkeypatch, "_get_above_threshold_articles",
              lambda sid, *a, **k: [{"id": 1, "title": "t", "pico_json": {"O": "x"}}])
    patch_app(monkeypatch, "_get_scenario_name", lambda sid: "Dengue")
    patch_app(monkeypatch, "corpus_digest", lambda sid, thr=None: dict(_DIGEST))
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", lambda **kw: client)
    _fake_engine_begin(monkeypatch)
    assert main._generate_recommended_actions("usr-x", lang="fr") == ["a", "b"]
    assert len(calls) == 2 and "CUT OFF" in calls[1]["messages"][0]["content"]
