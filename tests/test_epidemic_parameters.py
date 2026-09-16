"""Targeted extraction of the epidemiological parameters (api/variables.py).

The model spec is built from the 25 most relevant articles; on a corpus of thousands,
those almost never report an R0 or an incubation period, so the SEIR tab said "no
parameter extracted" for a chikungunya corpus of 2,732 articles. The corpus is now
searched for the articles that MEASURE a parameter, and only those are extracted.

The term matching and the merge are pure; the candidate query and the endpoints run
against the test database (skipped when none is reachable)."""
import main
from conftest import ensure_document_columns, patch_app  # noqa: E402

SID = "usr-epi-test"


def test_measurement_terms_are_recognised_without_false_positives():
    assert main.params_mentioned("Estimating the basic reproduction number of dengue") == ["r0"]
    assert main.params_mentioned("R0 and Rt during the outbreak") == ["r0"]
    assert main.params_mentioned("Nombre de reproduction du chikungunya") == ["r0"]
    assert main.params_mentioned("Serial interval and incubation period of mpox") == [
        "serial_interval_days", "incubation_period_days"]
    assert main.params_mentioned("Case-fatality ratio in hospitalised patients") == ["cfr"]
    assert main.params_mentioned("Waning immunity after vaccination") == ["immunity_duration_days"]
    # A short token never matches inside a word, and an unrelated paper matches nothing.
    assert main.params_mentioned("Macro0 bed occupancy model") == []
    assert main.params_mentioned("Larvicide programmes against Aedes albopictus") == []


def test_the_merge_adds_measured_observations_and_proves_transmissibility():
    narrative = {"applicable": False, "r0": {"value": None, "observations": [{"article_id": 7, "value": 2.0}]}}
    targeted = {
        "params": {
            "r0": {"observations": [{"article_id": 7, "value": 2.0},          # duplicate, kept once
                                    {"article_id": 8, "value": 2.4, "ci_low": 2.0, "ci_high": 2.9}]},
            "incubation_period_days": {"observations": [{"article_id": 9, "value": 3.0}]},
        },
        "disease": "chikungunya",
    }
    merged = main.merge_epidemic_observations(narrative, targeted)
    assert [o["value"] for o in merged["r0"]["observations"]] == [2.0, 2.4]
    assert merged["r0"]["provenance"] == [7, 8] and merged["r0"]["n_studies"] == 2
    assert merged["r0"]["unit"] == "ratio" and merged["incubation_period_days"]["unit"] == "days"
    # A measured transmission parameter settles the question the first pass got wrong.
    assert merged["applicable"] is True
    assert merged["population_disease"] == "chikungunya"

    # A case fatality alone does not make a scenario transmissible.
    only_cfr = main.merge_epidemic_observations(
        {"applicable": False}, {"params": {"cfr": {"observations": [{"article_id": 1, "value": 0.015}]}}})
    assert only_cfr.get("applicable") is False
    assert only_cfr["cfr"]["unit"] == "proportion"

    # Nothing measured: the narrative block comes back untouched.
    assert main.merge_epidemic_observations({"applicable": True, "r0": {"value": 3.0}}, {"params": {}}) == {
        "applicable": True, "r0": {"value": 3.0}}


def test_extraction_keeps_only_real_values_from_real_articles(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    patch_app(monkeypatch, "_epi_llm_client", lambda: object())          # no SDK, no network
    patch_app(monkeypatch, "_parameter_candidate_articles", lambda sid, thr=None, cap=40: [
        {"id": 1, "title": "R0 of chikungunya", "abstract": "...", "quality_score": 0.9,
         "year": 2024, "doi": "10.1/a", "study_design": "Systematic review", "source": "pubmed",
         "params_mentioned": ["r0"]},
        {"id": 2, "title": "Incubation period", "abstract": "...", "quality_score": 0.7,
         "year": 2023, "doi": None, "study_design": "Cohort", "source": "pubmed",
         "params_mentioned": ["incubation_period_days"]},
    ])
    patch_app(monkeypatch, "_epi_extract_batch", lambda client, batch, hint: [
        {"id": 1, "disease": "chikungunya", "parameters": [
            {"name": "r0", "value": 2.1, "ci_low": 1.8, "ci_high": 2.6},
            {"name": "cfr", "value": "not reported"},              # not a number: dropped
            {"name": "attack_rate", "value": 0.3}]},               # unknown parameter: dropped
        {"id": 2, "disease": "chikungunya", "parameters": [{"name": "incubation_period_days", "value": 3.0}]},
        {"id": 999, "parameters": [{"name": "r0", "value": 9.9}]},  # id not in the corpus: dropped
    ])
    out = main.extract_epidemic_observations(SID, disease_hint="Chikungunya")
    assert sorted(out["params"]) == ["incubation_period_days", "r0"]
    assert out["params"]["r0"]["observations"] == [
        {"article_id": 1, "value": 2.1, "ci_low": 1.8, "ci_high": 2.6}]
    assert out["n_candidates"] == 2 and out["n_with_values"] == 2
    assert out["disease"] == "chikungunya"
    assert [a["id"] for a in out["articles"]] == [1, 2]          # provenance pool for the spec


def test_without_an_openai_key_the_candidates_are_still_counted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    patch_app(monkeypatch, "_parameter_candidate_articles", lambda sid, thr=None, cap=40: [
        {"id": 1, "title": "t", "abstract": "a", "params_mentioned": ["r0"], "quality_score": 0.5,
         "year": 2024, "doi": None, "study_design": None, "source": None}])
    out = main.extract_epidemic_observations(SID)
    assert out["n_candidates"] == 1 and out["params"] == {} and out["n_with_values"] == 0


def _seed(db_conn):
    ensure_document_columns(db_conn.cursor())
    rows = [
        (9001, "Estimating the basic reproduction number of chikungunya", "R0 was 2.1.", "Systematic review", 0.9, 120),
        (9002, "Incubation period of chikungunya virus", "Median incubation period 3 days.", "Cohort", 0.7, 40),
        (9003, "Vector control against Aedes albopictus", "Larvicide programmes only.", "Cross-sectional", 0.6, 10),
        (9004, "Macro0 hospital bed occupancy", "Capacity model.", "Modelling", 0.95, 3),
    ]
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", ([r[0] for r in rows],))
        cur.execute("INSERT INTO user_scenarios (id, name, query, created_at, updated_at) "
                    "VALUES (%s, 'Chikungunya', 'chikungunya', NOW(), NOW()) ON CONFLICT (id) DO NOTHING", (SID,))
        for i, t, a, d, q, c in rows:
            cur.execute("INSERT INTO literature_document (id, title, abstract, study_design, quality_score, "
                        "citation_count, source, project_context, year) "
                        "VALUES (%s,%s,%s,%s,%s,%s,'pubmed','literev',2024)", (i, t, a, d, q, c))
            cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                        "VALUES (%s,%s,0.8)", (i, SID))


def test_the_corpus_query_finds_the_measuring_articles_only(db_conn):
    _seed(db_conn)
    cands = main._parameter_candidate_articles(SID)
    assert [c["id"] for c in cands] == [9001, 9002]             # the review first, then the cohort
    assert cands[0]["params_mentioned"] == ["r0"]
    # 9004 has the highest quality score but says nothing about a parameter, and "Macro0"
    # must not be read as R0; 9003 reports none either.
    assert 9003 not in [c["id"] for c in cands] and 9004 not in [c["id"] for c in cands]

    from fastapi.testclient import TestClient
    body = TestClient(main.app).get(f"/scenarios/{SID}/epidemic-parameters/candidates").json()
    assert body["n_candidates"] == 2
    assert body["by_parameter"] == {"r0": 1, "incubation_period_days": 1}
    assert body["articles"][0]["parameters"] == ["r0"]
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9001 AND 9004")
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
