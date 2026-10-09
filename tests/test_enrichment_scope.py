"""An enrichment is paid for one article at a time, so it must say which ones.

The three batch jobs (PICO, metadata, full text) each filtered their own way: PICO
excluded the articles a reviewer had set aside, metadata and full text did not, and
none of the three knew about the relevance threshold. Run on a scenario holding six
thousand five hundred references of which four hundred and sixty-seven pass the
threshold, they all billed fourteen times the useful batch for the same answer.

So a scope, written once (`scenario_scope_sql`) over the gate the rest of the
application already shares, and counters that announce what a run would process
before anyone presses the button.
"""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.scenario_store import SCOPES, relevant_gate_sql, scenario_scope_sql  # noqa: E402


# ── the predicate (pure) ─────────────────────────────────────────────────────

def test_the_two_scopes_are_the_whole_vocabulary():
    assert SCOPES == ("all", "relevant")


def test_an_unknown_scope_is_refused_rather_than_silently_widened():
    """A typo must not quietly enrich six thousand articles instead of five hundred."""
    with pytest.raises(ValueError) as e:
        scenario_scope_sql("relevent")
    assert "relevent" in str(e.value)


def test_the_relevant_scope_is_the_shared_gate_and_not_a_fourth_copy():
    """The condition that defines "relevant" is written in exactly one place."""
    sql = scenario_scope_sql("relevant", doc="ld", link="asn")
    threshold = ("COALESCE((SELECT ss.similarity_threshold FROM scenario_settings ss"
                 " WHERE ss.scenario_id = :sid), 0.45)")
    assert sql == relevant_gate_sql("ld", "asn", threshold)


def test_the_relevant_scope_reads_the_threshold_inside_the_statement():
    """Passing it as a parameter would reopen two readings at two instants."""
    sql = scenario_scope_sql("relevant")
    assert "SELECT ss.similarity_threshold FROM scenario_settings" in sql
    assert "0.45" in sql                      # the default when the scenario stored none


def test_the_whole_scenario_scope_still_drops_duplicates_and_excluded_articles():
    """Widest is not "everything": nobody pays a model for a duplicate."""
    sql = scenario_scope_sql("all", doc="ld", link="asn")
    assert "ld.is_duplicate IS NOT TRUE" in sql
    assert "IS DISTINCT FROM 'excluded'" in sql
    assert "similarity_threshold" not in sql   # and it ignores the threshold, by design


def test_relevant_is_strictly_narrower_than_all():
    """The relevant scope adds a condition, it never removes one.

    Compared clause by clause rather than by string prefix: both now read the status of
    THIS review (`asn.screening_status`) instead of falling back to the document row
    shared with every other scenario, so `relevant` carries the same two clauses as
    `all` plus the two thresholds."""
    wide, narrow = scenario_scope_sql("all"), scenario_scope_sql("relevant")
    for clause in wide.split(" AND "):
        assert clause in narrow, clause
    assert "similarity_score" in narrow and "rerank_score" in narrow
    assert len(narrow) > len(wide)


def test_the_counters_cover_both_scopes_and_the_three_jobs():
    sql = main._scope_counts_sql()
    for scope in SCOPES:
        assert f"AS {scope}_total" in sql
        for job in ("pico", "metadata", "fulltext"):
            assert f"AS {scope}_{job}_done" in sql
            assert f"AS {scope}_{job}_todo" in sql
    # Une seule instruction : le total annoncé et le reste à faire viennent du
    # même instantané, sans quoi le panneau annoncerait un chiffre et en
    # traiterait un autre.
    assert sql.count("SELECT COUNT") <= 1


def test_the_pico_batch_bounds_its_retries_in_the_todo_count():
    """An article that has already failed three times is not billed a fourth."""
    assert "pico_attempts" in main._scope_counts_sql()


# ── the endpoint (database) ──────────────────────────────────────────────────

SID = "enrich-scope"


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
        cur.execute("DROP TABLE IF EXISTS document_search")
        cur.execute("DROP TABLE IF EXISTS document_chunk, article_scenarios, literature_document CASCADE")
        cur.execute(
            "CREATE TABLE literature_document ("
            "id bigint PRIMARY KEY, title text NOT NULL, source text NOT NULL DEFAULT 'pubmed',"
            "abstract text, year int, is_duplicate boolean DEFAULT false, screening_status text,"
            "doi text, has_fulltext boolean, pico_json jsonb, pico_attempts int DEFAULT 0,"
            "metadata_json jsonb, journal text, created_at timestamptz DEFAULT now(),"
            "project_context text DEFAULT 'literev')")
        cur.execute(
            "CREATE TABLE article_scenarios (scenario_id text, document_id bigint,"
            "similarity_score double precision, rerank_score double precision,"
            "screening_status text, PRIMARY KEY (scenario_id, document_id))")
        # « Texte intégral fait » veut dire qu'on DÉTIENT le texte : un morceau en base.
        # Le drapeau `has_fulltext` répondait « un lien d'accès ouvert existe », posé dès
        # qu'Unpaywall rend une adresse, sans qu'une ligne de texte soit stockée.
        cur.execute(
            "CREATE TABLE document_chunk ("
            "id bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,"
            "document_id bigint NOT NULL REFERENCES literature_document(id) ON DELETE CASCADE,"
            "chunk_index int DEFAULT 0, content text, chunk_type text, embedding text)")
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute(
            "INSERT INTO user_scenarios (id, name, query, mode, filters) "
            "VALUES (%s, 'Scope', 'rsv AND wastewater', 'boolean', '{}')", (SID,))
        # 1 et 2 au-dessus du seuil ; 3 dessous ; 4 doublon ; 5 écarté par un
        # relecteur bien qu'au-dessus ; 6 repêché à la main bien que dessous.
        cur.execute(
            "INSERT INTO literature_document (id, title, abstract, is_duplicate, doi) VALUES "
            "(1, 'A', 'abstract one long enough to count', false, '10.1/a'),"
            "(2, 'B', 'abstract two long enough to count', false, '10.1/b'),"
            "(3, 'C', 'abstract three long enough to count', false, '10.1/c'),"
            "(4, 'D', 'a flagged duplicate', true, '10.1/d'),"
            "(5, 'E', 'set aside by a reviewer', false, '10.1/e'),"
            "(6, 'F', 'rescued by a reviewer', false, NULL)")
        cur.execute(
            "INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, screening_status) VALUES "
            "(%s, 1, 0.90, NULL), (%s, 2, 0.60, NULL), (%s, 3, 0.10, NULL),"
            "(%s, 4, 0.95, NULL), (%s, 5, 0.95, 'excluded'), (%s, 6, 0.01, 'included')",
            (SID, SID, SID, SID, SID, SID))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))


def test_the_status_counts_both_scopes_from_one_statement(seeded):
    s = main.get_enrichment_status(scenario_id=SID)
    by = s["by_scope"]
    # Tout le scénario : les six moins le doublon et l'article écarté.
    assert by["all"]["total"] == 4
    # Pertinents : les deux au-dessus du seuil, plus le repêché à la main.
    assert by["relevant"]["total"] == 3


def test_the_rescued_article_is_relevant_and_the_excluded_one_is_in_neither(seeded):
    """A reviewer's decision outranks the score, both ways round."""
    by = main.get_enrichment_status(scenario_id=SID)["by_scope"]
    assert by["relevant"]["total"] == 3 and by["all"]["total"] == 4
    assert by["relevant"]["total"] < by["all"]["total"]


def test_the_requested_scope_is_what_the_top_level_fields_describe(seeded):
    wide = main.get_enrichment_status(scenario_id=SID, scope="all")
    narrow = main.get_enrichment_status(scenario_id=SID, scope="relevant")
    assert wide["scope"] == "all" and narrow["scope"] == "relevant"
    assert wide["total"] == 4 and narrow["total"] == 3
    # Et les deux portent quand même les deux portées, pour annoncer l'autre choix.
    assert wide["by_scope"] == narrow["by_scope"]


def test_the_default_scope_leaves_the_previous_answer_unchanged(seeded):
    """Callers that never heard of a scope keep the behaviour they had."""
    assert main.get_enrichment_status(scenario_id=SID)["scope"] == "all"


def test_the_todo_count_is_what_a_run_would_actually_process(seeded, db_conn):
    before = main.get_enrichment_status(scenario_id=SID, scope="relevant")
    assert before["pico"]["todo"] == 3 and before["pico"]["count"] == 0
    with db_conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET pico_json = %s WHERE id = 1",
                    ('{"P": "x", "pico_confidence": 0.9}',))
    after = main.get_enrichment_status(scenario_id=SID, scope="relevant")
    assert after["pico"]["count"] == 1 and after["pico"]["todo"] == 2


def test_a_low_confidence_extraction_is_counted_as_still_to_do(seeded, db_conn):
    with db_conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET pico_json = %s WHERE id = 1",
                    ('{"P": "x", "pico_confidence": 0.2}',))
    s = main.get_enrichment_status(scenario_id=SID, scope="relevant")
    assert s["pico"]["todo"] == 3


def test_an_article_that_failed_three_times_is_not_billed_again(seeded, db_conn):
    with db_conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET pico_attempts = 3 WHERE id = 1")
    s = main.get_enrichment_status(scenario_id=SID, scope="relevant")
    assert s["pico"]["todo"] == 2


def test_full_text_only_counts_what_it_could_fetch(seeded):
    """No DOI, nothing to try: article 6 is relevant and still not a candidate."""
    s = main.get_enrichment_status(scenario_id=SID, scope="relevant")
    assert s["fulltext"]["todo"] == 2      # 1 and 2 have a DOI, 6 does not


def test_the_scope_follows_the_scenario_threshold(seeded, db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO scenario_settings (scenario_id, similarity_threshold) VALUES (%s, 0.55) "
            "ON CONFLICT (scenario_id) DO UPDATE SET similarity_threshold = 0.55", (SID,))
    s = main.get_enrichment_status(scenario_id=SID, scope="relevant")
    assert s["total"] == 3                 # 1 (0.90), 2 (0.60) and the rescued 6
    with db_conn.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET similarity_threshold = 0.80 WHERE scenario_id = %s", (SID,))
    assert main.get_enrichment_status(scenario_id=SID, scope="relevant")["total"] == 2


def test_an_unknown_scope_is_a_four_hundred_not_a_wider_run(seeded):
    from fastapi import HTTPException
    for call in (lambda: main.get_enrichment_status(scenario_id=SID, scope="everything"),
                 lambda: main.extract_pico_batch(scenario_id=SID, scope="everything"),
                 lambda: main.extract_metadata_batch(scenario_id=SID, scope="everything"),
                 lambda: main.fetch_fulltext_batch(scenario_id=SID, scope="everything")):
        with pytest.raises(HTTPException) as e:
            call()
        assert e.value.status_code == 400


def test_the_corpus_wide_status_still_answers_and_says_so(seeded):
    g = main.get_enrichment_status()
    assert g["scenario_id"] is None and g["by_scope"] is None
    assert g["total"] >= 4 and "todo" in g["pico"]


def test_done_and_todo_never_overlap(seeded, db_conn):
    """A weak extraction is reprocessed by the batch, so it is NOT done. Counting it
    in both made done plus todo exceed the total, and the bar contradict the line
    under it."""
    with db_conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET pico_json = %s WHERE id = 1",
                    ('{"P": "x", "pico_confidence": 0.2}',))      # faible
        cur.execute("UPDATE literature_document SET pico_json = %s WHERE id = 2",
                    ('{"P": "x", "pico_confidence": 0.9}',))      # bonne
    for scope in ("all", "relevant"):
        s = main.get_enrichment_status(scenario_id=SID, scope=scope)
        assert s["pico"]["count"] + s["pico"]["todo"] <= s["total"], scope
    s = main.get_enrichment_status(scenario_id=SID, scope="relevant")
    assert s["pico"]["count"] == 1                 # seule la bonne compte comme faite
    assert s["pico"]["todo"] == 2                  # la faible est reprise, avec la vide


def test_a_weak_extraction_is_not_counted_as_coverage_corpus_wide_either(seeded, db_conn):
    with db_conn.cursor() as cur:
        cur.execute("UPDATE literature_document SET pico_json = %s WHERE id = 1",
                    ('{"P": "x", "pico_confidence": 0.2}',))
    g = main.get_enrichment_status()
    assert g["pico"]["count"] + g["pico"]["todo"] <= g["total"]


# ── the pipeline's own enrichment, as a setting ──────────────────────────────

def test_the_pipeline_enriches_the_whole_scenario_by_default(monkeypatch):
    """The map half is what the rule "every extraction reads all the relevant
    articles" rests on: narrowing it is a choice someone makes, never a default."""
    monkeypatch.delenv("PIPELINE_ENRICH_SCOPE", raising=False)
    from api.scenario_store import pipeline_enrich_scope
    assert pipeline_enrich_scope() == "all"
    assert "similarity_threshold" not in main._enrich_gate()


def test_the_pipeline_can_be_held_to_the_relevant_subset(monkeypatch):
    monkeypatch.setenv("PIPELINE_ENRICH_SCOPE", "relevant")
    from api.scenario_store import pipeline_enrich_scope
    assert pipeline_enrich_scope() == "relevant"
    assert "similarity_threshold" in main._enrich_gate()


def test_an_unreadable_setting_keeps_the_whole_scenario(monkeypatch):
    """A typo in an environment variable must not quietly stop extracting."""
    for junk in ("", "  ", "RELEVENT", "true", "0"):
        monkeypatch.setenv("PIPELINE_ENRICH_SCOPE", junk)
        from api.scenario_store import pipeline_enrich_scope
        assert pipeline_enrich_scope() == "all", junk
