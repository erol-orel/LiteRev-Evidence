"""Every database says the exact query it receives, and can be queried again alone.

A reviewer writing a methods section has to report the strategy as submitted to each
database (PRISMA-S item 2), and until now LiteRev could not tell them. The plan said
"openalex: 4 parts" and "doaj: reduced", which names that something happened without
saying what was sent, and three things followed from that silence:

  - On the COVID exposure strategy of production (4,519 characters), OpenAIRE receives
    `workplace AND sars-cov-2 AND epidemiolog`. Three terms, because its API refuses more
    than four logical operators. Nothing in the interface said so, and its records were
    presented beside PubMed's as the product of the same search.
  - Semantic Scholar and arXiv cannot write a NOT, so a strategy with one exclusion sends
    them eight plain keywords. A reviewer reading "semantic_scholar 6,686" had no way to
    know that those 6,686 answered a word bag.
  - The plan was computed from the scenario's QUERY, while the federation sends
    `search_strategy.general`. On a scenario created from a natural-language question the
    plan therefore said "every source gets keywords" when in truth every source got the
    generated boolean.

So `sent` now carries the exact string each database receives, one per part when the query
is split, in that database's own syntax, ready to paste into its own site and check the
count. `planned_from` says which text was planned. And a reviewer who finds one database
short can ask for that one alone (`POST .../sources/{source}/rerun`) instead of a
three-quarter-hour search of thirteen.

The pure half of this file needs no database; the endpoint half skips without one.
"""
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api import search as S  # noqa: E402

#: A strategy with an exclusion, so the sources that cannot write a NOT show their fallback,
#: and short enough that the sources that can take it whole do.
SHORT = ('("Occupational Exposure"[mh] OR "poultry worker*"[tiab] OR "abattoir*"[tiab]) '
         'AND ("Influenza in Birds"[mh] OR "avian influenza"[tiab] OR "H5N1"[tiab]) '
         'NOT ("H1N1"[tiab])')

#: Long enough that the URL-carried sources must split it, as every real review strategy is.
LONG = ("(" + " OR ".join(f'"environmental exposure variant {i}"' for i in range(60)) + ")"
        " AND (" + " OR ".join(f'"coronavirus term {i}"' for i in range(60)) + ")")

SID = "usr-own-query"


def _plan(boolean: str, pubmed: str | None = None) -> dict:
    return {r["source"]: r for r in S.source_query_plan(boolean, pubmed=pubmed)}


# ── What each database receives, said exactly ────────────────────────────────

def test_every_source_names_the_exact_query_it_receives():
    """The whole point: no row may say only how its query travels. A reviewer must be able
    to read, copy and submit the string, for every source, in every carriage."""
    for boolean in (SHORT, LONG, "avian influenza in European poultry farms"):
        for row in S.source_query_plan(boolean):
            assert row["sent"], f"{row['source']} says how but not what ({row['how']})"
            assert all(isinstance(q, str) and q.strip() for q in row["sent"])


def test_pubmed_alone_keeps_its_field_tags():
    """eutils reads `[mh]` and `[tiab]`; every other API matches them as text, and Europe
    PMC once lost 99.9% of its recall to them. So PubMed is the one row that keeps them."""
    plan = _plan(SHORT, pubmed=SHORT)
    assert "[tiab]" in plan["pubmed"]["sent"][0]
    for name, row in plan.items():
        if name == "pubmed":
            continue
        assert "[tiab]" not in " ".join(row["sent"]), f"{name} must not be sent a field tag"


def test_pubmed_takes_the_strategy_s_own_pubmed_variant():
    """A strategy carries a MeSH-tagged variant for PubMed and a portable one for the rest.
    The plan must show PubMed the former: showing it the latter would be a query it never
    receives, and the point of this panel is that it cannot lie."""
    pubmed_only = '"Influenza in Birds"[Majr] AND "Occupational Exposure"[Majr]'
    plan = _plan(SHORT, pubmed=pubmed_only)
    assert plan["pubmed"]["sent"] == [pubmed_only]
    assert plan["openalex"]["sent"] != [pubmed_only], "the portable boolean, not PubMed's"


def test_a_split_source_shows_every_part_and_each_fits():
    """A split keeps every term: the parts' union IS the query. So the panel shows them all,
    and each one is under the limit that forced the split."""
    plan = _plan(LONG)
    for name in ("openalex", "clinicaltrials", "semantic_scholar"):
        row = plan[name]
        assert row["how"] == "parts", f"{name} should be split on this query"
        assert len(row["sent"]) == row["parts"] > 1
        cap = row["limit_value"]
        measure = len if name == "doaj" else S._url_encoded_len
        assert all(measure(q) <= cap for q in row["sent"]), f"a part of {name} exceeds its limit"


def test_a_reduced_source_shows_the_reduced_form_and_it_is_smaller():
    """OpenAIRE accepts four logical operators. On a review strategy it therefore receives a
    handful of terms, and that is the single most surprising thing this panel discloses."""
    row = _plan(LONG)["openaire"]
    assert row["how"] == "reduced"
    sent = row["sent"][0]
    assert S._count_operators(sent) <= S.OPENAIRE_MAX_OPERATORS
    assert len(sent) < len(S._clean_boolean(LONG)), "a reduction is smaller than the whole"


def test_a_source_that_cannot_express_the_query_shows_the_keywords_it_gets():
    """Semantic Scholar and arXiv have no syntax for a NOT. They then receive the keyword
    bag, and a reviewer reading their counts deserves to see the bag itself."""
    plan = _plan(SHORT)
    bag = S._plain_keywords(SHORT)
    assert bag, "the fixture is a boolean the keyword renderer can read"
    for name in ("semantic_scholar", "arxiv", "crossref"):
        assert plan[name]["how"] == "keywords"
        assert plan[name]["sent"] == [bag]


def test_no_boolean_at_all_still_names_the_keywords_every_source_gets():
    plan = S.source_query_plan("avian influenza in European poultry farms")
    assert {r["how"] for r in plan} == {"keywords"}
    assert all(r["sent"] for r in plan)


def test_the_plan_sends_what_the_search_derives_and_cannot_drift_from_it():
    """The anti-drift pin. `_run_user_scenario_populate` derives each source's query inline;
    the plan derives it again. The two must agree term for term, or the panel shows a query
    no database receives. Every pairing below is read off the search, not invented here."""
    portable = " ".join(S._strip_field_tags(LONG).split())
    clean = S._clean_boolean(portable)
    plan = _plan(LONG, pubmed=LONG)

    # PubMed: the strategy's own variant, before the year bounds and the preprint exclusion.
    assert plan["pubmed"]["sent"] == [LONG]
    # Europe PMC: the boolean without the field tags, via its own renderer.
    from api.sources import epmc_query
    assert plan["europepmc"]["sent"] == [epmc_query(LONG)]
    # CORE: the clean boolean in a POST body, whatever its length.
    assert plan["core"]["sent"] == [clean]
    # OpenAlex and ClinicalTrials.gov: split on the URL limit with the generic renderer.
    assert plan["clinicaltrials"]["sent"] == S._split_boolean(
        portable, S.URL_QUERY_MAX_ENCODED, measure=S._url_encoded_len)
    assert plan["openalex"]["sent"] == S._split_boolean(
        portable, S.URL_QUERY_MAX_ENCODED, measure=S._url_encoded_len)
    # Semantic Scholar and arXiv: their own compilers, their own split.
    assert plan["semantic_scholar"]["sent"] == S._split_boolean(
        portable, S.S2_MAX_QUERY_ENCODED, render=S._boolean_to_s2, measure=S._url_encoded_len)
    assert plan["arxiv"]["sent"] == S._split_boolean(
        portable, S.URL_QUERY_MAX_ENCODED, render=S._boolean_to_arxiv, measure=S._url_encoded_len)
    # OpenAIRE: reduced under its operator cap, measured on the encoded form.
    assert plan["openaire"]["sent"] == [S._shorten_boolean(
        portable, S.URL_QUERY_MAX_ENCODED, max_operators=S.OPENAIRE_MAX_OPERATORS,
        measure=S._url_encoded_len)]
    # Crossref: free text only, never a boolean.
    assert plan["crossref"]["sent"] == [S._plain_keywords(LONG)]


def test_the_federation_list_covers_every_source_the_plan_names():
    """A source a reviewer can see must be a source they can re-run, and the preprint facet
    is queried without a query of its own."""
    assert set(S.FEDERATION_SOURCES) == {r["source"] for r in S.source_query_plan(SHORT)} | {"preprints"}


# ── The scenario endpoint: the strategy, not the question that was typed ─────

def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:                                        # noqa: BLE001
        return False


@pytest.fixture()
def scenario(db_conn):
    """A scenario whose typed query is a question and whose strategy is a boolean: the shape
    the plan used to misread."""
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute(
            "INSERT INTO user_scenarios (id, name, query, mode, filters, search_strategy) "
            "VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)",
            (SID, "own query", "Which workers are exposed to avian influenza?", "literature",
             "{}", json.dumps({"general": SHORT, "pubmed": SHORT, "explanation": "kept"})))
    db_conn.commit()
    main._user_scenario_populate_jobs.pop(SID, None)
    yield SID
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
    db_conn.commit()
    main._user_scenario_populate_jobs.pop(SID, None)


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_the_plan_describes_the_strategy_the_search_sends_not_the_question_typed(scenario):
    """The correctness fix. The federation sends `search_strategy.general`; the plan read
    `query`. A scenario born of a question was therefore told that every database would
    receive keywords, while in truth they all received the generated boolean."""
    body = _client().get(f"/user-scenarios/{scenario}/source-query-plan").json()
    assert body["planned_from"] == "strategy"
    assert body["query_chars"] == len(SHORT)
    plan = {r["source"]: r for r in body["plan"]}
    assert plan["openalex"]["how"] == "whole", "the boolean reaches OpenAlex, not keywords"
    assert plan["pubmed"]["sent"] == [SHORT]
    assert body["pubmed_query"] == SHORT


def test_the_plan_falls_back_to_the_typed_query_when_there_is_no_usable_strategy(scenario, db_conn):
    """And says so, rather than inventing a boolean. `_strategy_is_degraded` is the same
    predicate the search uses to decide the same thing."""
    with db_conn.cursor() as cur:
        cur.execute("UPDATE user_scenarios SET search_strategy = NULL, query = %s WHERE id = %s",
                    (SHORT, SID))
    db_conn.commit()
    body = _client().get(f"/user-scenarios/{SID}/source-query-plan").json()
    assert body["planned_from"] == "query"
    assert body["query_chars"] == len(SHORT)


# ── An edited strategy: kept as typed, and the stale cache forgotten ─────────

def test_saving_a_strategy_keeps_it_exactly_as_typed(scenario, db_conn):
    """Never a silent rewrite: the product rule for a pasted strategy holds for an edited
    one. The explanation of the previous strategy is kept, and the edit is dated so nothing
    later mistakes a reviewer's boolean for a generated one."""
    edited = '("avian influenza" OR "H5N1") AND ("poultry worker" OR "culler")'
    r = _client().put(f"/user-scenarios/{SID}/search-strategy",
                      json={"general": edited},
                      headers={"X-API-Key": "test-write-key"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved"] is True and body["planned_from"] == "strategy"
    assert body["query_chars"] == len(edited)
    with db_conn.cursor() as cur:
        cur.execute("SELECT search_strategy FROM user_scenarios WHERE id = %s", (SID,))
        stored = cur.fetchone()[0]
    stored = stored if isinstance(stored, dict) else json.loads(stored)
    assert stored["general"] == edited
    assert stored["pubmed"] == edited, "no PubMed variant given means the same boolean"
    assert stored["explanation"] == "kept", "the previous explanation survives the edit"
    assert stored.get("edited_at"), "an edited strategy is dated as a reviewer's"


def test_saving_a_strategy_says_the_corpus_has_not_moved_yet(scenario):
    """The honest half. The corpus answers the boolean of the LAST SEARCH: saving a new one
    changes what the next search will do, not what the corpus holds now."""
    r = _client().put(f"/user-scenarios/{SID}/search-strategy",
                      json={"general": '("avian influenza") AND ("poultry worker")'},
                      headers={"X-API-Key": "test-write-key"})
    assert r.json()["corpus_stale"] is True


def test_saving_a_strategy_forgets_the_answers_cached_for_the_old_one(scenario, db_conn):
    """The trap this endpoint had to handle. The source cache is keyed by the scenario's
    QUERY, not by its strategy, so an edited strategy would otherwise re-run the search,
    replay every source's answer to the OLD boolean from the database, and report those
    counts under the new strategy. The cut positions go with it, for the same reason."""
    from api.pipeline import _save_source_cache
    from api.search import scenario_source_hash
    h = scenario_source_hash(SID)
    for fetcher in ("_fetch_pubmed", "_fetch_openalex"):
        _save_source_cache(fetcher, h, "old", [[1, "pubmed", True]])
    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM source_query_cache WHERE query_hash = %s", (h,))
        assert cur.fetchone()[0] == 2, "the fixture really did cache two answers"
    r = _client().put(f"/user-scenarios/{SID}/search-strategy",
                      json={"general": '("avian influenza") AND ("culler")'},
                      headers={"X-API-Key": "test-write-key"})
    assert r.json()["forgotten_source_answers"] == 2
    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM source_query_cache WHERE query_hash = %s", (h,))
        assert cur.fetchone()[0] == 0


def test_an_unbalanced_boolean_is_saved_as_typed_with_a_correction_proposed(scenario, db_conn):
    """A closing parenthesis too many STOPS the tokeniser and the rest of the query is
    ignored in silence. The rule is the one the search page already follows: keep what was
    typed, propose the fix, let the reviewer choose."""
    broken = '("avian influenza" OR "H5N1")) AND ("poultry worker")'
    body = _client().put(f"/user-scenarios/{SID}/search-strategy",
                         json={"general": broken},
                         headers={"X-API-Key": "test-write-key"}).json()
    assert body["issues"], "a real imbalance is reported"
    assert body["fixed"] and body["fixed"] != broken, "and a correction is offered"
    with db_conn.cursor() as cur:
        cur.execute("SELECT search_strategy->>'general' FROM user_scenarios WHERE id = %s", (SID,))
        assert cur.fetchone()[0] == broken, "saved as typed, never silently corrected"


def test_an_empty_strategy_is_refused(scenario):
    r = _client().put(f"/user-scenarios/{SID}/search-strategy", json={"general": "   "},
                      headers={"X-API-Key": "test-write-key"})
    assert r.status_code == 422


# ── One database queried again, and only that one ────────────────────────────

def test_rerunning_one_source_forgets_its_answer_and_keeps_the_others(scenario, db_conn, monkeypatch):
    """The mechanism, stated: a search replays whatever is still fresh, so forgetting ONE
    row is exactly "query that one again, replay the rest"."""
    from api import pipeline as P, scenarios as SC
    from api.search import scenario_source_hash
    h = scenario_source_hash(SID)
    for fetcher in ("_fetch_pubmed", "_fetch_openalex", "_fetch_doaj"):
        P._save_source_cache(fetcher, h, "answered", [[1, "pubmed", True]])
    launched: list = []
    monkeypatch.setattr(SC, "_launch_populate_job",
                        lambda *a, **k: (launched.append((a, k)), "started")[1])
    body = _client().post(f"/user-scenarios/{SID}/sources/openalex/rerun",
                          headers={"X-API-Key": "test-write-key"}).json()
    assert body["status"] == "started" and body["source"] == "openalex"
    assert body["forgot_cached_answer"] is True
    assert set(body["replayed_from_cache"]) == {"pubmed", "doaj"}, "the others are not re-queried"
    with db_conn.cursor() as cur:
        cur.execute("SELECT fetcher FROM source_query_cache WHERE query_hash = %s ORDER BY 1", (h,))
        assert [r[0] for r in cur.fetchall()] == ["_fetch_doaj", "_fetch_pubmed"]
    assert launched, "and the ordinary search is what runs"
    assert launched[0][1].get("force_live") is False, "force_live would re-query all thirteen"


def test_rerunning_a_source_that_never_answered_queries_it_anyway(scenario, monkeypatch):
    """A source that failed has no cached row to forget. Saying so is useful; refusing would
    be absurd, since a failed source is precisely the one to query again."""
    from api import scenarios as SC
    monkeypatch.setattr(SC, "_launch_populate_job", lambda *a, **k: "started")
    body = _client().post(f"/user-scenarios/{SID}/sources/europepmc/rerun",
                          headers={"X-API-Key": "test-write-key"}).json()
    assert body["status"] == "started" and body["forgot_cached_answer"] is False


def test_a_cut_source_resumes_where_it_stopped_unless_told_to_start_over(scenario, monkeypatch):
    """Resuming is the default, because that is what makes an exhaustive search converge:
    each round continues instead of rereading. Starting over is the explicit choice."""
    from api import pipeline as P, scenarios as SC
    from api.search import scenario_source_hash
    monkeypatch.setattr(SC, "_launch_populate_job", lambda *a, **k: "started")
    h = scenario_source_hash(SID)
    P._save_fetch_cursor(SID, "_fetch_openalex", h, position="cursor-42", fetched=5000, rounds=1)
    assert P._load_fetch_cursors(SID, h).get("_fetch_openalex"), "the fixture wrote a position"

    _client().post(f"/user-scenarios/{SID}/sources/openalex/rerun",
                   headers={"X-API-Key": "test-write-key"})
    assert P._load_fetch_cursors(SID, h).get("_fetch_openalex"), "resume keeps the position"

    _client().post(f"/user-scenarios/{SID}/sources/openalex/rerun?resume=false",
                   headers={"X-API-Key": "test-write-key"})
    assert not P._load_fetch_cursors(SID, h).get("_fetch_openalex"), "starting over forgets it"


def test_rerunning_a_source_the_federation_does_not_query_is_refused(scenario):
    """Named sources only: a typo must not silently launch a thirteen-source search."""
    r = _client().post(f"/user-scenarios/{SID}/sources/scopus/rerun",
                       headers={"X-API-Key": "test-write-key"})
    assert r.status_code == 404
    assert "openalex" in r.json()["detail"], "the answer names what is accepted"


def test_rerunning_while_a_search_is_running_is_refused(scenario):
    """Two federations on one scenario would interleave their PRISMA counts."""
    main._user_scenario_populate_jobs[SID] = {"status": "running"}
    try:
        r = _client().post(f"/user-scenarios/{SID}/sources/openalex/rerun",
                           headers={"X-API-Key": "test-write-key"})
        assert r.status_code == 409
    finally:
        main._user_scenario_populate_jobs.pop(SID, None)


# ── The cache key describes what the databases received ─────────────────────

def test_a_cached_answer_is_keyed_by_the_boolean_the_databases_received():
    """Measured on production: three scenarios carried the same 62 character question, one
    in intersection (the main boolean alone), one in union (`(main) OR (Machine learning)`),
    one with no facet. The key held the question, so all three shared it: the first to
    search wrote the cache and the others replayed its records as their own."""
    from api.pipeline import _source_query_hash
    question = "identification de la source des infections du sang"
    main = '("bloodstream infection" OR "bacteraemia") AND ("source attribution")'
    widened = f"({main}) OR (machine learning)"
    keys = {_source_query_hash(question, {}, 2000, "exhaustive", boolean=b)
            for b in (main, widened)}
    assert len(keys) == 2, "two federations, two keys"


def test_a_pasted_strategy_keeps_the_key_it_already_had():
    """The other half: every partner review pasted its boolean as the query, so the
    federated boolean IS the query and the key must not move. A key that moved would throw
    away four reviews' cached answers and make the next retry a full search of thirteen."""
    from api.pipeline import _source_query_hash
    before = _source_query_hash(SHORT, {"year_min": 2000}, 2000, "exhaustive")
    after = _source_query_hash(SHORT, {"year_min": 2000}, 2000, "exhaustive", boolean=SHORT)
    assert before == after
    # And trailing whitespace is not a different search.
    assert after == _source_query_hash(SHORT, {"year_min": 2000}, 2000, "exhaustive",
                                       boolean=f"  {SHORT}  ")


def test_the_search_records_the_key_it_wrote_so_a_reader_never_re_derives_it(scenario, db_conn):
    """Deriving the key outside the search would mean re-translating a natural-language
    question with the model and re-widening the OR facets. A key derived slightly
    differently reads as "nothing is cached", and asking for one database again would then
    query all thirteen. So the search records its key and the readers read it."""
    from api.search import _store_prisma_identification, scenario_source_hash
    _store_prisma_identification(SID, {"records_by_source": {"pubmed": 3},
                                       "source_query_hash": "written-by-the-search"})
    assert scenario_source_hash(SID) == "written-by-the-search"


def test_a_scenario_searched_before_that_was_recorded_still_finds_its_cache(scenario):
    """No figures, or figures from before the key was recorded: derived from the row, which
    for a pasted strategy is the key that is already in the database."""
    from api.pipeline import _source_query_hash
    from api.search import scenario_source_hash
    row = main._get_user_scenario_or_404(SID)
    assert scenario_source_hash(SID, row) == _source_query_hash(
        row["query"], row.get("filters") or {}, main.LIVE_MAX_PER_SOURCE, "exhaustive",
        boolean=SHORT)


def test_rerunning_a_source_needs_the_write_key(scenario):
    """It spends network calls and moves the corpus: never anonymous."""
    assert _client().post(f"/user-scenarios/{SID}/sources/openalex/rerun").status_code in (401, 403)
    assert _client().put(f"/user-scenarios/{SID}/search-strategy",
                         json={"general": "a AND b"}).status_code in (401, 403)
