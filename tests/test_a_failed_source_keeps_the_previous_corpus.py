"""A search whose federation did not complete keeps the corpus it had, and no search deletes
a reviewer's decision.

On 12 October 2026 a re-search of HPAI_last (exhaustive) ran while Europe PMC answered 503
for a few minutes. Europe PMC and its preprint facet ended in `error`, and the corpus was
rebuilt from the partial result: 8,494 articles instead of 15,000 to 18,000. What only
Europe PMC's native boolean brings (MeSH and synonym matches, which the local title and
abstract re-match does not reach) was unlinked, twice over: the populate reset the corpus to
the local match before the federation even started, then set it to the partial union at
assembly. The PRISMA card said "failed: europepmc"; the corpus shrank anyway. The same reset
deleted links unconditionally, a hand inclusion with its reason and votes included, and so
did the rebuild from the local library.

These tests drive the real populate against in-memory sources (Europe PMC up or down), over
a corpus seeded with links that only Europe PMC's boolean found and links a reviewer decided
on. Rule pinned here: an incomplete federation removes nothing the corpus had before the
search; a complete one removes what it no longer matches, except what a reviewer decided on.
"""
import json
import pathlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from api.scenario_store import relevant_gate_sql, reviewer_decision_sql  # noqa: E402
from conftest import patch_app  # noqa: E402
from search_tables import ensure_search_tables  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SID = "usr-keepprev-0001"
QUERY = '("wombat fever" OR wombatitis) AND (zookeeper OR "animal keeper")'
NOTHING = '("platypus fever" OR platypitis) AND zookeeper'
LONG = " A study long enough to be an abstract, about exposure, serology and follow-up in staff."

# The local library matches these two on their title and abstract.
LOCAL = {98901: "Wombat fever in zookeepers: a seroprevalence study",
         98902: "Wombatitis outbreak among animal keeper staff"}
# Linked by the previous search through Europe PMC's native boolean only: the local re-match
# does not reach them. 98904 is what Europe PMC returns again when it answers.
NATIVE = {98903: "Marsupial zoonoses surveillance in sanctuaries",
          98904: "Serology of captive marsupial handlers",
          98908: "Exposure of wildlife carers to marsupial pathogens"}
# Linked by the previous search too, and decided on by a reviewer: (status, vote, reason).
DECIDED = {98905: ("included", None, "relevant case report"),
           98906: ("excluded", None, "guideline, not a study"),
           98907: (None, "excluded", None)}            # a double-blind vote, nothing else
DECIDED_TITLES = {98905: "Case report: a keeper bitten by a marsupial",
                  98906: "Marsupial husbandry guidelines",
                  98907: "Veterinary handling of marsupials"}
TITLES = {**LOCAL, **NATIVE, **DECIDED_TITLES}
EVERY = set(TITLES)


class _Resp:
    def __init__(self, status=200, payload=None, body=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = body
        self.content = body.encode()
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Sources:
    """Every source answers from memory and finds nothing, except Europe PMC: down (503 on
    every attempt, as on 12 October) or up, returning `epmc_records` to its native boolean."""

    def __init__(self):
        self.epmc_down = False
        self.epmc_records = [98904]
        self.epmc_calls = 0

    def get(self, url, params=None, **kw):
        url, params = str(url), dict(params or {})
        if "europepmc" in url or "ebi.ac.uk" in url:
            self.epmc_calls += 1
            if self.epmc_down:
                return _Resp(503, body="Service Temporarily Unavailable")
            res = [] if "SRC:PPR" in str(params.get("query", "")) else [
                {"pmcid": f"PMC{d}", "title": TITLES[d], "abstractText": TITLES[d] + LONG,
                 "pubYear": "2021"} for d in self.epmc_records]
            return _Resp(200, {"resultList": {"result": res}, "hitCount": len(res), "nextCursorMark": None})
        if "eutils" in url:
            return _Resp(200, {"esearchresult": {"count": "0", "idlist": []}})
        if "crossref" in url:
            return _Resp(200, {"message": {"items": [], "total-results": 0}})
        if "openalex" in url:
            return _Resp(200, {"results": [], "meta": {"count": 0, "next_cursor": None}})
        if "clinicaltrials" in url:
            return _Resp(200, {"studies": [], "totalCount": 0})
        if "arxiv" in url:
            return _Resp(200, body="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")
        if "openaire" in url:
            return _Resp(200, {"results": [], "header": {}})
        if "semanticscholar" in url:
            return _Resp(200, {"data": [], "total": 0})
        if "doaj.org" in url:
            return _Resp(200, {"total": 0, "results": []})
        return _Resp(200, {})

    def post(self, url, data=None, json=None, **kw):              # noqa: A002 - the requests name
        return _Resp(200, {"results": [], "totalHits": 0})


def _ingest(**kw):
    """Europe PMC's records are the seeded documents (`PMC<id>`); nothing else is ingested."""
    ext = str(kw.get("external_id") or "")
    if ext.startswith("PMC") and ext[3:].isdigit() and int(ext[3:]) in EVERY:
        return int(ext[3:]), False
    return None, False


def _clean(cur):
    cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
    cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
    cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
    cur.execute("DELETE FROM document_chunk WHERE document_id BETWEEN 98900 AND 98920")
    cur.execute("DELETE FROM article_scenarios WHERE document_id BETWEEN 98900 AND 98920")
    cur.execute("DELETE FROM literature_document WHERE id BETWEEN 98900 AND 98920")
    cur.execute("DELETE FROM source_query_cache")


def _strategy(boolean: str) -> str:
    return json.dumps({"general": boolean, "pubmed": boolean, "explanation": "", "synonyms": []})


@pytest.fixture
def scenario(db_conn, monkeypatch):
    import requests as _real_requests

    from api import pipeline as P
    from api import relevance as R

    created_chunk_table = ensure_search_tables(db_conn.cursor())
    main._ensure_scenario_settings_table()
    fake = _Sources()
    monkeypatch.setattr(_real_requests, "get", fake.get)
    monkeypatch.setattr(_real_requests, "post", fake.post)
    monkeypatch.setattr(P, "_ncbi_get", lambda url, params, timeout=30, use_key=True: fake.get(url, params))
    monkeypatch.setattr(P, "_ingest_doc_direct", _ingest)
    monkeypatch.setattr(P, "FETCH_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))
    monkeypatch.setenv("CORE_API_KEY", "")
    monkeypatch.setenv("LEXICAL_SEARCH_ENGINE", "like")
    monkeypatch.setattr(R, "_run_semantic_rerank_inline", lambda *a, **k: 0)
    monkeypatch.setattr(R, "_run_cross_encoder_rerank", lambda *a, **k: {})
    monkeypatch.setattr(P, "_run_clustering_background", lambda *a, **k: None)
    monkeypatch.setattr(P, "_precompute_user_kg", lambda *a, **k: None)
    with db_conn.cursor() as cur:
        _clean(cur)
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, article_count, search_strategy) "
                    "VALUES (%s, 'Keep previous', %s, 'boolean', '{}', %s, %s::jsonb)",
                    (SID, QUERY, len(TITLES), _strategy(QUERY)))
        for doc_id, title in TITLES.items():
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                        "is_duplicate) VALUES (%s, %s, %s, 2021, 'europepmc', 'literev', false)",
                        (doc_id, title, title + LONG))
        # The corpus of the previous search: every document above, three of them decided on,
        # one left pending with a note (a note is not a decision).
        for doc_id in TITLES:
            status, vote, reason = DECIDED.get(doc_id, (None, None, None))
            notes = None
            if doc_id == 98908:
                status, notes = "pending", "to look at again"
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, "
                        "screening_status, screening_reason, screening_notes, reviewer_1_status) "
                        "VALUES (%s, %s, 0.5, %s, %s, %s, %s)", (SID, doc_id, status, reason, notes, vote))

    def run(boolean: str = QUERY):
        with db_conn.cursor() as cur:
            cur.execute("UPDATE user_scenarios SET search_strategy = %s::jsonb WHERE id = %s",
                        (_strategy(boolean), SID))
        P._run_user_scenario_populate(SID, boolean, {}, include_live=True, auto_pipeline=False,
                                      force_live=True, exhaustive=True)
        return P._user_scenario_populate_jobs.get(SID) or {}

    yield run, fake, db_conn
    with db_conn.cursor() as cur:
        _clean(cur)
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")
    main._RERANK_JOBS.pop(SID, None)


def _linked(conn) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT document_id FROM article_scenarios WHERE scenario_id = %s", (SID,))
        return {r[0] for r in cur.fetchall()}


def _decisions(conn) -> dict[int, tuple]:
    with conn.cursor() as cur:
        cur.execute("SELECT document_id, screening_status, screening_reason, reviewer_1_status "
                    "FROM article_scenarios WHERE scenario_id = %s AND document_id = ANY(%s)",
                    (SID, list(DECIDED)))
        return {r[0]: (r[1], r[3], r[2]) for r in cur.fetchall()}


def _adds_up(fig: dict) -> bool:
    """identified − duplicates − removals + kept = screened, the card's arithmetic."""
    return (fig["records_identified"] - fig["duplicates_removed"] - fig["removed_before_screening"]
            + fig["kept_previous_links"] + fig["kept_reviewed_links"]) == fig["records_screened"]


# ── the search, end to end ───────────────────────────────────────────────────

def test_a_failed_source_keeps_the_previous_links_and_says_so(scenario):
    run, fake, conn = scenario
    fake.epmc_down = True
    job = run()
    outcomes = job.get("source_outcomes") or {}
    assert outcomes.get("europepmc") == "error" and outcomes.get("preprints") == "error", outcomes
    assert fake.epmc_calls >= 3, "the 503 was not retried"
    # Nothing the corpus had is gone: not the links only Europe PMC's boolean found, not the
    # pending one. The populate used to reset the corpus to the local match before the
    # federation started, and the assembly then set it to the partial union.
    assert _linked(conn) == EVERY
    fig = job.get("prisma_identification") or {}
    assert fig.get("kept_previous_links") == 3, fig                 # 98903, 98904, 98908
    assert fig.get("kept_reviewed_links") == 3, fig
    assert fig.get("kept_previous_failed_sources") == ["europepmc", "preprints"], fig
    assert fig.get("records_screened") == len(EVERY) and _adds_up(fig), fig
    assert _decisions(conn) == DECIDED


def test_the_card_receives_the_kept_links_and_the_arithmetic_holds(scenario):
    run, fake, _conn = scenario
    fake.epmc_down = True
    run()
    ident = main.get_user_scenario_prisma(SID, threshold=0.45)["identification"]
    assert ident["kept_previous_links"] == 3 and ident["kept_reviewed_links"] == 3, ident
    assert ident["kept_previous_failed_sources"] == ["europepmc", "preprints"], ident
    assert "europepmc" in ident["sources_failed"]
    assert (ident["total_records"] - ident["duplicates_removed"] - ident["removed_before_screening"]
            + ident["kept_previous_links"] + ident["kept_reviewed_links"]
            + ident["added_after_search"] - ident["removed_after_search"]) == ident["records_screened"], ident


def test_a_complete_search_still_drops_the_stale_links(scenario):
    run, fake, conn = scenario
    job = run()
    outcomes = job.get("source_outcomes") or {}
    assert not {o for o in outcomes.values()} & {"error", "cut_by_budget"}, outcomes
    # 98904 came back through Europe PMC's native boolean; 98903 and the pending 98908 did
    # not, and nobody decided on them: they leave. The decided ones stay.
    assert _linked(conn) == set(LOCAL) | {98904} | set(DECIDED)
    fig = job.get("prisma_identification") or {}
    assert fig.get("kept_previous_links") == 0 and fig.get("kept_previous_failed_sources") == [], fig
    assert fig.get("kept_reviewed_links") == 3 and _adds_up(fig), fig
    assert fig["records_by_source"].get("europepmc") == 1, fig


def test_a_hand_inclusion_survives_a_re_search(scenario):
    run, _fake, conn = scenario
    run()
    run()                                                    # and the next one
    assert _decisions(conn) == DECIDED, "a decision, its reason or a vote was lost"
    with conn.cursor() as cur:
        cur.execute(f"SELECT d.id FROM literature_document d JOIN article_scenarios ars "
                    f"ON ars.document_id = d.id AND ars.scenario_id = %(sid)s "
                    f"WHERE {relevant_gate_sql('d', 'ars', '%(thr)s')}", {"sid": SID, "thr": 0.99})
        relevant = {r[0] for r in cur.fetchall()}
    assert 98905 in relevant, "above the threshold, plus what a reviewer included by hand"
    p = main.get_user_scenario_prisma(SID, threshold=0.45)
    assert p["identification"]["kept_reviewed_links"] == 3
    assert p["manual_curation"]["included"] == 1 and p["manual_curation"]["excluded"] == 1


def test_an_empty_result_of_a_failed_federation_empties_nothing(scenario):
    run, fake, conn = scenario
    fake.epmc_down = True
    job = run(NOTHING)
    assert _linked(conn) == EVERY
    fig = job.get("prisma_identification") or {}
    assert fig.get("records_identified") == 0 and fig.get("kept_previous_links") == 5, fig
    assert _adds_up(fig), fig


def test_an_empty_result_of_a_complete_federation_keeps_only_the_decisions(scenario):
    run, fake, conn = scenario
    fake.epmc_records = []
    run(NOTHING)
    assert _linked(conn) == set(DECIDED)


def test_the_rebuild_keeps_the_decisions_too(scenario, monkeypatch):
    import threading

    class _SyncThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self.target, self.args, self.kwargs = target, args, kwargs or {}

        def start(self):
            self.target(*self.args, **self.kwargs)

    _run, _fake, conn = scenario
    monkeypatch.setattr(threading, "Thread", _SyncThread)
    patch_app(monkeypatch, "_run_semantic_rerank_inline", lambda sid, q, report=None, **k: 0)
    patch_app(monkeypatch, "_run_cross_encoder_rerank", lambda *a, **k: {"scored": 0})
    assert main.rebuild_corpus(SID)["status"] == "started"
    assert main._RERANK_JOBS[SID]["status"] == "done", main._RERANK_JOBS[SID]
    # The local library only: the native-only links go, the decided ones stay.
    assert _linked(conn) == set(LOCAL) | set(DECIDED)
    assert _decisions(conn) == DECIDED
    fig = main._load_prisma_identification(SID)
    assert fig["method"] == "rebuild" and fig["kept_reviewed_links"] == 3, fig
    assert fig["records_screened"] == 5 and _adds_up(fig), fig


# ── the primitive ────────────────────────────────────────────────────────────

def test_set_scenario_corpus_keeps_what_it_is_told_to_and_every_decision(scenario):
    _run, _fake, conn = scenario
    assert main._set_scenario_corpus(SID, []) == 0
    assert _linked(conn) == EVERY, "an empty result touches nothing unless allowed"
    main._set_scenario_corpus(SID, [98901], keep=[98903])
    assert _linked(conn) == {98901, 98903} | set(DECIDED)
    main._set_scenario_corpus(SID, [], allow_empty=True)
    assert _linked(conn) == set(DECIDED)
    assert _decisions(conn) == DECIDED


def _db_or_skip():
    try:
        with main.engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"no reachable Postgres ({type(e).__name__})")


def test_the_decision_predicate_is_never_null():
    """`NOT` is applied to it as is: a NULL would keep every link nobody ever judged."""
    _db_or_skip()
    rows = [  # (screening_status, reviewer_1, reviewer_2, kappa_final, decided)
        (None, None, None, None, False),
        ("pending", None, None, None, False),
        ("included", None, None, None, True),
        ("excluded", None, None, None, True),
        (None, "excluded", None, None, True),
        (None, None, "pending", None, True),
        (None, None, None, "included", True),
    ]
    values = ", ".join(f"(:s{i}, :r1{i}, :r2{i}, :k{i})" for i in range(len(rows)))
    params = {}
    for i, (s, r1, r2, k, _d) in enumerate(rows):
        params.update({f"s{i}": s, f"r1{i}": r1, f"r2{i}": r2, f"k{i}": k})
    sql = (f"SELECT {reviewer_decision_sql('a')} AS decided FROM (VALUES {values}) "
           f"AS a(screening_status, reviewer_1_status, reviewer_2_status, kappa_final_status)")
    with main.engine.connect() as c:
        got = [r.decided for r in c.execute(text(sql), params)]
    assert got == [d for *_r, d in rows]


def test_the_search_criteria_apply_with_the_semantics_of_the_old_deletes(db_conn):
    """Years and preprints, now applied before the corpus is written: a document leaves only
    when the condition says so (an unknown year is not out of range, a preprint predicate
    that cannot decide is not a preprint)."""
    ensure_search_tables(db_conn.cursor())
    rows = [  # (id, year, source, doi)
        (98911, 1987, "pubmed", "10.5555/old"),
        (98912, 2020, "pubmed", "10.5555/recent"),
        (98913, None, "pubmed", "10.5555/undated"),
        (98914, 2021, "openalex", "10.1101/2020.03.05.20031773"),     # a medRxiv preprint
        (98915, 2021, "crossref", None),                              # undecidable: kept
    ]
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 98911 AND 98915")
        for doc_id, year, source, doi in rows:
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, doi, project_context, "
                        "is_duplicate) VALUES (%s, %s, %s, %s, %s, %s, 'literev', false)",
                        (doc_id, f"Criteria probe {doc_id}", "Criteria probe" + LONG, year, source, doi))
    try:
        ids = [r[0] for r in rows]
        assert main._ids_within_criteria(ids, {}) == ids
        assert set(main._ids_within_criteria(ids, {"year_min": 2015, "exclude_preprints": True})) == \
            {98912, 98913, 98915}
        assert set(main._ids_within_criteria(ids, {"year_max": 2020})) == {98911, 98912, 98913}
        assert set(main._ids_within_criteria(ids, {"exclude_preprints": "false", "year_min": ""})) == set(ids)
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM literature_document WHERE id BETWEEN 98911 AND 98915")


# ── the figures (pure) ───────────────────────────────────────────────────────

def test_the_kept_links_have_their_lines_and_the_arithmetic_holds():
    f = main._prisma_identification_figures(
        {"db_cache": 2, "europepmc": 1}, 3, 0, 9, removed_not_matching=1,
        kept_previous_links=5, kept_previous_failed_sources=["_fetch_europepmc", "preprints"],
        kept_reviewed_links=2)
    assert (f["kept_previous_links"], f["kept_reviewed_links"]) == (5, 2)
    assert f["kept_previous_failed_sources"] == ["europepmc", "preprints"]
    assert f["removed_not_matching"] == 1 and f["removed_other_reasons"] == 0, f
    assert _adds_up(f) and f["records_screened"] == 9
    # Brought to the corpus as it stands later, the identity still holds.
    r = main._reconcile_prisma_identification(f, 12)
    assert (r["records_identified"] - r["duplicates_removed"] - r["removed_before_screening"]
            + r["kept_previous_links"] + r["kept_reviewed_links"]
            + r["added_after_search"] - r["removed_after_search"]) == r["records_screened"] == 12


def test_the_kept_lines_never_exceed_what_was_screened():
    f = main._prisma_identification_figures({"pubmed": 1}, 1, 0, 2, kept_previous_links=5,
                                            kept_previous_failed_sources=["europepmc"], kept_reviewed_links=5)
    assert (f["kept_reviewed_links"], f["kept_previous_links"]) == (2, 0)
    assert f["kept_previous_failed_sources"] == [], "no record kept from before, no source to blame"
    assert _adds_up(f)


def test_a_search_that_kept_nothing_says_nothing_about_it():
    f = main._prisma_identification_figures({"pubmed": 4}, 4, 0, 3, removed_not_matching=1)
    assert (f["kept_previous_links"], f["kept_reviewed_links"], f["kept_previous_failed_sources"]) == (0, 0, [])
    assert _adds_up(f)


# ── the interface (source pinned) ────────────────────────────────────────────

def test_the_card_says_how_many_were_kept_why_and_what_to_do():
    page = (ROOT / "frontend" / "src" / "components" / "ScenarioDetailPage.tsx").read_text(encoding="utf-8")
    assert "ident.kept_previous_links" in page and "ident.kept_reviewed_links" in page
    assert "scenarioDetail.prisma.keptPreviousNote" in page
    api_ts = (ROOT / "frontend" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    for field in ("kept_previous_links?: number", "kept_previous_failed_sources?: string[]",
                  "kept_reviewed_links?: number"):
        assert field in api_ts, field
    for lang in ("fr", "en"):
        loc = (ROOT / "frontend" / "src" / "i18n" / "locales" / f"{lang}.ts").read_text(encoding="utf-8")
        note = next(line for line in loc.splitlines() if line.strip().startswith("keptPreviousNote:"))
        assert "{n}" in note and "{sources}" in note, (lang, note)
        assert "keptPreviousLinks:" in loc and "keptReviewedLinks:" in loc, lang
