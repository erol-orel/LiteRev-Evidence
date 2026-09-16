"""The app must do exactly what it says it does.

Each test here pins one place where a claim and the behaviour behind it had drifted
apart. They are cheap, pure tests: what they guard is agreement between two parts of the
code (a writer and a reader, a docstring and a response, a promise and its prerequisite),
which is exactly the kind of thing that silently rots between two commits.
"""
import main
from conftest import patch_app


# ── The evidence brief: one fingerprint, written and read ────────────────────
def test_brief_context_version_is_one_shared_constant():
    """The writer stamps the cached brief with a context version and the reader looks it
    up by the same one. They were two literals once ("brief-v4-..." written,
    "brief-v3-..." read): the fingerprint never matched again, so GET
    /evidence-brief/llm answered "generation in progress" for ever, started a new thread
    on every poll, and the PDF came out without its narrative. A single constant, used
    twice, is what makes that unrepresentable."""
    import inspect

    from api import evidence

    assert isinstance(evidence.BRIEF_CONTEXT_VERSION, str)
    assert evidence.BRIEF_CONTEXT_VERSION.strip()

    writer = inspect.getsource(evidence._generate_evidence_brief_llm)
    reader = inspect.getsource(evidence.get_llm_evidence_brief)
    for name, src in (("writer", writer), ("reader", reader)):
        assert "BRIEF_CONTEXT_VERSION" in src, f"the {name} must use the shared constant"
        # No hand-written "brief-vN-..." literal may come back alongside it.
        assert "\"brief-v" not in src and "'brief-v" not in src, (
            f"the {name} carries a literal brief version again; use BRIEF_CONTEXT_VERSION")


# ── The corpus digest: a failed aggregation is not "complete" ─────────────────
def test_digest_reports_incomplete_when_the_aggregation_fails(monkeypatch):
    """`complete` used to be set outside the try, so a database error returned
    {"n_articles": 0, "complete": True}: the generators fell back to their 30 reproduced
    articles without a word, and the coverage note announced "ALL 0 articles". The one
    failure mode that breaks the house rule declared itself compliant."""
    from api import digest as digest_mod

    class _Boom:
        def connect(self):
            raise RuntimeError("no database")

    patch_app(monkeypatch, "engine", _Boom())
    out = digest_mod.corpus_digest("usr-nothing", 0.45)
    assert out["complete"] is False
    assert out["n_articles"] == 0
    assert "no database" in out.get("error", "")


def test_an_incomplete_digest_asserts_nothing_in_a_prompt():
    """A digest that could not be computed must produce an EMPTY prompt block, so the
    generator falls back to its explicit wording instead of stating a false total as if
    it were exhaustive."""
    from api.digest import digest_to_prompt

    broken = {"n_articles": 12, "complete": False, "error": "boom"}
    assert digest_to_prompt(broken) == ""
    # ... while a complete one does speak, and says how many articles it covers.
    good = {"n_articles": 12, "complete": True, "years": {"min": 2015, "max": 2024}}
    assert "12" in digest_to_prompt(good)


def test_the_coverage_sentence_inverts_instead_of_claiming_a_false_total():
    """The block of figures disappears on a failed digest, but the sentence under it went
    on saying "the figures above cover the TOTALITY of 0 relevant articles". The model
    then read a guarantee of exhaustiveness covering nothing, while holding only the
    reproduced articles: precisely the sample the house rule forbids. On a failure the
    sentence now forbids generalising instead of falling silent."""
    from api.digest import digest_coverage_note

    broken = digest_coverage_note({"n_articles": 0, "complete": False}, 30)
    assert "TOTALITE" not in broken
    assert "30" in broken and "AUCUN total" in broken

    good = digest_coverage_note({"n_articles": 1200, "complete": True}, 30)
    assert "TOTALITE des 1200" in good


# ── /sources/health: six probes are not the whole federation ─────────────────
def test_sources_health_names_what_it_does_not_probe():
    """The endpoint probes six fetchers out of twelve, so "6 reachable out of 6" must not
    read as "the federation is healthy". The response carries `probed` and `not_probed`
    so the gap is in the payload, not only in the docstring."""
    import inspect

    from api import sources

    src = inspect.getsource(sources.sources_health)
    assert '"probed"' in src and '"not_probed"' in src
    for missing in ("Semantic Scholar", "DOAJ", "ClinicalTrials.gov", "CORE", "arXiv",
                    "OpenAIRE"):
        assert missing in src, f"{missing} is a fetcher this endpoint does not probe"
    # The docstring must not claim it covers every upstream API.
    doc = sources.sources_health.__doc__ or ""
    assert "chaque API amont" not in doc


# ── Email alerts: subscribing promises nothing the server cannot keep ────────
def test_subscribe_does_not_promise_an_email_nothing_will_send():
    """The API has no scheduler: POST /alerts/subscribe stores a row, and the digests go
    out only when a cron calls /alerts/run-digests with SMTP configured. The old response
    said "you will receive {frequency} alerts", which was false on every deployment where
    either piece was missing."""
    import inspect

    from api import alerts

    src = inspect.getsource(alerts.subscribe_alerts)
    assert '"delivery"' in src
    assert "smtp_configured" in src
    assert "requires_scheduled_runner" in src
    assert "Vous recevrez des alertes" not in src
    doc = alerts.subscribe_alerts.__doc__ or ""
    assert "run-digests" in doc


# ── The rate limiter: the expensive list is what the docs say it is ──────────
def test_search_is_not_on_the_expensive_rate_limit():
    """The architecture note listed `/search` among the 30/min paths. It never was, and
    it must not become one: the search page fires several requests per keystroke, so a
    30/min bucket would 429 a single user typing."""
    from api.core import EXPENSIVE_PATHS, _EXPENSIVE_PATTERNS

    assert "/search" not in EXPENSIVE_PATHS
    assert not any(rx.match("/search") for rx in _EXPENSIVE_PATTERNS)
    # The ones that ARE expensive still match, sub-paths included.
    for path in ("/ask", "/ask/stream/filtered", "/user-scenarios/usr-1/rag",
                 "/scenarios/usr-1/full-pipeline"):
        assert any(rx.match(path) for rx in _EXPENSIVE_PATTERNS), path
    # ... and a parameterised expensive route never swallows its sibling routes.
    for sibling in ("/user-scenarios/usr-1/prisma", "/user-scenarios/usr-1/clustering",
                    "/user-scenarios/usr-1/evidence-brief"):
        assert not any(rx.match(sibling) for rx in _EXPENSIVE_PATTERNS), sibling


# ── Moving the threshold invalidates everything computed from the old corpus ─
def test_moving_the_threshold_drops_every_artefact_computed_from_the_old_corpus():
    """Each cached artefact is a function of the relevant subset, so changing the
    threshold must drop it. The three visualisations were nulled and the brief and the
    variables self-invalidate (their fingerprint carries the threshold), but the
    recommended actions were keyed only by (scenario, language): moving the slider from
    0.45 to 0.60 went on serving, indefinitely, actions drawn from the previous corpus
    while the tab presented them as the current one's."""
    import inspect

    from api import relevance

    from api.scenario_store import CORPUS_DERIVED_CACHE_RESET

    for column in ("clustering_json", "knowledge_graph_json", "concept_graph_json",
                   "recommended_actions_json"):
        assert f"{column} = NULL" in CORPUS_DERIVED_CACHE_RESET, f"{column} is not reset"
    # The language marker goes too, or a stale cache is served as if it were fresh.
    assert "recommended_actions_lang = NULL" in CORPUS_DERIVED_CACHE_RESET

    # Both places that invalidate must use that ONE list, not their own copy: the two
    # SQL statements had already drifted apart over the recommended actions.
    threshold_path = inspect.getsource(relevance.update_scenario_settings)
    assert "CORPUS_DERIVED_CACHE_RESET" in threshold_path

    from api import living_review

    assert "CORPUS_DERIVED_CACHE_RESET" in inspect.getsource(living_review.trigger_living_review)


# ── data_connectors: the module docstring lists the connectors that exist ────
def test_connector_docstring_lists_every_registered_connector():
    """The docstring said "two connectors shipped here" long after six were registered,
    so anyone reading it to know what the modelling tab can pull in was misled."""
    import data_connectors

    doc = data_connectors.__doc__ or ""
    for cid in data_connectors.CONNECTORS:
        assert cid in doc, f"connector {cid} is registered but absent from the docstring"


# ── A corpus built but not ranked is neither "done" nor a failed search ─────
def test_an_unranked_corpus_is_its_own_status_not_done_and_not_error():
    """When the semantic scoring produces no score (no OpenAI key, quota spent), the
    corpus IS built: the articles are in the database and readable. Two states could not
    say that. "done" claimed a ranking that does not exist, and the threshold and the
    ordering were presented as meaningful; "error" threw away a perfectly good corpus
    behind "corpus build failed". `unranked` is the third: the articles show, with a
    warning that their order is not a relevance order."""
    import inspect

    from api import pipeline

    src = inspect.getsource(pipeline._run_user_scenario_populate)
    assert '"status": "done" if _auto_ok[0] else "unranked"' in src
    assert '"st": "done" if _ok else "unranked"' in src
    # The reason travels with the status, so the interface does not have to guess it.
    assert '"reason_code": "no_scores"' in src


def test_the_interface_shows_an_unranked_corpus_instead_of_discarding_it():
    """The search loop used to `throw` on anything that was not 'done', which hid the
    corpus. 'unranked' must break out of the polling loop like 'done' does, and set the
    warning that the locales carry in both languages."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    app_src = (root / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "status === 'unranked'" in app_src
    assert "setSearchUnranked(t(\"search.corpusUnranked\"))" in app_src
    for loc in ("en", "fr"):
        text_loc = (root / "frontend" / "src" / "i18n" / "locales" / f"{loc}.ts").read_text(encoding="utf-8")
        assert "corpusUnranked:" in text_loc, f"{loc} is missing the warning string"


# ── The audit script does not quietly recompute anything ────────────────────
def test_audit_script_can_skip_the_one_check_that_triggers_work():
    """GET /clustering computes the projection when the language is not cached, so the
    "read-only" audit did start a job. --no-clustering makes the read-only claim true."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "audit_scenario.py"
    text_src = src.read_text(encoding="utf-8")
    assert "--no-clustering" in text_src
    assert "with_clustering" in text_src
    # The header must carry the caveat rather than a flat "nothing is recomputed".
    assert "nothing is generated or recomputed" not in text_src
