"""The app must do exactly what it says it does.

Each test here pins one place where a claim and the behaviour behind it had drifted
apart. They are cheap, pure tests: what they guard is agreement between two parts of the
code (a writer and a reader, a docstring and a response, a promise and its prerequisite),
which is exactly the kind of thing that silently rots between two commits.
"""
import pytest

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


# ── One place says which model does which job ───────────────────────────────
def test_no_module_hardcodes_a_model_name():
    """The model was written out at 27 call sites. Upgrading meant finding and editing
    all 27, so in practice nobody did, and two of them sat on gpt-4o-mini long after
    everything else had moved to 4.1. The name now comes from `llm_usage.model_for`,
    which reads an environment variable, so a deployment changes a model without a code
    change and `/health` can be asked what it is actually running."""
    import pathlib
    import re

    api = pathlib.Path(__file__).resolve().parent.parent / "api"
    # Three spellings, because the first version of this test only knew the first, and
    # `"model": "gpt-4.1"` in the evidence brief's provenance walked straight past it and
    # went on stamping every brief with the name of a model that had not written it for
    # four days. A dict value is a hardcoded model name like any other.
    patterns = (r'model\s*=\s*"(?:gpt-|text-embedding-|o[0-9]-)',
                r'"model"\s*:\s*"(?:gpt-|text-embedding-|o[0-9]-)',
                r"'model'\s*:\s*'(?:gpt-|text-embedding-|o[0-9]-)")
    bad: list[str] = []
    for path in sorted(api.glob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(re.search(p, line) for p in patterns):
                bad.append(f"{path.name}:{n}: {line.strip()[:70]}")
    assert not bad, "a model name is hardcoded again:\n  " + "\n  ".join(bad)


def test_every_role_resolves_and_is_overridable(monkeypatch):
    from llm_usage import model_for, models_in_use

    roles = models_in_use()
    assert set(roles) == {"bulk", "write", "chat", "embedding"}
    assert all(isinstance(v, str) and v for v in roles.values())
    # The assistant's two calls were the stragglers; its default must not be the old one.
    assert roles["chat"] != "gpt-4o-mini"
    # Read at CALL time, so a restart with a new variable is enough.
    monkeypatch.setenv("LLM_MODEL_BULK", "some-newer-model")
    assert model_for("bulk") == "some-newer-model"
    with pytest.raises(ValueError):
        model_for("not-a-role")


def test_health_says_which_models_are_running_and_in_what_shape():
    """The names are environment-overridable, so the only way to know what a deployment
    is running is to ask it, and any measurement of extraction quality is a measurement
    of these exact models - at this exact reasoning effort, since that decides whether the
    temperature the call sites ask for was sent or dropped."""
    import inspect

    from api import system

    src = inspect.getsource(system.health)
    assert "models_in_use" in src
    assert "request_policy" in src


def test_the_token_ceiling_the_call_sites_pass_reaches_the_model_under_some_name():
    """The gpt-5 generation answers `max_tokens` with a 400. Twenty call sites pass it,
    and every one of them is wrapped in `except Exception`, so the symptom of getting this
    wrong is not an error page: it is PICO that stops filling, briefs that come back empty
    and an assistant that apologises, with /health still green.

    Asserted as behaviour rather than as a name, so that it keeps holding whatever the
    models become."""
    from llm_usage import legacy_request_shape, model_for, shape_chat_kwargs

    for role in ("bulk", "write", "chat"):
        sent = shape_chat_kwargs({"model": model_for(role), "max_tokens": 800,
                                  "temperature": 0, "seed": 42})
        if legacy_request_shape(sent["model"]):
            assert sent["max_tokens"] == 800
        else:
            assert "max_tokens" not in sent and sent["max_completion_tokens"] == 800
        # Temperature 0 and seed 42 are what make an extraction repeatable, and they are
        # only accepted while the model is not reasoning. If a default ever drops them,
        # the gold standard's figures stop being reproducible and must say so.
        assert sent.get("seed") == 42
        assert "temperature" in sent or sent.get("reasoning_effort") not in (None, "none")


def test_no_call_site_in_the_api_sends_a_parameter_its_model_would_refuse():
    """Read the real call sites out of `api/*.py` and put what each one passes through the
    shaper, so this cannot drift away from the code the way a hand-written list would.

    The parameter sets here are the ones in production. If a new call site passes
    `max_tokens` to a gpt-5 model, or a temperature while the model is reasoning, this
    fails at the call site instead of silently in a worker's log."""
    import pathlib
    import re as _re

    from llm_usage import legacy_request_shape, model_for, shape_chat_kwargs

    api = pathlib.Path(__file__).resolve().parent.parent / "api"
    sites: list[tuple[str, str, dict]] = []
    for path in sorted(api.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        for match in _re.finditer(r"chat\.completions\.create\(", src):
            # Scan to the matching parenthesis: the call spans several lines.
            depth, i = 1, match.end()
            while i < len(src) and depth:
                depth += {"(": 1, ")": -1}.get(src[i], 0)
                i += 1
            body = src[match.end():i - 1]
            # Top-level `name=` only: nested dicts and lists carry their own.
            names, depth = [], 0
            for token in _re.finditer(r"[(\[{]|[)\]}]|(\w+)\s*=", body):
                if token.group(1) and depth == 0:
                    names.append(token.group(1))
                elif token.group(0) in "([{":
                    depth += 1
                elif token.group(0) in ")]}":
                    depth -= 1
            role = (_re.search(r'_model\("(\w+)"\)', body) or [None, "chat"])[1]
            sites.append((path.name, role, {n: "x" for n in names}))

    assert len(sites) >= 10, f"only found {len(sites)} call sites; the scan is broken"
    for name, role, passed in sites:
        passed["model"] = model_for(role)
        sent = shape_chat_kwargs(passed)
        where = f"{name} ({role} -> {sent['model']})"
        if not legacy_request_shape(sent["model"]):
            assert "max_tokens" not in sent, f"{where} would be refused for max_tokens"
            if sent.get("reasoning_effort") not in (None, "none"):
                assert "temperature" not in sent, f"{where} reasons AND sets a temperature"
        if "max_tokens" in passed or "max_completion_tokens" in passed:
            assert "max_completion_tokens" in sent or "max_tokens" in sent, \
                f"{where} lost its ceiling in translation"


def test_a_deployment_can_prove_its_models_answer_without_exercising_the_app():
    """A model switch is otherwise unverifiable: nothing reports a refused request, so the
    only test is to use each feature and read the logs. POST /llm-selftest makes one
    minimal call per role instead."""
    import inspect

    from api import system

    src = inspect.getsource(system.llm_selftest)
    for role in ("bulk", "write", "chat", "embedding"):
        assert f'"{role}"' in src or f"'{role}'" in src
    # It must send the shape the application sends, or it proves nothing about it.
    assert "max_tokens=16" in src and "temperature=0" in src
    # And it must be the write key, because it spends money.
    assert "require_api_key" in src


# ── One module decides where configuration comes from ───────────────────────
def test_no_module_carries_its_own_list_of_env_file_paths():
    """There were seven loaders, each with its own list in its own order, and
    `/etc/literev-api.env` - the file the systemd unit passes to the service - was first in
    none of them. The service survived on the fact that systemd populates the environment
    before Python starts and no loader overrides it; the cost landed elsewhere, as a key
    read out of a file that the API running on that same machine rejected.

    So the paths live in exactly one module. A new loader would reintroduce the problem one
    script at a time, which is how it happened the first time."""
    import ast
    import pathlib

    from env_files import SECONDARY

    root = pathlib.Path(__file__).resolve().parent.parent
    allowed = {"env_files.py"}                        # and nothing else, ever
    bad: list[str] = []
    for path in sorted(root.glob("*.py")) + sorted(root.glob("api/*.py")) \
            + sorted(root.glob("scripts/*.py")) + sorted(root.glob("tools/*.py")):
        if path.name in allowed or "archive" in str(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Docstrings and comments may discuss these paths freely - that is how the history
        # gets recorded. A STRING LITERAL holding one is either a loader list or an
        # instruction telling an operator to edit the wrong file.
        docstrings = {id(ast.get_docstring(node, clean=False))
                      for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                           ast.AsyncFunctionDef))}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node.value) in docstrings:
                continue
            for secondary in SECONDARY:
                if secondary in node.value:
                    bad.append(f"{path.relative_to(root)}:{node.lineno}: {secondary}")
    assert not bad, ("a non-canonical env file path is written into the code, as a loader "
                     "list or as advice to edit a file that is not the one in force:\n  "
                     + "\n  ".join(bad))


def test_the_api_loads_its_environment_through_the_one_loader():
    """api/core.py had its own parser and its own five-path list, and that list did not
    contain the canonical file at all."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent / "api" / "core.py").read_text(
        encoding="utf-8")
    assert "from env_files import load_env" in src
    assert "def _load_env_file" not in src, "the second parser is back"


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
