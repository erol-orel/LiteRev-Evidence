"""Every search is exhaustive: the screen offers no depth, and the server has none to choose.

The search screen offered two depths, "exhaustive" and "standard", each with its line of
explanation. Production lifts the standard cap to 100,000 (LIVE_MAX_PER_SOURCE), so both lines
read "up to 100,000 per database" and nobody could tell them apart. What was left of the
difference only ever cost records: the standard search kept OpenAlex's 2,000 best matches
instead of every title and abstract match, sent a shortened query (fewer synonyms) to the
databases a long query does not fit, and had 10 minutes instead of 45. Asked on 13 October
2026: keep the exhaustive search only. These tests hold both sides: the endpoints launch an
exhaustive search whatever an older caller sends, the job and the store have no mode left,
and the screen shows one line, with the server's own caps and time budget.
"""
import inspect
import pathlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from conftest import patch_app  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SID = "usr-depthscreen01"
HDR = {"X-API-Key": "test-write-key"}


@pytest.fixture()
def scenario(db_conn):
    try:
        main._ensure_user_scenarios_table()
        with main.engine.begin() as c:
            c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
            c.execute(text("INSERT INTO user_scenarios (id, name, query, mode, filters, article_count) "
                           "VALUES (:s, 'Depth', 'avian influenza', 'boolean', '{}', 0)"), {"s": SID})
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"main.engine cannot reach the database ({type(e).__name__})")
    yield SID
    with main.engine.begin() as c:
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_the_populate_endpoint_launches_an_exhaustive_search_whatever_it_is_sent(scenario, monkeypatch):
    from api.search import LIVE_MAX_PER_SOURCE
    seen: list[dict] = []

    def _launch(scenario_id, query, filters, max_results, include_live=True, lang=None,
                force_live=False, auto_pipeline=True):
        seen.append({"max_results": max_results, "include_live": include_live})
        return "started"

    patch_app(monkeypatch, "_launch_populate_job", _launch)
    client = _client()
    # An older client still sending a mode is not refused, and gets the one search there is.
    for query in ("?exhaustive=false&include_live=true", "?exhaustive=true", ""):
        r = client.post(f"/user-scenarios/{SID}/populate{query}", headers=HDR)
        assert r.status_code == 200, r.text
        assert r.json()["search_mode"] == "exhaustive", r.json()
        # What the screen sends: no cap, the server's applies.
        assert seen[-1]["max_results"] == LIVE_MAX_PER_SOURCE


def test_the_pipeline_endpoint_takes_no_mode_either(scenario, monkeypatch):
    from api.search import LIVE_MAX_PER_SOURCE
    launched: list[tuple] = []
    patch_app(monkeypatch, "_launch_full_pipeline",
              lambda sid, max_results=LIVE_MAX_PER_SOURCE, lang=None: launched.append((sid, max_results)) or "started")
    r = _client().post(f"/user-scenarios/{SID}/pipeline?exhaustive=false&lang=en", headers=HDR)
    assert r.status_code == 200, r.text
    assert launched[-1] == (SID, LIVE_MAX_PER_SOURCE)


def test_no_mode_is_left_to_choose_or_to_read():
    """The job, its launcher and the endpoints have no `exhaustive` parameter, and the
    scenario's stored mode is no longer read: a scenario saved as standard when there were
    two is searched again like any other."""
    from api import pipeline as P
    from api import scenario_store as St
    from api import scenarios as Sc
    from api.search import LIVE_MAX_PER_SOURCE
    params = inspect.signature(P._run_user_scenario_populate).parameters
    assert "exhaustive" not in params
    assert params["max_results"].default == LIVE_MAX_PER_SOURCE
    assert "exhaustive" not in inspect.signature(Sc._launch_populate_job).parameters
    assert "exhaustive" not in inspect.signature(Sc.populate_user_scenario).parameters
    assert "exhaustive" not in inspect.signature(Sc.start_user_scenario_pipeline).parameters
    assert not hasattr(St, "scenario_search_mode") and not hasattr(St, "set_scenario_search_mode")
    src = inspect.getsource(P._run_user_scenario_populate)
    assert '_search_mode = "exhaustive"' in src
    assert "if not exhaustive" not in src and "if exhaustive" not in src


def test_the_screen_offers_one_search_and_says_what_it_takes():
    app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    for gone in ("exhaustiveSearch", 'name="search-depth"', "search.depthExhaustive", "search.depthStandard",
                 "standard_per_source", "exhaustive_per_source"):
        assert gone not in app, gone
    assert "populateUserScenario(sid, { includeLive, excludePreprints })" in app
    assert "startUserScenarioPipeline(scenarioId)" in app
    # The caps and the time budget come from the server (`/search/limits`), not from a
    # 2,000 and a 20,000 written in the texts.
    assert "fetchSearchLimits()" in app
    for used in ("searchLimits.per_source", "searchLimits.ranked_per_source", "searchLimits.budget_s"):
        assert used in app, used
    api_ts = (ROOT / "frontend" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "params.set('exhaustive'" not in api_ts
    assert "maxResults = 100000" not in api_ts
    for loc in ("fr", "en"):
        txt = (ROOT / "frontend" / "src" / "i18n" / "locales" / f"{loc}.ts").read_text(encoding="utf-8")
        assert "depthExhaustive" not in txt and "depthStandard" not in txt and "depthLabel" not in txt, loc
        lines = txt.splitlines()
        i = next(k for k, line in enumerate(lines) if "includeLiveHint:" in line)
        hint = " ".join(lines[i:i + 2])
        assert all(m in hint for m in ("{n}", "{ranked}", "{minutes}", "Crossref")), (loc, hint)
        assert "000" not in hint, f"{loc}: a cap written in the text ({hint})"


@pytest.mark.parametrize("live,exhaustive,ranked,expected", [
    # The defaults: every match up to 20,000 per database, 2,000 for the ranked passes.
    (2000, 20000, 2000, (20000, 2000)),
    # Production: LIVE_MAX_PER_SOURCE at 100,000 lifts the cap; the ranked passes keep theirs.
    (100000, 20000, 2000, (100000, 2000)),
    # A lower LIVE_MAX_PER_SOURCE bounds the ranked passes, never the cap under 20,000.
    (500, 20000, 2000, (20000, 500)),
])
def test_the_server_says_which_caps_and_budget_it_applies(monkeypatch, live, exhaustive, ranked, expected):
    from api import sources as S
    monkeypatch.setattr(S, "LIVE_MAX_PER_SOURCE", live)
    monkeypatch.setattr(S, "EXHAUSTIVE_MAX_PER_SOURCE", exhaustive)
    monkeypatch.setattr(S, "RANKED_PASS_MAX", ranked)
    monkeypatch.setattr(S, "POPULATE_FEDERATION_BUDGET", 600.0)
    monkeypatch.setattr(S, "EXHAUSTIVE_FEDERATION_BUDGET", 2700.0)
    r = _client().get("/search/limits")
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["per_source"], j["ranked_per_source"]) == expected, j
    assert j["budget_s"] == 2700, j
    assert "standard_per_source" not in j and "exhaustive_per_source" not in j
