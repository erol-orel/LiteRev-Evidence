"""La recherche exhaustive, proposée par l'écran de recherche, et choisie par défaut.

Le serveur avait un mode exhaustif (toutes les notices que chaque base apparie, jusqu'à
EXHAUSTIVE_MAX_PER_SOURCE par base) que l'écran n'offrait jamais : il envoyait toujours
`max_results=2000`. L'écran envoie désormais `exhaustive` et plus aucun plafond. Ces tests
tiennent le côté serveur : le paramètre passe jusqu'au job, il est enregistré sur le
scénario, le pipeline lancé à l'épinglage l'enregistre aussi, et un appelant de l'API qui
ne dit rien reste en standard.
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


def _mode() -> str:
    from api.scenario_store import scenario_search_mode
    return scenario_search_mode(SID)


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_the_populate_endpoint_passes_the_mode_through_and_records_it(scenario, monkeypatch):
    from api.search import LIVE_MAX_PER_SOURCE
    seen: list[dict] = []

    def _launch(scenario_id, query, filters, max_results, include_live=True, lang=None,
                force_live=False, exhaustive=None, auto_pipeline=True):
        seen.append({"max_results": max_results, "exhaustive": exhaustive})
        return "started"

    patch_app(monkeypatch, "_launch_populate_job", _launch)
    client = _client()
    r = client.post(f"/user-scenarios/{SID}/populate?exhaustive=true&include_live=true", headers=HDR)
    assert r.status_code == 200, r.text
    assert seen[-1]["exhaustive"] is True
    assert r.json()["search_mode"] == "exhaustive" and _mode() == "exhaustive"
    # Ce que l'écran envoie : aucun plafond, celui du serveur s'applique.
    assert seen[-1]["max_results"] == LIVE_MAX_PER_SOURCE
    # Sans le paramètre, le mode enregistré reste, et le job relit le scénario.
    client.post(f"/user-scenarios/{SID}/populate", headers=HDR)
    assert seen[-1]["exhaustive"] is None and _mode() == "exhaustive"
    client.post(f"/user-scenarios/{SID}/populate?exhaustive=false", headers=HDR)
    assert seen[-1]["exhaustive"] is False and _mode() == "standard"


def test_an_api_caller_that_says_nothing_stays_standard(scenario, monkeypatch):
    patch_app(monkeypatch, "_launch_populate_job", lambda *a, **k: "started")
    r = _client().post(f"/user-scenarios/{SID}/populate", headers=HDR)
    assert r.status_code == 200 and r.json()["search_mode"] == "standard"
    # Et le job lui-même : son plafond par défaut est celui de l'application, et un mode
    # absent se lit sur le scénario (standard s'il n'en a pas).
    from api import pipeline as P
    from api.search import LIVE_MAX_PER_SOURCE
    params = inspect.signature(P._run_user_scenario_populate).parameters
    assert params["max_results"].default == LIVE_MAX_PER_SOURCE
    assert params["exhaustive"].default is None


def test_the_pipeline_records_the_mode_of_a_scenario_still_to_build(scenario, monkeypatch):
    """« Enregistrer comme scénario » après une recherche multi-facettes crée un scénario
    neuf, que le pipeline construit : sans le mode de l'écran, il l'aurait construit en
    standard après une recherche exhaustive."""
    from api.search import LIVE_MAX_PER_SOURCE
    launched: list[tuple] = []
    patch_app(monkeypatch, "_launch_full_pipeline",
              lambda sid, max_results=LIVE_MAX_PER_SOURCE, lang=None: launched.append((sid, max_results)) or "started")
    client = _client()
    r = client.post(f"/user-scenarios/{SID}/pipeline?exhaustive=true&lang=en", headers=HDR)
    assert r.status_code == 200, r.text
    assert _mode() == "exhaustive" and launched[-1] == (SID, LIVE_MAX_PER_SOURCE)
    client.post(f"/user-scenarios/{SID}/pipeline", headers=HDR)
    assert _mode() == "exhaustive", "sans le paramètre, le scénario garde son mode"
    client.post(f"/user-scenarios/{SID}/pipeline?exhaustive=false", headers=HDR)
    assert _mode() == "standard"


def test_the_screen_offers_both_depths_and_sends_the_one_chosen():
    app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "const [exhaustiveSearch, setExhaustiveSearch] = useState(true);" in app, "exhaustive par défaut"
    assert "exhaustive: exhaustiveSearch" in app
    assert "maxResults: 2000" not in app, "l'écran n'envoie plus de plafond"
    for key in ("depthExhaustive", "depthExhaustiveHint", "depthStandard", "depthStandardHint"):
        assert f'"search.{key}"' in app, key
    api_ts = (ROOT / "frontend" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "params.set('exhaustive'" in api_ts
    assert "maxResults = 100000" not in api_ts
    for loc in ("fr", "en"):
        txt = (ROOT / "frontend" / "src" / "i18n" / "locales" / f"{loc}.ts").read_text(encoding="utf-8")
        hint = next(line for line in txt.splitlines() if "depthExhaustiveHint:" in line)
        assert "20" in hint and "000" in hint, hint
        hint = next(line for line in txt.splitlines() if "depthStandardHint:" in line)
        assert "2" in hint and "000" in hint, hint
