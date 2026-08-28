"""Auto-fetch must ADD public covariates to the user's data, not replace it.

Regression guard for a silent data-loss bug: a connector can never supply the variable
being predicted — that comes from the user's own data (ED visits, incoming calls). But
auto-fetch deactivated the uploaded dataset and activated the covariates-only frame it
had just assembled, so the scenario lost its outcome column. Worse, the validation report
was computed on the assembled columns alone, so the response said
`still_needed_user_columns: []` — "nothing is missing" — while the required outcome was
gone. Silent, and wrong in the direction that hides itself.

Needs Postgres; skips cleanly without it.
"""
import json
import os
import uuid

import pytest

pytest.importorskip("pandas")
pytest.importorskip("fastapi")


def _spec():
    return {
        "schema": "model_spec/1.0",
        "outcome": {"name": "ED visits", "machine_name": "ed_visits",
                    "task_type": "regression"},
        "features": [{"name": "Mean temperature", "machine_name": "temp_mean",
                      "source": "public_api"}],
        "algorithm": {"family": "random_forest", "metric": "r2"},
        "data_template": {
            "target_column": "ed_visits", "datetime_column": "date",
            "columns": [
                {"name": "date", "role": "datetime", "dtype": "datetime",
                 "required": True, "source": "user"},
                {"name": "ed_visits", "role": "outcome", "dtype": "float",
                 "required": True, "source": "user"},
                {"name": "temp_mean", "role": "feature", "dtype": "float",
                 "required": True, "source": "public_api"},
            ],
        },
    }


@pytest.fixture()
def client_and_scenario():
    """A TestClient plus a scenario whose model_spec is persisted, or skip."""
    import main
    from sqlalchemy import text
    from fastapi.testclient import TestClient

    sid = f"usr-{uuid.uuid4().hex[:12]}"
    try:
        with main.engine.begin() as c:
            c.execute(text(
                "INSERT INTO scenario_settings (scenario_id, variables_json) "
                "VALUES (:s, CAST(:v AS JSONB)) ON CONFLICT (scenario_id) DO UPDATE "
                "SET variables_json = EXCLUDED.variables_json"),
                {"s": sid, "v": json.dumps({"model_spec": _spec()})})
    except Exception as e:
        pytest.skip(f"no reachable database: {e}")
    try:
        yield TestClient(main.app), sid, main
    finally:
        try:
            with main.engine.begin() as c:
                c.execute(text("DELETE FROM scenario_model_dataset WHERE scenario_id = :s"),
                          {"s": sid})
                c.execute(text("DELETE FROM scenario_settings WHERE scenario_id = :s"),
                          {"s": sid})
        except Exception:
            pass


def _active_columns(main, sid):
    from sqlalchemy import text
    with main.engine.begin() as c:
        row = c.execute(text(
            "SELECT columns_json FROM scenario_model_dataset "
            "WHERE scenario_id = :s AND is_active = TRUE"), {"s": sid}).mappings().first()
    return list(row["columns_json"]) if row else None


def test_auto_fetch_preserves_the_users_uploaded_outcome(client_and_scenario, monkeypatch):
    import data_connectors

    cl, sid, main = client_and_scenario
    key = {"x-api-key": os.getenv("WRITE_API_KEY", "test-write-key")}

    # The user uploads their own outcome — no connector can produce this column.
    csv = "date,ed_visits\n" + "\n".join(f"2024-01-{d:02d},{100 + d}" for d in range(1, 29))
    up = cl.post(f"/scenarios/{sid}/model/data", headers=key,
                 files={"file": ("mine.csv", csv, "text/csv")})
    assert up.status_code == 200, up.text
    assert _active_columns(main, sid) == ["date", "ed_visits"]

    # A connector supplies only the covariate.
    monkeypatch.setattr(data_connectors, "fetch_series", lambda cid, params: [
        {"date": f"2024-01-{d:02d}", "temp_mean": 5.0 + d * 0.1} for d in range(1, 29)])

    r = cl.post(f"/scenarios/{sid}/model/data/auto-fetch", headers=key, json={
        "start_date": "2024-01-01", "end_date": "2024-01-28", "frequency": "D",
        "mappings": [{"template_column": "temp_mean",
                      "connector_id": "open-meteo-weather",
                      "connector_variable": "temp_mean"}]})
    assert r.status_code == 200, r.text
    body = r.json()

    cols = _active_columns(main, sid)
    assert "ed_visits" in cols, (
        f"auto-fetch destroyed the user's outcome column; active dataset is {cols}")
    assert "temp_mean" in cols, "the fetched covariate should have been added"
    assert body["merged_from_dataset_id"] is not None
    assert "ed_visits" in body["preserved_columns"]


def test_all_seir_scenario_needs_no_explicit_mappings(client_and_scenario, monkeypatch):
    """A template whose columns are all SEIR-derived was rejected at the door.

    `mappings` carried min_length=1, so the request 422'd before reaching the code that
    auto-adds a mapping for every source="seir" column.
    """
    import data_connectors
    from sqlalchemy import text

    cl, sid, main = client_and_scenario
    key = {"x-api-key": os.getenv("WRITE_API_KEY", "test-write-key")}

    spec = _spec()
    spec["data_template"]["columns"] = [
        {"name": "date", "role": "datetime", "dtype": "datetime",
         "required": True, "source": "user"},
        {"name": "seir_incidence", "role": "feature", "dtype": "float",
         "required": True, "source": "seir", "seir_column": "seir_incidence"},
    ]
    spec["data_template"]["target_column"] = "seir_incidence"
    with main.engine.begin() as c:
        c.execute(text("UPDATE scenario_settings SET variables_json = CAST(:v AS JSONB) "
                       "WHERE scenario_id = :s"),
                  {"s": sid, "v": json.dumps({"model_spec": spec})})

    monkeypatch.setattr(data_connectors, "fetch_series", lambda cid, params: [
        {"date": f"2024-01-{d:02d}", "seir_incidence": float(d)} for d in range(1, 29)])

    r = cl.post(f"/scenarios/{sid}/model/data/auto-fetch", headers=key, json={
        "start_date": "2024-01-01", "end_date": "2024-01-28", "frequency": "D",
        "mappings": []})
    assert r.status_code == 200, f"the all-SEIR case must be accepted: {r.status_code} {r.text}"
    assert "seir_incidence" in _active_columns(main, sid)


def test_nothing_to_fetch_is_an_explicit_error(client_and_scenario):
    """No mappings AND no SEIR columns is a real error, not a silent empty fetch."""
    from sqlalchemy import text

    cl, sid, main = client_and_scenario
    key = {"x-api-key": os.getenv("WRITE_API_KEY", "test-write-key")}

    spec = _spec()
    spec["data_template"]["columns"] = [
        {"name": "date", "role": "datetime", "dtype": "datetime",
         "required": True, "source": "user"},
        {"name": "ed_visits", "role": "outcome", "dtype": "float",
         "required": True, "source": "user"},
    ]
    with main.engine.begin() as c:
        c.execute(text("UPDATE scenario_settings SET variables_json = CAST(:v AS JSONB) "
                       "WHERE scenario_id = :s"),
                  {"s": sid, "v": json.dumps({"model_spec": spec})})

    r = cl.post(f"/scenarios/{sid}/model/data/auto-fetch", headers=key, json={
        "start_date": "2024-01-01", "end_date": "2024-01-28", "mappings": []})
    assert r.status_code == 422
    assert "Rien à récupérer" in r.text
