"""Pure-function tests for the editable model spec (main.py helpers).

`_derive_data_template` and `_coerce_family_for_task` are pure; `main` is
import-only (env set by conftest). These lock in the two invariants the editable
spec relies on:
  1. the upload data_template is ALWAYS derived from the current outcome/features,
     so removing/adding a variable never leaves the validator expecting a phantom
     column (or missing a real one);
  2. an algorithm family chosen for the wrong task is coerced to a compatible one
     (logistic↔linear), so a spec edit can't produce an untrainable pairing.
"""
import pytest

# main imports pandas/sklearn lazily but is importable without them; keep parity
# with the other suite so a wheel-less runner skips cleanly instead of erroring.
pytest.importorskip("pandas")

import main


def _feat(mn, dtype="float", importance="medium"):
    return {"machine_name": mn, "dtype": dtype, "importance": importance,
            "source": "user", "name": mn.upper()}


# ── data_template derivation ─────────────────────────────────────────────────
def test_derive_data_template_matches_features_in_order():
    outcome = {"machine_name": "y", "name": "Outcome", "task_type": "regression"}
    features = [_feat("x1", "float", "high"), _feat("x2", "int", "low")]
    tmpl = main._derive_data_template(outcome, features)
    assert tmpl["target_column"] == "y"
    assert [c["name"] for c in tmpl["columns"]] == ["y", "x1", "x2"]   # outcome first
    req = {c["name"]: c["required"] for c in tmpl["columns"]}
    assert req["x1"] is True and req["x2"] is False                    # required == high importance
    assert set(tmpl["user_columns"]) == {"y", "x1", "x2"}


def test_derive_data_template_target_dtype_follows_task():
    feats = [_feat("x")]
    for task, dtype in (("classification", "category"), ("regression", "float"),
                        ("count", "int"), ("survival", "float")):
        tmpl = main._derive_data_template({"machine_name": "y", "task_type": task}, feats)
        assert tmpl["columns"][0]["dtype"] == dtype


def test_derive_data_template_has_no_phantom_columns_after_removal():
    # Simulate the editor dropping a feature: the template must not reference it.
    outcome = {"machine_name": "y", "task_type": "regression"}
    tmpl = main._derive_data_template(outcome, [_feat("keep")])
    assert {c["name"] for c in tmpl["columns"]} == {"y", "keep"}


def test_derive_data_template_public_column_partitioning():
    outcome = {"machine_name": "y", "task_type": "regression"}
    features = [
        {"machine_name": "temp", "dtype": "float", "importance": "medium",
         "source": "public_api", "public_provider": "open-meteo", "name": "Temp"},
        _feat("dose"),
    ]
    tmpl = main._derive_data_template(outcome, features)
    assert "temp" in tmpl["public_columns"]
    assert "dose" in tmpl["user_columns"] and "temp" not in tmpl["user_columns"]


# ── family ↔ task coercion ───────────────────────────────────────────────────
def test_coerce_family_swaps_linear_logistic_by_task():
    assert main._coerce_family_for_task("logistic_regression", "regression") == "linear_regression"
    assert main._coerce_family_for_task("linear_regression", "classification") == "logistic_regression"
    assert main._coerce_family_for_task("elasticnet", "classification") == "logistic_regression"


def test_coerce_family_keeps_compatible_choices():
    assert main._coerce_family_for_task("lightgbm", "regression") == "lightgbm"
    assert main._coerce_family_for_task("xgboost", "classification") == "xgboost"
    assert main._coerce_family_for_task("random_forest", "regression") == "random_forest"
    assert main._coerce_family_for_task("gradient_boosting", "classification") == "gradient_boosting"


def test_coerce_family_unknown_falls_back_to_gradient_boosting():
    assert main._coerce_family_for_task("not_a_real_family", "classification") == "gradient_boosting"


def test_boosting_families_are_selectable_in_spec():
    # PR #1 made lightgbm/xgboost trainable; the editable spec must expose them.
    assert "lightgbm" in main._ALGO_FAMILIES
    assert "xgboost" in main._ALGO_FAMILIES


# ── The SEIR inputs are part of the diff, and the change report reads them ───
def test_the_spec_diff_covers_the_seir_inputs():
    """_diff_model_spec compared the outcome, the features and the algorithm but not
    epidemic_parameters, which is what SEIR actually reads. A regeneration could take R0
    from 2.1 to 4.8, change the projected curve completely, and the diff answered "no
    changes". It is the one part of the spec an epidemiologist reads as a number."""
    d = main._diff_model_spec(
        {"epidemic_parameters": {"applicable": True,
                                 "params": {"r0": {"value": 2.1}, "cfr": {"value": 0.01}}}},
        {"epidemic_parameters": {"applicable": True,
                                 "params": {"r0": {"value": 4.8},
                                            "incubation_period": {"value": 5.2}}}})
    epi = d["epidemic_parameters"]
    assert epi["params_added"] == ["incubation_period"]
    assert epi["params_removed"] == ["cfr"]
    assert epi["params_shifted"][0]["param"] == "r0"
    assert epi["params_shifted"][0]["old"] == 2.1 and epi["params_shifted"][0]["new"] == 4.8
    assert d["has_changes"] is True
    assert d["summary"]["epidemic_parameters_changed"] is True


def test_a_negligible_parameter_move_is_not_reported_as_a_change():
    """A decimal more in one article is extraction noise. Reporting it every time would
    teach the reader to ignore the report."""
    d = main._diff_model_spec(
        {"epidemic_parameters": {"params": {"r0": {"value": 2.00}}}},
        {"epidemic_parameters": {"params": {"r0": {"value": 2.04}}}})   # 2 %
    assert d["epidemic_parameters"]["params_shifted"] == []
    assert d["has_changes"] is False
    # ... but a real move is.
    d2 = main._diff_model_spec(
        {"epidemic_parameters": {"params": {"r0": {"value": 2.00}}}},
        {"epidemic_parameters": {"params": {"r0": {"value": 2.40}}}})   # 20 %
    assert d2["epidemic_parameters"]["params_shifted"][0]["relative"] == 0.2


def test_a_scenario_becoming_projectable_at_all_is_a_change():
    """`applicable` decides whether the scenario has a SEIR projection at all, so it
    flipping is a change on its own even with identical parameters."""
    d = main._diff_model_spec(
        {"epidemic_parameters": {"applicable": False, "params": {}}},
        {"epidemic_parameters": {"applicable": True, "params": {}}})
    assert d["epidemic_parameters"]["applicable_changed"] is True
    assert d["has_changes"] is True


def test_the_change_report_lists_what_moved_and_applies_nothing(monkeypatch):
    """The report compares two specs that were really generated, so unlike the digest's
    cheap signals it can say a conclusion moved. It must never be the thing that moves
    it: the proposal stays in its own slot until accepted."""
    from fastapi.testclient import TestClient
    from conftest import patch_app

    class _Row(dict):
        pass

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a, **k): return self
        def mappings(self): return self
        def first(self):
            return {
                "variables_json": {"_meta": {"n_articles": 100},
                                   "model_spec": {"epidemic_parameters": {
                                       "applicable": True, "params": {"r0": {"value": 2.0}}}}},
                "variables_proposal_json": {"_meta": {"n_articles": 140},
                                            "model_spec": {"epidemic_parameters": {
                                                "applicable": True,
                                                "params": {"r0": {"value": 3.0}}}}},
                "proposal_generated_at": None, "variables_generated_at": None,
            }

    class _Engine:
        def connect(self): return _Conn()

    patch_app(monkeypatch, "engine", _Engine())
    main._SPEC_PROPOSAL_JOBS.pop("usr-x", None)
    r = TestClient(main.app).get("/scenarios/usr-x/change-report")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ready" and body["has_changes"] is True
    assert any("r0" in c for c in body["changes"])
    assert body["corpus"] == {"active_articles": 100, "proposal_articles": 140, "delta": 40}
    # The report is a report.
    assert body["applied"] is False and "validate" in body["apply_endpoint"]
