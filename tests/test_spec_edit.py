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
