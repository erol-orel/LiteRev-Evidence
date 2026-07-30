"""Pure-logic tests for demo_seed.py — no DB, no pandas/sklearn (those are imported
lazily inside train_demo, so importing the module and building the spec is stdlib-only).

The end-to-end DB seeding (main._seed_demo_scenarios) is verified separately against a
real Postgres; here we lock down the model_spec shape the dashboard/monitor read back."""
import json

import demo_seed


def test_demo_scenario_identity_is_stable():
    # Stable id → the startup seed stays idempotent across boots.
    assert demo_seed.DEMO_SCENARIO_ID == "usr-deadbeef0001"
    assert demo_seed.DEMO_SCENARIO_NAME and demo_seed.DEMO_SCENARIO_QUERY


def test_model_spec_shape_is_dashboard_ready():
    spec = demo_seed.demo_model_spec()
    assert spec["schema"] == "model_spec/1.0"
    assert spec["outcome"]["machine_name"] == "flu_a_load"
    assert spec["outcome"]["task_type"] == "regression"
    assert spec["algorithm"]["family"] == "random_forest"
    dt = spec["data_template"]
    assert dt["target_column"] == "flu_a_load"
    # the outcome column is present in the template and marked as such
    roles = {c["name"]: c["role"] for c in dt["columns"]}
    assert roles["flu_a_load"] == "outcome"
    feats = {f["machine_name"] for f in spec["features"]}
    assert feats == {"flu_b_load", "rsv_load", "sars_cov2_load", "season_sin", "season_cos"}


def test_variables_json_carries_model_spec_and_is_jsonb_safe():
    vj = demo_seed.demo_variables_json()
    # _get_model_spec reads variables_json['model_spec']
    assert vj["model_spec"]["outcome"]["machine_name"] == "flu_a_load"
    json.dumps(vj)   # must serialise cleanly into a JSONB column


def test_dataset_path_points_at_committed_trial_csv():
    p = demo_seed.dataset_path("/repo")
    assert p.endswith("scripts/trial_output/ch-influenza_dataset.csv")
