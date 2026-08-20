"""Tests for the GESICA outcome-template catalogue (pure — no DB/FastAPI)."""
import json

import outcome_templates as ot


def test_catalogue_ids_and_wellformed():
    ids = {t["id"] for t in ot.as_list()}
    assert ids == {"ed_overload", "bed_occupancy", "call_volume", "call_surge"}
    for t in ot.as_list():
        o = t["outcome"]
        assert o["name"] and o["machine_name"]
        assert o["task_type"] in ("classification", "regression", "count")
        assert t["algorithm"]["family"]
        assert t["features"] and all(f["machine_name"] and f["dtype"] for f in t["features"])
        json.dumps(t)                       # JSON-safe (served + stored)


def test_classification_template_has_positive_class():
    eo = ot.get("ed_overload")
    assert eo["outcome"]["task_type"] == "classification"
    assert eo["outcome"]["positive_class"] == "surcharge"


def test_surge_template_uses_extremal_rf_with_quantile():
    cs = ot.get("call_surge")
    assert cs["algorithm"]["family"] == "extremal_rf"
    assert cs["algorithm"]["quantile"] == 0.9
    assert cs["outcome"]["task_type"] == "regression"


def test_get_unknown_is_none_and_get_returns_a_copy():
    assert ot.get("nope") is None
    a = ot.get("call_volume")
    a["features"].append({"x": 1})
    assert len(ot.get("call_volume")["features"]) < len(a["features"])   # deep-ish copy, catalogue intact
