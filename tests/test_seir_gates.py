"""Tests for the three gates that decide whether a SEIR projection may be served.

These lock in an audit finding: `epidemic_parameters.applicable` was never checked, so a
scenario the extraction had explicitly flagged as NON-transmissible still rendered a full
compartmental projection as soon as one stray numeric parameter survived normalisation —
a clinician would see an epidemic curve for an oncology scenario. A second gate stops the
engine's hard-coded R0 = 2.5 fallback from being presented as literature-derived.

Pure: `_get_model_spec` is monkeypatched, so no database is needed (the geography lookup
and the observed-series overlay both fail closed onto defaults when the DB is unreachable).
"""
import main


def _param(value, lo=None, hi=None, unit="", provenance=(1, 2)):
    return {"value": value, "ci_low": lo, "ci_high": hi, "unit": unit,
            "n_studies": len(provenance), "provenance": list(provenance)}


def _spec(applicable, params, disease="X"):
    return {"epidemic_parameters": {"applicable": applicable, "disease": disease,
                                    "params": params}}


def _payload(monkeypatch, spec, **kw):
    monkeypatch.setattr(main, "_get_model_spec", lambda _sid: spec)
    kw.setdefault("days", 120)
    kw.setdefault("n_samples", 40)
    return main._seir_projection_payload("usr-test00000001", **kw)


# ── gate 1: the extraction said this is not a transmissible disease ──────────
def test_non_transmissible_scenario_is_refused_even_with_a_stray_parameter():
    import pytest
    mp = pytest.MonkeyPatch()
    try:
        out = _payload(mp, _spec(False, {"cfr": _param(0.23)}, "cancer du sein"))
    finally:
        mp.undo()
    assert out["applicable"] is False
    assert "NON transmissible" in out["reason"]
    assert "summary" not in out          # no curve, no attack rate, nothing to misread


# ── gate 2: a parameter that does not drive the dynamics is not enough ───────
def test_missing_r0_and_beta_is_refused_instead_of_falling_back_to_2_5():
    import pytest
    mp = pytest.MonkeyPatch()
    try:
        out = _payload(mp, _spec(True, {"cfr": _param(0.6)}, "Ebola"))
    finally:
        mp.undo()
    assert out["applicable"] is False
    assert out["missing"] == ["r0"]
    assert out["available_parameters"] == ["cfr"]
    assert "2.5" not in str(out)         # the fabricated value never reaches the client


def test_beta_alone_satisfies_the_transmission_gate():
    import pytest
    mp = pytest.MonkeyPatch()
    try:
        out = _payload(mp, _spec(True, {"beta": _param(0.3),
                                        "infectious_period_days": _param(5.0, unit="days")}))
    finally:
        mp.undo()
    assert out["applicable"] is True


# ── the escape hatch: an explicit user override is a conscious decision ──────
def test_user_override_unlocks_the_gates_and_is_labelled_as_such():
    import pytest
    mp = pytest.MonkeyPatch()
    try:
        out = _payload(mp, _spec(True, {"cfr": _param(0.6)}, "Ebola"),
                       overrides={"r0": {"value": 1.8}})
    finally:
        mp.undo()
    assert out["applicable"] is True
    assert out["forced"] is True             # the UI must not present this as sourced
    assert out["r0_source"] == "user"        # ... and certainly not as "literature"
    assert out["overrides_applied"] == {"r0": 1.8}


def test_override_also_unlocks_an_explicitly_non_transmissible_scenario():
    import pytest
    mp = pytest.MonkeyPatch()
    try:
        out = _payload(mp, _spec(False, {"cfr": _param(0.2)}),
                       overrides={"r0": {"value": 2.0},
                                  "infectious_period_days": {"value": 5.0}})
    finally:
        mp.undo()
    assert out["applicable"] is True and out["forced"] is True


# ── the happy path still works, and reports honestly ─────────────────────────
def test_fully_parameterised_scenario_projects_with_an_ordered_band():
    import pytest
    mp = pytest.MonkeyPatch()
    try:
        out = _payload(mp, _spec(True, {
            "r0": _param(1.3, 1.2, 1.4),
            "incubation_period_days": _param(2.0, 1.4, 2.6, "days"),
            "infectious_period_days": _param(4.0, 3.0, 5.0, "days"),
            "cfr": _param(0.001, 0.0005, 0.002, "proportion"),
        }, "Influenza"), days=180)
    finally:
        mp.undo()
    assert out["applicable"] is True
    assert out["forced"] is False
    assert out["r0_source"] == "literature"
    assert out["model"] == "SEIRD"
    for series in out["series"].values():
        for t in range(len(series["median"])):
            assert series["lower"][t] <= series["median"][t] <= series["upper"][t] + 1e-9


def test_shipped_demo_scenario_actually_has_a_seir_model():
    """The demo is about influenza; its SEIR tab used to show an empty state."""
    import pytest
    import demo_seed
    mp = pytest.MonkeyPatch()
    try:
        out = _payload(mp, demo_seed.demo_model_spec(), days=180)
    finally:
        mp.undo()
    assert out["applicable"] is True
    assert out["disease"] == "Influenza (seasonal)"
    assert out["summary"]["peak_incidence"]["median"] > 0


def test_demo_parameters_claim_no_provenance_they_do_not_have():
    """Demo values are published orders of magnitude, not corpus extractions."""
    import demo_seed
    epi = demo_seed.demo_model_spec()["epidemic_parameters"]
    assert epi["applicable"] is True and epi["is_demo"] is True
    assert epi["note"]
    for name, blk in epi["params"].items():
        assert blk["provenance"] == [], f"{name} claims provenance it does not have"
        assert blk["n_studies"] == 0
