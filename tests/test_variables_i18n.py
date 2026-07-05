"""Tests for the non-destructive per-language localization of Variables/Model spec.

variables_json is FUNCTIONAL (machine_name, dtype, model_spec, data_template drive
uploads + training). Switching the FR/EN toggle must translate only the DISPLAY text
and never touch the functional fields. These tests mock the LLM translator (no network)
and lock in: display strings are translated positionally, machine names / dtypes /
provenance / structure are byte-identical, the model_spec mirror gets the translated
names + rationale, and the language-detection / no-op guards behave.
"""
import pytest

pytest.importorskip("pandas")   # main is import-only; keep parity with other suites

import main


def _sample_variables():
    return {
        "primary_outcome": {
            "name": "ARI surge incidence", "definition": "Weekly ARI rate",
            "measurement": "cases per 100k", "timeframe": "7 days",
            "machine_name": "ari_incidence", "task_type": "regression",
            "unit": "/100k", "positive_class": None, "provenance": [1, 2],
        },
        "predictor_variables": [
            {"name": "Recent ARI incidence", "definition": "Historical weekly rate",
             "data_source": "surveillance", "machine_name": "ari_incidence_lagged",
             "dtype": "float", "importance": "high", "provenance": [1]},
            {"name": "Meteorological variables", "definition": "Temperature, humidity",
             "data_source": "open-meteo", "machine_name": "meteo_vars",
             "dtype": "float", "importance": "high", "provenance": [3]},
        ],
        "recommended_algorithm": {
            "primary": "Seasonal ARIMA with exogenous variables (SARIMAX)",
            "alternatives": ["DeepAR", "XGBoost with lagged features"],
            "rationale": "SARIMAX captures seasonality and exogenous effects",
            "family": "sarimax", "metric": "rmse", "provenance": [2],
        },
        "alert_thresholds": {
            "green": {"label": "Normal", "range": "< 15 /100k", "rationale": "baseline levels"},
            "orange": {"label": "Increased activity", "range": "15-40 /100k", "rationale": "rising"},
            "red": {"label": "Surge/Alert", "range": "> 40 /100k", "rationale": "outbreak"},
        },
        "required_databases": ["Syndromic surveillance", "Lab networks"],
        "implementation_notes": "Update weekly",
        "model_spec": {
            "outcome": {"name": "ARI surge incidence", "machine_name": "ari_incidence", "task_type": "regression"},
            "features": [
                {"name": "Recent ARI incidence", "machine_name": "ari_incidence_lagged", "dtype": "float"},
                {"name": "Meteorological variables", "machine_name": "meteo_vars", "dtype": "float"},
            ],
            "algorithm": {"family": "sarimax", "rationale": "SARIMAX captures seasonality and exogenous effects"},
            "data_template": {"target_column": "ari_incidence",
                              "columns": [{"name": "ari_incidence", "role": "outcome"},
                                          {"name": "ari_incidence_lagged", "role": "feature"}]},
        },
    }


@pytest.fixture()
def fake_translate(monkeypatch):
    # Deterministic stand-in for the LLM: prefix each string, preserving order/length.
    def _fake(texts, target_lang):
        return [f"[{target_lang}] {t}" for t in texts]
    monkeypatch.setattr(main, "_llm_translate_strings", _fake)


def test_translation_translates_display_and_preserves_functional(fake_translate):
    out = main._translate_variables_payload(_sample_variables(), "fr")

    # display fields translated
    assert out["primary_outcome"]["name"] == "[fr] ARI surge incidence"
    assert out["primary_outcome"]["definition"].startswith("[fr] ")
    assert out["predictor_variables"][0]["name"] == "[fr] Recent ARI incidence"
    assert out["recommended_algorithm"]["rationale"].startswith("[fr] ")
    assert out["recommended_algorithm"]["alternatives"] == ["[fr] DeepAR", "[fr] XGBoost with lagged features"]
    assert out["alert_thresholds"]["orange"]["label"] == "[fr] Increased activity"
    assert out["required_databases"][0] == "[fr] Syndromic surveillance"

    # FUNCTIONAL fields untouched
    assert out["primary_outcome"]["machine_name"] == "ari_incidence"
    assert out["primary_outcome"]["task_type"] == "regression"
    assert out["primary_outcome"]["unit"] == "/100k"
    assert out["primary_outcome"]["provenance"] == [1, 2]
    assert out["predictor_variables"][0]["machine_name"] == "ari_incidence_lagged"
    assert out["predictor_variables"][0]["dtype"] == "float"
    assert out["recommended_algorithm"]["family"] == "sarimax"
    assert out["recommended_algorithm"]["metric"] == "rmse"
    assert out["alert_thresholds"]["green"]["range"] == "< 15 /100k"   # numeric range NOT translated


def test_model_spec_mirror_updated(fake_translate):
    out = main._translate_variables_payload(_sample_variables(), "en")
    ms = out["model_spec"]
    # names + rationale mirror the translated values, matched by machine_name
    assert ms["outcome"]["name"] == "[en] ARI surge incidence"
    assert ms["features"][0]["name"] == "[en] Recent ARI incidence"
    assert ms["features"][1]["name"] == "[en] Meteorological variables"
    assert ms["algorithm"]["rationale"].startswith("[en] ")
    # data_template + machine names still invariant (upload contract preserved)
    assert ms["data_template"]["target_column"] == "ari_incidence"
    assert [c["name"] for c in ms["data_template"]["columns"]] == ["ari_incidence", "ari_incidence_lagged"]
    assert ms["features"][0]["machine_name"] == "ari_incidence_lagged"


def test_positional_integrity_no_crosswiring(fake_translate):
    # Each translated string must land back on its OWN field (the setter list indexing
    # is the risky part). The prefix scheme lets us assert the original text is intact.
    out = main._translate_variables_payload(_sample_variables(), "fr")
    assert out["primary_outcome"]["measurement"] == "[fr] cases per 100k"
    assert out["predictor_variables"][1]["data_source"] == "[fr] open-meteo"
    assert out["alert_thresholds"]["red"]["rationale"] == "[fr] outbreak"


def test_translator_length_mismatch_is_safe(monkeypatch):
    # If the LLM returns the wrong number of items, we must NOT corrupt — return original.
    monkeypatch.setattr(main, "_llm_translate_strings", lambda texts, lang: ["oops"])
    src = _sample_variables()
    out = main._translate_variables_payload(src, "fr")
    assert out["primary_outcome"]["name"] == "ARI surge incidence"   # unchanged


def test_norm_lang_and_detect():
    assert main._norm_lang("en-US") == "en"
    assert main._norm_lang("fr") == "fr"
    assert main._norm_lang(None) is None
    assert main._norm_lang("") is None
    # detection heuristic on the sample (English content)
    assert main._detect_variables_lang(_sample_variables()) == "en"
    fr_vars = {"primary_outcome": {"name": "Incidence des infections",
                                   "definition": "Le taux hebdomadaire des cas pour la population",
                                   "measurement": "cas pour 100k"}}
    assert main._detect_variables_lang(fr_vars) == "fr"
