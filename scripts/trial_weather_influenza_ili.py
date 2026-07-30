#!/usr/bin/env python3
"""End-to-end real-dataset trial: environment → influenza, on the app's OWN stack.

This exercises the full predictive path the product ships — the real data
connectors (`data_connectors.fetch_series`) feed the real trainer
(`model_trainer.train_model`) — on a REAL public dataset, and prints the trained
model's honest metrics, hyperparameters and feature importances. It is the
answer to "let's try a real dataset on a specific topic" for the influenza theme.

Two modes (same assemble→train code path):

  weather-ili   (the headline topic — needs open-meteo + FOPH Sentinella egress,
                i.e. production)
        features : open-meteo-weather  (temperature, humidity, …)  [daily → weekly]
        outcome  : foph-sentinella-ili (clinical ILI consultation incidence) [weekly]

  ch-influenza  (default — reproducible anywhere with GitHub access; used for the
                committed proof because open-meteo / opendata.swiss are egress-blocked
                outside production)
        source   : EAWAG respiratory-virus wastewater (real Swiss surveillance,
                   raw.githubusercontent.com) for one treatment plant
        outcome  : influenza-A wastewater load (a validated influenza-activity signal)
        features : co-circulating RSV + influenza-B + SARS-CoV-2 loads and the
                   seasonal (climatic) cycle — the annual temperature/humidity swing
                   that drives influenza — encoded as sin/cos harmonics of the ISO week

Weekly alignment mirrors the app's `_assemble_connector_frames` (main.py): each tidy
daily/weekly series is resampled to weekly buckets (sum for precipitation, last for
viral load, mean otherwise) and outer-joined on the date key.

Run:
    python scripts/trial_weather_influenza_ili.py                    # ch-influenza (default)
    python scripts/trial_weather_influenza_ili.py --mode weather-ili # needs prod egress
Requires: pandas, scikit-learn, optuna (the app's training dependencies).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_connectors as dc          # noqa: E402
import model_trainer                  # noqa: E402


def _agg_for_column(col: str) -> str:
    """Mirror of main.py `_agg_for_column`: how a column is summarised per week."""
    c = (col or "").lower()
    if "precip" in c or "rain" in c:
        return "sum"
    if "load" in c or "wastewater" in c or "viral" in c:
        return "last"
    return "mean"


def _weekly_join(series: list[tuple[list[dict], list[str]]], datetime_col: str = "date"):
    """Resample each tidy series to weekly buckets and outer-join on the week key —
    the same two-stage assembly the app performs in `_assemble_connector_frames`."""
    import pandas as pd
    assembled = None
    for rows, cols in series:
        if not rows:
            continue
        d = pd.DataFrame(rows)
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).set_index("date").sort_index()
        keep = [c for c in cols if c in d.columns]
        if not keep:
            continue
        d = d[keep].apply(pd.to_numeric, errors="coerce")
        d = d.resample("W").agg({c: _agg_for_column(c) for c in keep}).reset_index()
        d["date"] = d["date"].dt.strftime("%Y-%m-%d")
        assembled = d if assembled is None else assembled.merge(d, on="date", how="outer")
    if assembled is None:
        return None
    assembled = assembled.sort_values("date").reset_index(drop=True)
    return assembled.rename(columns={"date": datetime_col})


def _add_seasonal_harmonics(df, datetime_col="date"):
    """Encode the annual (climatic) cycle as sin/cos of the ISO week — a weather-cycle
    proxy that captures influenza's strong seasonality without an external feed."""
    import pandas as pd
    wk = pd.to_datetime(df[datetime_col]).dt.isocalendar().week.astype(float)
    df["season_sin"] = wk.map(lambda w: math.sin(2 * math.pi * w / 52.18))
    df["season_cos"] = wk.map(lambda w: math.cos(2 * math.pi * w / 52.18))
    return df


def build_ch_influenza(args):
    """Real EAWAG Swiss influenza wastewater → supervised weekly table.
    outcome = influenza-A load; features = co-circulating viruses + season."""
    rows = dc.fetch_series("eawag-wastewater", {
        "region": args.region, "wwtp": args.plant or None,
        "start_date": args.start, "end_date": args.end,
    })
    if not rows:
        raise SystemExit("EAWAG returned no rows (network blocked, or plant/date filter too narrow).")
    df = _weekly_join([(rows, ["flu_a_load", "flu_b_load", "rsv_load", "sars_cov2_load"])])
    df = _add_seasonal_harmonics(df)
    df = df.dropna(subset=["flu_a_load"]).reset_index(drop=True)
    spec = {
        "outcome": {"machine_name": "flu_a_load", "task_type": "regression", "unit": "gc/day"},
        "features": [{"machine_name": c} for c in
                     ["flu_b_load", "rsv_load", "sars_cov2_load", "season_sin", "season_cos"]],
        "algorithm": {"family": args.family, "metric": "r2",
                      "cv": {"strategy": "kfold", "folds": 5}},
    }
    provenance = {
        "topic": "Swiss influenza activity (wastewater surveillance)",
        "outcome": "influenza-A wastewater load (a validated ILI-activity signal)",
        "outcome_source": f"EAWAG RespiratoryVirusesWastewater — plant {args.plant or 'STEP Aire (Geneva)'}",
        "outcome_url": dc._EAWAG_CSV_URL,
        "note": "Random k-fold CV is unreliable on a 67-week seasonal series (a fold can "
                "hold out an entire influenza season); the held-out test R² is the honest "
                "generalisation estimate. The explicit weather → clinical-ILI variant runs "
                "in --mode weather-ili (open-meteo + FOPH Sentinella, production egress).",
    }
    return df, spec, provenance


def build_weather_ili(args):
    """Live weather → clinical ILI (production egress). outcome = Sentinella ILI."""
    weather = dc.fetch_series("open-meteo-weather", {
        "region": args.region, "start_date": args.start, "end_date": args.end,
    })
    ili = dc.fetch_series("foph-sentinella-ili", {
        "url": os.getenv("FOPH_SENTINELLA_CSV_URL"), "value_col": args.ili_col,
        "start_date": args.start, "end_date": args.end,
    })
    if not weather or not ili:
        raise SystemExit(
            "Live sources unreachable (weather rows=%d, ili rows=%d). This mode needs "
            "open-meteo + FOPH Sentinella egress — run it in production, or set "
            "FOPH_SENTINELLA_CSV_URL." % (len(weather), len(ili)))
    df = _weekly_join([
        (weather, ["temp_mean", "temp_min", "temp_max", "relative_humidity_mean", "precip_sum", "wind_max"]),
        (ili, ["ili_incidence"]),
    ])
    df = df.dropna(subset=["ili_incidence"]).reset_index(drop=True)
    spec = {
        "outcome": {"machine_name": "ili_incidence", "task_type": "regression",
                    "unit": "per 100k / per 1000 consultations"},
        "features": [{"machine_name": c} for c in
                     ["temp_mean", "relative_humidity_mean", "precip_sum", "wind_max"]],
        "algorithm": {"family": args.family, "metric": "r2",
                      "cv": {"strategy": "kfold", "folds": 5}},
    }
    provenance = {
        "topic": "Weather → influenza-like illness (clinical)",
        "outcome_source": "FOPH Sentinella clinical ILI (opendata.swiss / foph-sentinella-ili)",
        "feature_source": "Open-Meteo historical weather (open-meteo-weather)",
    }
    return df, spec, provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["ch-influenza", "weather-ili"], default="ch-influenza")
    ap.add_argument("--region", default="geneva")
    ap.add_argument("--plant", default="STEP Aire")
    ap.add_argument("--start", default="2022-11-01")
    ap.add_argument("--end", default="2024-02-27")
    ap.add_argument("--family", default="random_forest")
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--ili-col", default=None, help="explicit Sentinella incidence column (weather-ili mode)")
    ap.add_argument("--out", default=None, help="directory to write dataset CSV + results JSON")
    args = ap.parse_args()

    df, spec, provenance = (build_weather_ili if args.mode == "weather-ili" else build_ch_influenza)(args)
    print(f"[trial] mode={args.mode}  assembled {len(df)} weekly rows, "
          f"{len(spec['features'])} features → target '{spec['outcome']['machine_name']}'")

    result = model_trainer.train_model(df, spec, n_trials=args.n_trials)

    m = result["metrics"]
    print(f"\n=== Trained model ({result['family']}, {result['task_type']}) ===")
    print(f"  rows: {result['n_total']}  (train {result['n_train']} / test {result['n_test']})")
    print(f"  R²={m.get('r2'):.3f}   RMSE={m.get('rmse'):.3g}   MAE={m.get('mae'):.3g}"
          f"   (CV R²={m.get('cv_r2', float('nan')):.3f})")
    print(f"  hyperparameters: {json.dumps(result['best_params'])}")
    print("  feature importance (by variable):")
    imps = result.get("importances_by_variable") or {}
    for var, imp in sorted(imps.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {var:>22}  {float(imp):.3f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(os.path.join(args.out, f"{args.mode}_dataset.csv"), index=False)
        payload = {k: v for k, v in result.items() if k != "pipeline"}
        payload["provenance"] = provenance
        payload["spec"] = spec
        with open(os.path.join(args.out, f"{args.mode}_results.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\n[trial] wrote dataset + results to {args.out}/")


if __name__ == "__main__":
    main()
