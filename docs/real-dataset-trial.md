# Real-dataset trial — environment → influenza

This documents the first end-to-end training run on a **real public dataset**, on the
influenza theme, using the product's own stack: the real data connectors
(`data_connectors.fetch_series`) feed the real trainer (`model_trainer.train_model`).
Runner: [`scripts/trial_weather_influenza_ili.py`](../scripts/trial_weather_influenza_ili.py).

## Why influenza + weather

Influenza has strong, well-characterised environmental drivers (cold, dry air) and rich
open surveillance data, so the extracted predictor variables (temperature, humidity) and
the outcome (influenza-like illness) both have real, machine-readable datasets — exactly
the criterion for a first real trial.

The predictor connector already shipped (`open-meteo-weather`, live in production). The
missing piece was the **clinical ILI outcome**: this trial adds the
**`foph-sentinella-ili`** connector (FOPH Sentinella sentinel-GP network, weekly
influenza-like-illness consultation incidence, opendata.swiss), which the app previously
flagged as *"not yet wired → manual upload"*. Weekly ISO-week rows are converted to the
week's Monday date so the series join-aligns with the daily weather connector after weekly
resampling — the same assembly the app performs in `_assemble_connector_frames`.

## Two modes (identical assemble → train code path)

| mode | features | outcome | egress |
|------|----------|---------|--------|
| `weather-ili` | `open-meteo-weather` (temperature, humidity, precip, wind) | `foph-sentinella-ili` (clinical ILI incidence) | open-meteo + opendata.swiss — **production** |
| `ch-influenza` *(default)* | co-circulating RSV / influenza-B / SARS-CoV-2 loads + seasonal (climatic) harmonics | influenza-A wastewater load (a validated influenza-activity signal) | GitHub only — **reproducible anywhere** |

`weather-ili` is the headline topic but needs egress that only production has
(open-meteo and opendata.swiss are blocked from CI/sandboxes). The default `ch-influenza`
mode trains on the real EAWAG Swiss respiratory-virus wastewater dataset
(`raw.githubusercontent.com`), so the proof reproduces without special network access.

## Result (committed, `ch-influenza`, real EAWAG data, plant STEP Aire / Geneva)

```
67 weekly rows (2022-11 → 2024-02), 5 features → target influenza-A load
random_forest (Optuna-tuned: n_estimators=387, max_depth=13, min_samples_split=10)
held-out test R² = 0.51    RMSE = 8.3e6    MAE = 5.1e6

feature importance (by variable)
  season_cos      0.88     ← influenza is strongly seasonal
  flu_b_load      0.06     ← co-circulating influenza-B
  sars_cov2_load  0.02
  season_sin      0.02
  rsv_load        0.02
```

The trained model attributes influenza-A activity primarily to the **seasonal cycle**,
with a secondary **co-circulation** signal from influenza-B — epidemiologically sensible,
learned end-to-end from real surveillance data. (Random k-fold CV is unreliable on a
67-week seasonal series — a fold can hold out an entire season — so the held-out test R²
is the honest generalisation estimate.) The full dataset and results are committed under
[`scripts/trial_output/`](../scripts/trial_output/) for reproducibility.

## Run it

```bash
pip install pandas scikit-learn optuna          # the app's training deps
python scripts/trial_weather_influenza_ili.py                     # ch-influenza (default)
python scripts/trial_weather_influenza_ili.py --mode weather-ili  # needs production egress
```

In the app itself, the same path is reachable without code: create an influenza scenario,
let it derive weather predictors + an ILI outcome, and the connector picker now auto-binds
weather → `open-meteo-weather` and ILI → `foph-sentinella-ili`, fetches, and trains.
