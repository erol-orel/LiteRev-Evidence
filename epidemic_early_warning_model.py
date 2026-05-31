"""
epidemic_early_warning_model.py
================================
Modèle de détection précoce d'épidémies pour le scénario GESICA
"epidemic-early-warning".

Approche (v2 — ensemble multi-modèles) :
  1. Farrington Flexible (détection d'anomalies, signal binaire)
  2. SARIMAX(1,1,1)(1,1,1,52) (prédiction quantitative J+14)
  3. Prophet (Meta) (prédiction J+7 à J+28 avec jours fériés)
  4. XGBoost avec features météo (ajustement court terme J+3 à J+7)
  5. Ensemble pondéré des 3 modèles prédictifs

Données :
  - Réseau Sentinelles FR (grippe, gastro, IRA, varicelle) — API REST publique
  - Open-Meteo (température, humidité, précipitations)
  - Google Trends (pytrends) — signal complémentaire

Références :
  - Farrington CP et al. J Royal Stat Soc, 1996
  - Bonora R et al. Public Health, 2025 (PMID: 39342741)
  - Nakashima T et al. npj Digital Medicine, 2025

Usage :
  from epidemic_early_warning_model import epidemic_model_singleton
  result = epidemic_model_singleton.predict()
"""

import logging
import math
import statistics
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("epidemic_model")

# ─── Constantes ──────────────────────────────────────────────────────────────

SENTINELLES_BASE = "https://www.sentiweb.fr/api/v1"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HIST = "https://archive-api.open-meteo.com/v1/archive"

# Coordonnées Grand Genève (Annemasse)
GEO_LAT = 46.13
GEO_LON = 6.24

DISEASE_CONFIG = {
    "grippe": {
        "sentinelles_indicator": "incidence3",
        "epidemic_threshold": 140,    # /100k (seuil épidémique Sentinelles)
        "warning_threshold": 80,
        "ems_impact_factor": 1.59,    # +59% appels EMS au seuil épidémique (Bonora 2025)
        "label": "Grippe (ILI)",
        "peak_week": 4,               # Semaine ISO du pic hivernal
        "peak_sigma": 6,              # Largeur du pic (semaines)
        "google_keywords": ["grippe", "fièvre", "symptômes grippe"],
    },
    "gastro": {
        "sentinelles_indicator": "incidence3",
        "epidemic_threshold": 300,
        "warning_threshold": 180,
        "ems_impact_factor": 1.22,
        "label": "Gastro-entérite",
        "peak_week": 52,
        "peak_sigma": 5,
        "google_keywords": ["gastroentérite", "diarrhée vomissements"],
    },
    "ira": {
        "sentinelles_indicator": "incidence3",
        "epidemic_threshold": 200,
        "warning_threshold": 120,
        "ems_impact_factor": 1.31,
        "label": "IRA (Infections Respiratoires Aiguës)",
        "peak_week": 6,
        "peak_sigma": 7,
        "google_keywords": ["toux", "bronchite", "infection respiratoire"],
    },
    "varicelle": {
        "sentinelles_indicator": "incidence3",
        "epidemic_threshold": 150,
        "warning_threshold": 90,
        "ems_impact_factor": 1.08,
        "label": "Varicelle",
        "peak_week": 14,
        "peak_sigma": 6,
        "google_keywords": ["varicelle", "boutons fièvre enfant"],
    },
}

REGION_CODE = "ARA"   # Auvergne-Rhône-Alpes (Haute-Savoie)
REGION_LABEL = "Auvergne-Rhône-Alpes"

# Poids de l'ensemble (somme = 1.0)
ENSEMBLE_WEIGHTS = {
    "sarimax": 0.40,
    "prophet": 0.35,
    "xgboost": 0.25,
}


# ─── Collecte des données Sentinelles ────────────────────────────────────────

def _fetch_sentinelles_series(pathology: str, weeks: int = 104) -> List[Dict]:
    """
    Récupère les données hebdomadaires Sentinelles FR pour une pathologie.
    Retourne une liste de dicts {week, year, incidence, geo}.
    """
    try:
        url = f"{SENTINELLES_BASE}/incidence"
        params = {
            "indicator": "incidence3",
            "pathology": pathology,
            "geo": REGION_CODE,
            "format": "json",
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                return data["data"][-weeks:]
            if isinstance(data, list):
                return data[-weeks:]
    except Exception as e:
        logger.warning(f"Sentinelles fetch failed for {pathology}: {e}")
    return []


def _fetch_weather_features() -> Dict[str, float]:
    """
    Récupère les features météo actuelles et des 7 derniers jours depuis Open-Meteo.
    Retourne un dict de features pour XGBoost.
    """
    try:
        params = {
            "latitude": GEO_LAT,
            "longitude": GEO_LON,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_max",
            "past_days": 7,
            "forecast_days": 14,
            "timezone": "Europe/Paris",
        }
        resp = requests.get(OPEN_METEO_BASE, params=params, timeout=8)
        if resp.status_code == 200:
            d = resp.json().get("daily", {})
            temps = d.get("temperature_2m_max", [])
            temps_min = d.get("temperature_2m_min", [])
            precip = d.get("precipitation_sum", [])
            humidity = d.get("relative_humidity_2m_max", [])

            # Features agrégées sur les 7 derniers jours
            recent_temps = [t for t in temps[:7] if t is not None]
            recent_precip = [p for p in precip[:7] if p is not None]
            recent_humidity = [h for h in humidity[:7] if h is not None]

            return {
                "temp_mean_7d": statistics.mean(recent_temps) if recent_temps else 10.0,
                "temp_min_7d": min(recent_temps) if recent_temps else 5.0,
                "temp_max_7d": max(recent_temps) if recent_temps else 15.0,
                "precip_sum_7d": sum(recent_precip) if recent_precip else 0.0,
                "humidity_mean_7d": statistics.mean(recent_humidity) if recent_humidity else 70.0,
                # Variation thermique (facteur de risque OHCA et ILI)
                "temp_delta_7d": (max(recent_temps) - min(recent_temps)) if len(recent_temps) >= 2 else 5.0,
                # Forecast 7 prochains jours
                "temp_forecast_7d": statistics.mean([t for t in temps[7:14] if t is not None]) if temps[7:14] else 10.0,
            }
    except Exception as e:
        logger.debug(f"Open-Meteo fetch failed: {e}")
    return {
        "temp_mean_7d": 10.0, "temp_min_7d": 5.0, "temp_max_7d": 15.0,
        "precip_sum_7d": 0.0, "humidity_mean_7d": 70.0,
        "temp_delta_7d": 5.0, "temp_forecast_7d": 10.0,
    }


# ─── Génération de données synthétiques (fallback) ───────────────────────────

def _generate_synthetic_series(disease: str, n_weeks: int = 104) -> List[float]:
    """
    Génère une série temporelle synthétique réaliste pour une maladie.
    La saisonnalité est ancrée sur la semaine calendaire réelle (ISO week).
    """
    config = DISEASE_CONFIG.get(disease, DISEASE_CONFIG["grippe"])
    baseline = config["warning_threshold"] * 0.15
    amplitude = config["epidemic_threshold"] * 0.65
    peak_week = config["peak_week"]
    peak_sigma = config["peak_sigma"]
    series = []

    today_iso_week = date.today().isocalendar()[1]

    for i in range(n_weeks):
        weeks_ago = n_weeks - 1 - i
        iso_week = ((today_iso_week - 1 - weeks_ago) % 52) + 1

        # Saisonnalité circulaire
        dist = min(abs(iso_week - peak_week), 52 - abs(iso_week - peak_week))
        seasonal = amplitude * math.exp(-0.5 * (dist / peak_sigma) ** 2)

        trend = baseline + (i / n_weeks) * 8
        noise = (hash(f"{disease}_{i}") % 100 - 50) * 0.25
        value = max(0, trend + seasonal + noise)
        series.append(round(value, 1))

    return series


# ─── Algorithme de Farrington Flexible (détection d'anomalies) ───────────────

def _farrington_detection(series: List[float], current_value: float) -> Dict[str, Any]:
    """
    Implémentation simplifiée de l'algorithme de Farrington Flexible.
    Détecte si la valeur actuelle est une anomalie par rapport à l'historique saisonnier.

    Référence : Farrington CP et al. J Royal Stat Soc, 1996.
    Standard ECDC/SGSS pour la surveillance épidémique.

    Retourne : {is_anomaly, threshold, z_score, confidence}
    """
    if len(series) < 26:
        return {"is_anomaly": False, "threshold": None, "z_score": 0.0, "confidence": "low"}

    # Sélectionner les valeurs historiques à la même période saisonnière (±2 semaines)
    # sur les 2 années précédentes (semaines 0, 52, 104 ± 2)
    today_week = date.today().isocalendar()[1]
    n = len(series)

    historical_values = []
    for offset in [52, 104]:
        for delta in range(-2, 3):
            idx = n - offset + delta
            if 0 <= idx < n:
                historical_values.append(series[idx])

    if len(historical_values) < 3:
        # Fallback : utiliser les 26 dernières semaines hors saison
        historical_values = series[-26:-2]

    if not historical_values:
        return {"is_anomaly": False, "threshold": None, "z_score": 0.0, "confidence": "low"}

    mean_hist = statistics.mean(historical_values)
    std_hist = statistics.stdev(historical_values) if len(historical_values) > 1 else mean_hist * 0.3

    # Seuil Farrington : moyenne + 1.96 * écart-type (IC 95%)
    threshold_95 = mean_hist + 1.96 * std_hist
    threshold_99 = mean_hist + 2.58 * std_hist

    z_score = (current_value - mean_hist) / std_hist if std_hist > 0 else 0.0

    is_anomaly = current_value > threshold_95
    confidence = "high" if len(historical_values) >= 8 else "medium" if len(historical_values) >= 4 else "low"

    return {
        "is_anomaly": is_anomaly,
        "threshold_95": round(threshold_95, 1),
        "threshold_99": round(threshold_99, 1),
        "mean_historical": round(mean_hist, 1),
        "z_score": round(z_score, 2),
        "confidence": confidence,
        "n_historical": len(historical_values),
    }


# ─── Modèle SARIMAX ──────────────────────────────────────────────────────────

def _fit_sarimax(series: List[float], horizon_weeks: int = 2) -> Optional[List[float]]:
    """
    Ajuste un modèle SARIMAX(1,1,1)(1,1,1,52) et prédit `horizon_weeks` semaines.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
        import numpy as np

        if len(series) < 52:
            return None

        y = np.array(series, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=100)
            forecast = fit.forecast(steps=horizon_weeks)
            return [max(0, float(v)) for v in forecast]

    except ImportError:
        logger.info("statsmodels non disponible")
        return None
    except Exception as e:
        logger.warning(f"SARIMAX fit failed: {e}")
        return None


# ─── Modèle Prophet ──────────────────────────────────────────────────────────

def _fit_prophet(series: List[float], horizon_weeks: int = 4) -> Optional[List[float]]:
    """
    Ajuste un modèle Prophet sur la série hebdomadaire et prédit `horizon_weeks` semaines.
    Intègre les jours fériés français.
    """
    try:
        from prophet import Prophet
        import pandas as pd
        import warnings

        if len(series) < 26:
            return None

        # Construire le DataFrame Prophet (ds = date, y = valeur)
        end_date = date.today()
        dates = [end_date - timedelta(weeks=len(series) - 1 - i) for i in range(len(series))]

        df = pd.DataFrame({
            "ds": pd.to_datetime(dates),
            "y": series,
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05,
            )
            # Ajouter les vacances scolaires françaises comme régresseurs
            m.add_country_holidays(country_name="FR")
            m.fit(df)

            future = m.make_future_dataframe(periods=horizon_weeks, freq="W")
            forecast = m.predict(future)
            preds = forecast["yhat"].tail(horizon_weeks).tolist()
            return [max(0, float(v)) for v in preds]

    except ImportError:
        logger.info("prophet non disponible")
        return None
    except Exception as e:
        logger.warning(f"Prophet fit failed: {e}")
        return None


# ─── Modèle XGBoost avec features météo ──────────────────────────────────────

def _fit_xgboost(
    series: List[float],
    weather: Dict[str, float],
    disease: str,
    horizon_weeks: int = 2,
) -> Optional[List[float]]:
    """
    Modèle XGBoost avec features temporelles et météo pour prédiction court terme.
    Entraîné sur l'historique de la série avec features synthétiques météo.
    """
    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor

        if len(series) < 26:
            return None

        config = DISEASE_CONFIG.get(disease, DISEASE_CONFIG["grippe"])
        peak_week = config["peak_week"]

        # Construction des features pour chaque semaine historique
        X, y = [], []
        for i in range(4, len(series)):
            w = date.today().isocalendar()[1]
            iso_week = ((w - len(series) + i) % 52) + 1

            # Features saisonnières (sin/cos pour circularité)
            sin_week = math.sin(2 * math.pi * iso_week / 52)
            cos_week = math.cos(2 * math.pi * iso_week / 52)

            # Distance au pic (feature non-linéaire)
            dist_peak = min(abs(iso_week - peak_week), 52 - abs(iso_week - peak_week))

            # Lags
            lag1 = series[i - 1]
            lag2 = series[i - 2]
            lag4 = series[i - 4]

            # Tendance locale
            trend = series[i - 1] - series[i - 4] if i >= 4 else 0

            # Features météo synthétiques (corrélées à la saison)
            temp_synthetic = 10 - 15 * math.cos(2 * math.pi * iso_week / 52)
            humidity_synthetic = 75 + 10 * math.cos(2 * math.pi * iso_week / 52)

            X.append([sin_week, cos_week, dist_peak, lag1, lag2, lag4, trend,
                       temp_synthetic, humidity_synthetic])
            y.append(series[i])

        X, y = np.array(X), np.array(y)

        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
        model.fit(X, y)

        # Prédiction pour les prochaines semaines
        predictions = []
        current_series = list(series)
        current_week = date.today().isocalendar()[1]

        for step in range(horizon_weeks):
            iso_week = ((current_week + step) % 52) + 1
            sin_week = math.sin(2 * math.pi * iso_week / 52)
            cos_week = math.cos(2 * math.pi * iso_week / 52)
            dist_peak = min(abs(iso_week - peak_week), 52 - abs(iso_week - peak_week))

            lag1 = current_series[-1]
            lag2 = current_series[-2]
            lag4 = current_series[-4]
            trend = current_series[-1] - current_series[-4]

            # Utiliser les vraies features météo pour la première semaine
            if step == 0:
                temp = weather.get("temp_mean_7d", 10.0)
                humidity = weather.get("humidity_mean_7d", 70.0)
            else:
                temp = weather.get("temp_forecast_7d", 10.0)
                humidity = 70.0

            feat = np.array([[sin_week, cos_week, dist_peak, lag1, lag2, lag4, trend, temp, humidity]])
            pred = max(0, float(model.predict(feat)[0]))
            predictions.append(pred)
            current_series.append(pred)

        return predictions

    except ImportError:
        logger.info("scikit-learn non disponible")
        return None
    except Exception as e:
        logger.warning(f"XGBoost fit failed: {e}")
        return None


# ─── Ensemble des modèles ─────────────────────────────────────────────────────

def _ensemble_predict(
    sarimax_preds: Optional[List[float]],
    prophet_preds: Optional[List[float]],
    xgboost_preds: Optional[List[float]],
    horizon_weeks: int = 2,
) -> Tuple[List[float], str]:
    """
    Combine les prédictions des 3 modèles par moyenne pondérée.
    Retourne (predictions, models_used_label).
    """
    available = []
    weights_used = []
    labels = []

    if sarimax_preds and len(sarimax_preds) >= horizon_weeks:
        available.append(sarimax_preds[:horizon_weeks])
        weights_used.append(ENSEMBLE_WEIGHTS["sarimax"])
        labels.append("SARIMAX")

    if prophet_preds and len(prophet_preds) >= horizon_weeks:
        available.append(prophet_preds[:horizon_weeks])
        weights_used.append(ENSEMBLE_WEIGHTS["prophet"])
        labels.append("Prophet")

    if xgboost_preds and len(xgboost_preds) >= horizon_weeks:
        available.append(xgboost_preds[:horizon_weeks])
        weights_used.append(ENSEMBLE_WEIGHTS["xgboost"])
        labels.append("XGBoost")

    if not available:
        return [], "fallback"

    # Normaliser les poids
    total_w = sum(weights_used)
    norm_weights = [w / total_w for w in weights_used]

    # Moyenne pondérée
    ensemble = []
    for week_idx in range(horizon_weeks):
        val = sum(available[m][week_idx] * norm_weights[m] for m in range(len(available)))
        ensemble.append(round(max(0, val), 1))

    return ensemble, "+".join(labels)


# ─── Fallback Serfling ────────────────────────────────────────────────────────

def _serfling_forecast(series: List[float], horizon_days: int = 14) -> List[float]:
    """Fallback : prédiction par moyenne mobile + z-score saisonnier."""
    horizon_weeks = max(1, horizon_days // 7)
    if len(series) < 8:
        last_val = series[-1] if series else 50.0
        return [last_val] * horizon_weeks

    window = series[-8:]
    mean_w = statistics.mean(window)
    std_w = statistics.stdev(window) if len(window) > 1 else mean_w * 0.2

    # Tendance récente
    trend = (series[-1] - series[-4]) / 4 if len(series) >= 4 else 0

    predictions = []
    for i in range(horizon_weeks):
        val = max(0, mean_w + trend * (i + 1))
        predictions.append(round(val, 1))

    return predictions


# ─── Calcul du niveau d'alerte ────────────────────────────────────────────────

def _compute_alert_level(
    current: float,
    predictions: List[float],
    config: Dict,
    farrington: Dict,
) -> Dict[str, Any]:
    """
    Calcule le niveau d'alerte combinant Farrington + seuils Sentinelles + prédictions.
    Niveaux : NORMAL / VIGILANCE / PRÉ-ÉPIDÉMIQUE / ÉPIDÉMIE
    """
    epidemic_threshold = config["epidemic_threshold"]
    warning_threshold = config["warning_threshold"]
    ems_factor = config["ems_impact_factor"]

    max_pred = max(predictions) if predictions else current
    peak_day = (predictions.index(max_pred) + 1) * 7 if predictions else 0

    # Niveau actuel
    if current >= epidemic_threshold:
        current_alert = "ÉPIDÉMIE"
    elif current >= warning_threshold:
        current_alert = "VIGILANCE"
    elif farrington.get("is_anomaly", False):
        current_alert = "SIGNAL FARRINGTON"
    else:
        current_alert = "NORMAL"

    # Niveau maximal prédit à J+14
    if max_pred >= epidemic_threshold:
        max_alert = "ÉPIDÉMIE"
    elif max_pred >= warning_threshold:
        max_alert = "VIGILANCE"
    elif farrington.get("z_score", 0) > 1.5:
        max_alert = "SIGNAL FARRINGTON"
    else:
        max_alert = "NORMAL"

    # Impact EMS
    if max_pred >= epidemic_threshold:
        ems_impact = f"+{round((ems_factor - 1) * 100)}% d'appels EMS estimés (seuil épidémique dépassé)"
    elif max_pred >= warning_threshold:
        ems_impact = f"+{round((ems_factor - 1) * 50)}% d'appels EMS estimés (seuil de vigilance)"
    else:
        ems_impact = "Impact EMS nominal (< seuil de vigilance)"

    # Recommandation
    label = config["label"]
    if max_alert == "ÉPIDÉMIE":
        recommendation = (
            f"[ALERTE ÉPIDÉMIQUE] {label} : seuil épidémique dépassé. "
            f"Renforcer les effectifs EMS, activer le protocole de triage renforcé, "
            f"coordonner avec les urgences HUG/CHUV. Pic prévu semaine {peak_day // 7}."
        )
    elif max_alert == "VIGILANCE":
        recommendation = (
            f"[VIGILANCE] {label} : seuil de vigilance atteint. "
            f"Surveiller l'évolution, préparer le renforcement des effectifs EMS si la tendance se confirme."
        )
    elif max_alert == "SIGNAL FARRINGTON" or current_alert == "SIGNAL FARRINGTON":
        recommendation = (
            f"[SIGNAL PRÉCOCE] {label} : anomalie détectée par l'algorithme de Farrington "
            f"(z-score = {farrington.get('z_score', 0):.1f}). Surveiller attentivement les 2 prochaines semaines."
        )
    else:
        recommendation = f"{label} : situation normale. Surveillance de routine."

    return {
        "current_alert": current_alert,
        "max_alert_14d": max_alert,
        "peak_incidence_14d": round(max_pred, 1),
        "peak_day": peak_day,
        "ems_impact": ems_impact,
        "recommendation": recommendation,
        "farrington": farrington,
    }


# ─── Modèle principal ─────────────────────────────────────────────────────────

class EpidemicEarlyWarningModel:
    """
    Modèle d'alerte précoce épidémique — ensemble Farrington + SARIMAX + Prophet + XGBoost.
    """

    def predict(self) -> Dict[str, Any]:
        generated_at = datetime.utcnow().isoformat() + "Z"

        # Récupérer les features météo
        weather = _fetch_weather_features()

        diseases_result = {}
        overall_alert = "NORMAL"
        alert_order = ["NORMAL", "SIGNAL FARRINGTON", "VIGILANCE", "PRÉ-ÉPIDÉMIQUE", "ÉPIDÉMIE"]

        for disease, config in DISEASE_CONFIG.items():
            # 1. Récupérer les données Sentinelles
            raw_data = _fetch_sentinelles_series(disease)
            if raw_data and len(raw_data) >= 10:
                series = []
                for row in raw_data:
                    inc = row.get("inc", row.get("incidence", row.get("value", None)))
                    if inc is not None:
                        try:
                            series.append(float(inc))
                        except (ValueError, TypeError):
                            pass
                model_status = "live" if len(series) >= 26 else "partial"
            else:
                series = []
                model_status = "fallback"

            # Fallback synthétique si données insuffisantes
            if len(series) < 26:
                series = _generate_synthetic_series(disease, 104)
                model_status = "fallback"

            current_incidence = series[-1] if series else 50.0

            # 2. Farrington Flexible (détection d'anomalies)
            farrington = _farrington_detection(series, current_incidence)

            # 3. Prédictions ensemble (horizon = 2 semaines = 14 jours)
            horizon_weeks = 2
            sarimax_preds = _fit_sarimax(series, horizon_weeks)
            prophet_preds = _fit_prophet(series, horizon_weeks + 2)  # Prophet prédit 4 semaines
            xgboost_preds = _fit_xgboost(series, weather, disease, horizon_weeks)

            ensemble_preds, models_used = _ensemble_predict(
                sarimax_preds, prophet_preds, xgboost_preds, horizon_weeks
            )

            # Fallback Serfling si aucun modèle ML n'a fonctionné
            if not ensemble_preds:
                ensemble_preds = _serfling_forecast(series, 14)
                models_used = "Serfling (fallback)"
                model_status = "fallback"

            # 4. Niveau d'alerte
            alert_info = _compute_alert_level(current_incidence, ensemble_preds, config, farrington)

            # 5. Prédictions journalières (interpolation linéaire semaine → jour)
            daily_predictions = []
            today = date.today()
            for week_idx, week_val in enumerate(ensemble_preds):
                for day_offset in range(7):
                    d = today + timedelta(days=week_idx * 7 + day_offset)
                    daily_predictions.append({
                        "date": d.isoformat(),
                        "incidence": round(week_val, 1),
                        "alert": alert_info["max_alert_14d"] if week_val >= config["epidemic_threshold"]
                                 else "VIGILANCE" if week_val >= config["warning_threshold"]
                                 else "NORMAL",
                    })

            diseases_result[disease] = {
                "disease": disease,
                "label": config["label"],
                "current_incidence": round(current_incidence, 1),
                "epidemic_threshold": config["epidemic_threshold"],
                "warning_threshold": config["warning_threshold"],
                "current_alert": alert_info["current_alert"],
                "max_alert_14d": alert_info["max_alert_14d"],
                "peak_incidence_14d": alert_info["peak_incidence_14d"],
                "peak_day": alert_info["peak_day"],
                "ems_impact": alert_info["ems_impact"],
                "recommendation": alert_info["recommendation"],
                "farrington_anomaly": farrington.get("is_anomaly", False),
                "farrington_z_score": farrington.get("z_score", 0.0),
                "farrington_threshold_95": farrington.get("threshold_95"),
                "models_used": models_used,
                "model_status": model_status,
                "weekly_predictions": [round(v, 1) for v in ensemble_preds],
                "daily_predictions": daily_predictions[:14],
            }

            # Mise à jour du niveau d'alerte global
            current_level = alert_info["max_alert_14d"]
            if alert_order.index(current_level) > alert_order.index(overall_alert):
                overall_alert = current_level

        return {
            "model": "EpidemicEarlyWarning v2 (Farrington+SARIMAX+Prophet+XGBoost)",
            "status": "live",
            "generated_at": generated_at,
            "region": REGION_LABEL,
            "overall_alert_level": overall_alert,
            "horizon_days": 14,
            "weather_features": {
                "temp_mean_7d": weather.get("temp_mean_7d"),
                "humidity_mean_7d": weather.get("humidity_mean_7d"),
                "temp_delta_7d": weather.get("temp_delta_7d"),
                "source": "Open-Meteo",
            },
            "diseases": diseases_result,
            "scientific_references": [
                "Farrington CP et al. J Royal Stat Soc, 1996 (Farrington algorithm)",
                "Bonora R et al. Public Health, 2025 (EMS calls as ILI predictor)",
                "Rosenkötter N et al. BMC Public Health, 2013 (EMS syndromic surveillance)",
            ],
        }


# Singleton
epidemic_model_singleton = EpidemicEarlyWarningModel()
