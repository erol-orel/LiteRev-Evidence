"""
epidemic_early_warning_model.py
================================
Modèle de détection précoce d'épidémies pour le scénario GESICA
"epidemic-early-warning".

Approche :
  - Données réelles : API Sentinelles FR (grippe, gastro, IRA, varicelle)
  - Modèle principal : SARIMAX (statsmodels) — série temporelle saisonnière
  - Fallback analytique : seuil de Serfling + z-score glissant
  - Sortie : prédiction J+14 (incidence /100k), niveau d'alerte (NORMAL / VIGILANCE / ÉPIDÉMIE),
    recommandations opérationnelles EMS

Usage :
  from epidemic_early_warning_model import epidemic_model_singleton
  result = epidemic_model_singleton.predict()
"""

import logging
import math
import statistics
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("epidemic_model")

# ─── Constantes ──────────────────────────────────────────────────────────────

SENTINELLES_BASE = "https://www.sentiweb.fr/api/v1"
ECDC_BASE = "https://opendata.ecdc.europa.eu/covid19"

DISEASE_CONFIG = {
    "grippe": {
        "sentinelles_indicator": "incidence3",  # ILI incidence
        "epidemic_threshold": 140,  # /100k (seuil épidémique Sentinelles)
        "warning_threshold": 80,
        "ems_impact_factor": 1.4,
        "label": "Grippe (ILI)",
    },
    "gastro": {
        "sentinelles_indicator": "incidence3",
        "epidemic_threshold": 300,
        "warning_threshold": 180,
        "ems_impact_factor": 1.2,
        "label": "Gastro-entérite",
    },
    "ira": {
        "sentinelles_indicator": "incidence3",
        "epidemic_threshold": 200,
        "warning_threshold": 120,
        "ems_impact_factor": 1.3,
        "label": "IRA (Infections Respiratoires Aiguës)",
    },
    "varicelle": {
        "sentinelles_indicator": "incidence3",
        "epidemic_threshold": 150,
        "warning_threshold": 90,
        "ems_impact_factor": 1.1,
        "label": "Varicelle",
    },
}

REGION_CODE = "ARA"  # Auvergne-Rhône-Alpes (Haute-Savoie)
REGION_LABEL = "Auvergne-Rhône-Alpes"


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


def _fetch_ecdc_flu() -> Optional[float]:
    """Récupère le taux d'incidence grippe ECDC (Europe) comme signal complémentaire."""
    try:
        url = f"{ECDC_BASE}/nationalcasecounts/data.json"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            # Extraction simplifiée — on cherche France, semaine la plus récente
            records = [r for r in data if r.get("country") == "FR" and r.get("indicator") == "cases"]
            if records:
                latest = sorted(records, key=lambda x: x.get("year_week", ""), reverse=True)[0]
                return float(latest.get("value", 0))
    except Exception as e:
        logger.debug(f"ECDC fetch failed: {e}")
    return None


# ─── Génération de données synthétiques (fallback) ───────────────────────────

def _generate_synthetic_series(disease: str, n_weeks: int = 104) -> List[float]:
    """
    Génère une série temporelle synthétique réaliste pour une maladie.
    Utilise un modèle saisonnier avec bruit gaussien.
    """
    config = DISEASE_CONFIG.get(disease, DISEASE_CONFIG["grippe"])
    baseline = config["warning_threshold"] * 0.4
    amplitude = config["epidemic_threshold"] * 0.6
    series = []

    for i in range(n_weeks):
        # Saisonnalité annuelle (pic hivernal semaine 2-8)
        week_of_year = (i % 52) + 1
        seasonal = amplitude * math.exp(-0.5 * ((week_of_year - 4) / 8) ** 2)
        # Tendance légèrement croissante sur 2 ans
        trend = baseline + (i / n_weeks) * 10
        # Bruit gaussien
        noise = (hash(f"{disease}_{i}") % 100 - 50) * 0.3
        value = max(0, trend + seasonal + noise)
        series.append(round(value, 1))

    return series


# ─── Modèle SARIMAX (statsmodels) ────────────────────────────────────────────

def _fit_sarimax(series: List[float], horizon: int = 14) -> Optional[List[float]]:
    """
    Ajuste un modèle SARIMAX(1,1,1)(1,1,1,52) sur la série et prédit `horizon` jours.
    Retourne les prédictions ou None si statsmodels non disponible.
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
            forecast = fit.forecast(steps=horizon)
            return [max(0, float(v)) for v in forecast]

    except ImportError:
        logger.info("statsmodels non disponible, utilisation du fallback analytique")
        return None
    except Exception as e:
        logger.warning(f"SARIMAX fit failed: {e}")
        return None


def _serfling_forecast(series: List[float], horizon_days: int = 14) -> List[float]:
    """
    Fallback analytique : prédiction par moyenne mobile + z-score saisonnier (méthode Serfling).
    Horizon en jours → converti en semaines (horizon_days / 7).
    """
    horizon_weeks = max(1, horizon_days // 7)
    if len(series) < 8:
        last_val = series[-1] if series else 50.0
        return [last_val] * horizon_weeks

    # Moyenne mobile sur les 8 dernières semaines
    recent = series[-8:]
    ma = statistics.mean(recent)
    # Tendance linéaire simple
    if len(recent) >= 4:
        slope = (recent[-1] - recent[-4]) / 4
    else:
        slope = 0.0

    predictions = []
    for i in range(1, horizon_weeks + 1):
        pred = max(0, ma + slope * i)
        predictions.append(round(pred, 1))

    return predictions


# ─── Calcul du niveau d'alerte ───────────────────────────────────────────────

def _compute_alert_level(incidence: float, disease: str) -> str:
    config = DISEASE_CONFIG.get(disease, DISEASE_CONFIG["grippe"])
    if incidence >= config["epidemic_threshold"]:
        return "ÉPIDÉMIE"
    elif incidence >= config["warning_threshold"]:
        return "VIGILANCE"
    return "NORMAL"


def _compute_ems_impact(alert_level: str, disease: str) -> str:
    factor = DISEASE_CONFIG.get(disease, DISEASE_CONFIG["grippe"])["ems_impact_factor"]
    if alert_level == "ÉPIDÉMIE":
        pct = int((factor - 1) * 100 * 1.5)
        return f"+{pct}% d'appels EMS estimés (seuil épidémique dépassé)"
    elif alert_level == "VIGILANCE":
        pct = int((factor - 1) * 100 * 0.7)
        return f"+{pct}% d'appels EMS estimés (phase de vigilance)"
    return "Impact EMS nominal"


def _generate_recommendation(disease: str, alert_level: str, peak_week: int) -> str:
    label = DISEASE_CONFIG[disease]["label"]
    if alert_level == "ÉPIDÉMIE":
        return (
            f"[ALERTE ÉPIDÉMIQUE] {label} : seuil épidémique dépassé. "
            f"Renforcer les effectifs EMS, activer le protocole de triage renforcé, "
            f"coordonner avec les urgences HUG/CHUV. Pic prévu semaine {peak_week}."
        )
    elif alert_level == "VIGILANCE":
        return (
            f"[VIGILANCE] {label} : incidence en hausse. "
            f"Surveiller les indicateurs quotidiennement, préparer les ressources supplémentaires. "
            f"Pic prévu semaine {peak_week}."
        )
    return f"{label} : situation normale. Surveillance de routine."


# ─── Modèle principal ────────────────────────────────────────────────────────

class EpidemicEarlyWarningModel:
    """
    Modèle de détection précoce d'épidémies pour le Grand Genève.
    Utilise les données Sentinelles FR + SARIMAX pour prédire J+14.
    """

    def __init__(self):
        self._cache: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_hours = 6

    def _is_cache_valid(self) -> bool:
        if self._cache is None or self._cache_time is None:
            return False
        return (datetime.utcnow() - self._cache_time).total_seconds() < self._cache_ttl_hours * 3600

    def predict(self, force_refresh: bool = False) -> Dict[str, Any]:
        if not force_refresh and self._is_cache_valid():
            return self._cache  # type: ignore

        result = self._run_prediction()
        self._cache = result
        self._cache_time = datetime.utcnow()
        return result

    def _run_prediction(self) -> Dict[str, Any]:
        today = date.today()
        predictions_by_disease = {}
        overall_alert = "NORMAL"
        alert_priority = {"NORMAL": 0, "VIGILANCE": 1, "ÉPIDÉMIE": 2}
        used_model = "SARIMAX"
        status = "live"

        for disease, config in DISEASE_CONFIG.items():
            # 1. Tenter de récupérer les données Sentinelles réelles
            raw_data = _fetch_sentinelles_series(disease)
            if raw_data and len(raw_data) >= 20:
                series = []
                for record in raw_data:
                    val = record.get("inc", record.get("incidence", record.get("value", 0)))
                    try:
                        series.append(float(val))
                    except (ValueError, TypeError):
                        series.append(0.0)
                data_source = "Sentinelles FR (données réelles)"
            else:
                # Fallback sur données synthétiques
                series = _generate_synthetic_series(disease)
                data_source = "Données synthétiques (Sentinelles indisponible)"
                status = "fallback"

            current_incidence = series[-1] if series else 0.0

            # 2. Ajuster SARIMAX ou fallback Serfling
            sarimax_preds = _fit_sarimax(series, horizon=14)
            if sarimax_preds:
                horizon_preds = sarimax_preds  # 14 valeurs journalières
                model_used = "SARIMAX(1,1,1)(1,1,1,52)"
            else:
                # Serfling retourne des semaines → interpoler en jours
                weekly_preds = _serfling_forecast(series, horizon_days=14)
                horizon_preds = []
                for w in weekly_preds:
                    horizon_preds.extend([w] * 7)
                horizon_preds = horizon_preds[:14]
                model_used = "Serfling (fallback analytique)"
                used_model = "Serfling"

            # 3. Construire les prédictions J+1 à J+14
            daily_predictions = []
            peak_incidence = current_incidence
            peak_day = 0
            for i, pred_val in enumerate(horizon_preds):
                pred_date = today + timedelta(days=i + 1)
                alert = _compute_alert_level(pred_val, disease)
                if pred_val > peak_incidence:
                    peak_incidence = pred_val
                    peak_day = i + 1
                daily_predictions.append({
                    "date": pred_date.strftime("%Y-%m-%d"),
                    "day_label": pred_date.strftime("%a %d/%m"),
                    "incidence_per_100k": round(pred_val, 1),
                    "alert_level": alert,
                    "ems_impact": _compute_ems_impact(alert, disease),
                })

            # 4. Niveau d'alerte maximal sur J+14
            max_alert = max(
                (d["alert_level"] for d in daily_predictions),
                key=lambda a: alert_priority.get(a, 0),
                default="NORMAL",
            )
            if alert_priority.get(max_alert, 0) > alert_priority.get(overall_alert, 0):
                overall_alert = max_alert

            # Semaine du pic (approximation)
            peak_week = math.ceil(peak_day / 7) if peak_day > 0 else 1

            predictions_by_disease[disease] = {
                "disease": disease,
                "label": config["label"],
                "current_incidence": round(current_incidence, 1),
                "epidemic_threshold": config["epidemic_threshold"],
                "warning_threshold": config["warning_threshold"],
                "current_alert": _compute_alert_level(current_incidence, disease),
                "max_alert_14d": max_alert,
                "peak_incidence_14d": round(peak_incidence, 1),
                "peak_day": peak_day,
                "ems_impact": _compute_ems_impact(max_alert, disease),
                "recommendation": _generate_recommendation(disease, max_alert, peak_week),
                "data_source": data_source,
                "model_used": model_used,
                "daily_predictions": daily_predictions,
            }

        # Résumé global
        most_critical = max(
            predictions_by_disease.values(),
            key=lambda d: alert_priority.get(d["max_alert_14d"], 0),
        )

        return {
            "model": used_model,
            "status": status,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "region": REGION_LABEL,
            "overall_alert_level": overall_alert,
            "horizon_days": 14,
            "diseases": predictions_by_disease,
            "most_critical_disease": most_critical["disease"],
            "global_recommendation": most_critical["recommendation"],
            "ecdc_supplement": _fetch_ecdc_flu(),
            "data_sources": [
                "Sentinelles FR (réseau de médecins sentinelles)",
                "ECDC (European Centre for Disease Prevention and Control)",
                "Données synthétiques (fallback si API indisponible)",
            ],
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

epidemic_model_singleton = EpidemicEarlyWarningModel()
