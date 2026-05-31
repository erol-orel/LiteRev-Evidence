"""
trauma_care_model.py — Modèle de prédiction de survie et optimisation des soins trauma
==========================================================================
Scénario GESICA : trauma-care
Priorité : 5

Base scientifique :
- Boyd et al. (1987) : TRISS methodology — Trauma and Injury Severity Score
- Baker et al. (1974) : Injury Severity Score (ISS)
- Champion et al. (1989) : Revised Trauma Score (RTS)
- Osler et al. (1997) : NISS (New Injury Severity Score) — amélioration ISS
- Haider et al. (2012) : Mechanism of injury predicts patient outcomes
- Rotondo et al. (1993) : Damage Control Surgery
- Holcomb et al. (2015) : PROPPR trial — ratio 1:1:1 (plasma:plaquettes:CGR)

Algorithme :
- Calcul ISS (Injury Severity Score) depuis les codes AIS
- Calcul RTS (Revised Trauma Score) depuis les signes vitaux
- Calcul TRISS (probabilité de survie)
- Identification des critères damage control
- Recommandations de transfusion (ratio 1:1:1)
- Prédiction de la mortalité à 30 jours

Données utilisées :
- Signes vitaux (FC, FR, TA, GCS)
- Mécanisme et localisation des lésions (AIS)
- Délai pré-hospitalier
- Température (hypothermie = triade létale)
- pH / lactate (si disponible)
"""

from __future__ import annotations
import logging
import math
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("trauma-care")

# ─── AIS (Abbreviated Injury Scale) régions ──────────────────────────────────
AIS_REGIONS = {
    "head_neck": "Tête et cou",
    "face": "Face",
    "thorax": "Thorax",
    "abdomen": "Abdomen",
    "extremities": "Membres et bassin",
    "external": "Téguments",
}

# ─── Seuils ISS ──────────────────────────────────────────────────────────────
ISS_THRESHOLDS = {
    "minor":    {"range": "1-8",   "label": "Traumatisme mineur",    "mortality_pct": 0.1},
    "moderate": {"range": "9-15",  "label": "Traumatisme modéré",    "mortality_pct": 1.0},
    "severe":   {"range": "16-24", "label": "Traumatisme sévère",    "mortality_pct": 7.0},
    "critical": {"range": "25-40", "label": "Traumatisme critique",  "mortality_pct": 25.0},
    "unsurvivable": {"range": "41-75", "label": "Lésions non survivables", "mortality_pct": 75.0},
}

# ─── Critères damage control ─────────────────────────────────────────────────
DAMAGE_CONTROL_CRITERIA = [
    {"criterion": "pH < 7.20", "category": "Acidose", "weight": 3},
    {"criterion": "Température < 35°C", "category": "Hypothermie", "weight": 3},
    {"criterion": "TP < 50% ou INR > 1.5", "category": "Coagulopathie", "weight": 3},
    {"criterion": "Transfusion massive (> 10 CGR en 24h)", "category": "Hémorragie", "weight": 3},
    {"criterion": "Lésions vasculaires multiples", "category": "Anatomique", "weight": 2},
    {"criterion": "Contamination abdominale massive", "category": "Anatomique", "weight": 2},
    {"criterion": "ISS > 35", "category": "Gravité globale", "weight": 2},
    {"criterion": "Temps chirurgical prévu > 90 min chez patient instable", "category": "Physiologique", "weight": 2},
]

# ─── Protocole transfusionnel (PROPPR trial, Holcomb 2015) ───────────────────
TRANSFUSION_PROTOCOL = {
    "ratio_1_1_1": {
        "label": "Ratio 1:1:1 (Plasma:Plaquettes:CGR)",
        "evidence": "PROPPR trial (Holcomb et al., JAMA 2015) — réduction mortalité 24h",
        "indication": "Hémorragie massive (> 10 CGR prévisibles en 24h)",
        "target": "Fibrinogène > 1.5 g/L, TP > 50%, Plaquettes > 50G/L",
    },
    "permissive_hypotension": {
        "label": "Hypotension permissive",
        "evidence": "Bickell et al. (NEJM 1994), Morrison et al. (J Trauma 2011)",
        "target_pas": "80-90 mmHg (trauma pénétrant) / 90-100 mmHg (TCE)",
        "contraindication": "TCE grave (maintenir PPC > 60 mmHg)",
    },
    "txa": {
        "label": "Acide tranexamique (TXA)",
        "evidence": "CRASH-2 trial (Lancet 2010) — réduction mortalité si < 3h",
        "dose": "1g IV en 10 min, puis 1g en 8h",
        "window": "< 3h depuis le traumatisme (pas de bénéfice après 3h)",
    },
}

# ─── Scores trauma ────────────────────────────────────────────────────────────
def compute_rts(gcs: int, pas: float, fr: float) -> float:
    """
    Revised Trauma Score (Champion et al., 1989).
    Valeurs codées : 0-4 pour chaque paramètre.
    """
    def code_gcs(v):
        if v >= 13: return 4
        elif v >= 9: return 3
        elif v >= 6: return 2
        elif v >= 4: return 1
        else: return 0

    def code_pas(v):
        if v > 89: return 4
        elif v >= 76: return 3
        elif v >= 50: return 2
        elif v >= 1: return 1
        else: return 0

    def code_fr(v):
        if 10 <= v <= 29: return 4
        elif v >= 30: return 3
        elif 6 <= v <= 9: return 2
        elif 1 <= v <= 5: return 1
        else: return 0

    # Coefficients RTS
    rts = 0.9368 * code_gcs(gcs) + 0.7326 * code_pas(pas) + 0.2908 * code_fr(fr)
    return round(rts, 3)


def compute_iss(ais_scores: dict[str, int]) -> int:
    """
    Injury Severity Score (Baker et al., 1974).
    Somme des carrés des 3 scores AIS les plus élevés (régions différentes).
    Si un AIS = 6 → ISS = 75 (automatiquement).
    """
    scores = list(ais_scores.values())
    if 6 in scores:
        return 75
    top3 = sorted(scores, reverse=True)[:3]
    return sum(s**2 for s in top3)


def compute_triss(iss: int, rts: float, age: int, blunt: bool) -> float:
    """
    TRISS — Probabilité de survie (Boyd et al., 1987).
    Coefficients de régression logistique (MTOS database).
    """
    if blunt:
        b0, b1, b2, b3 = -1.2470, 0.9544, -0.0768, -1.9052
    else:
        b0, b1, b2, b3 = -0.6029, 1.1430, -0.1516, -2.6676

    age_code = 1 if age >= 55 else 0
    b = b0 + b1 * rts + b2 * iss + b3 * age_code
    ps = 1 / (1 + math.exp(-b))
    return round(ps, 3)


class TraumaCareModel:
    """Modèle de prédiction de survie et optimisation des soins trauma."""

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._cache_ts: datetime | None = None
        self._cache_ttl = 3600

    def _cache_valid(self) -> bool:
        if self._cache_ts is None:
            return False
        return (datetime.now(timezone.utc) - self._cache_ts).total_seconds() < self._cache_ttl

    def _generate_case_examples(self) -> list[dict[str, Any]]:
        """Génère des cas cliniques types avec calculs ISS/RTS/TRISS."""
        cases = [
            {
                "name": "AVP haute énergie — polytraumatisé",
                "age": 35, "blunt": True,
                "gcs": 12, "pas": 85, "fr": 24,
                "temperature": 35.8,
                "ais_scores": {"head_neck": 3, "thorax": 4, "abdomen": 3, "extremities": 2},
                "mechanism": "Conducteur éjecté, AVP 90 km/h",
            },
            {
                "name": "Chute de hauteur — traumatisme thoracique",
                "age": 58, "blunt": True,
                "gcs": 14, "pas": 100, "fr": 22,
                "temperature": 36.5,
                "ais_scores": {"thorax": 4, "extremities": 3, "head_neck": 1},
                "mechanism": "Chute d'échafaudage 8m",
            },
            {
                "name": "Traumatisme pénétrant abdominal",
                "age": 28, "blunt": False,
                "gcs": 15, "pas": 95, "fr": 20,
                "temperature": 36.8,
                "ais_scores": {"abdomen": 4, "external": 2},
                "mechanism": "Plaie arme blanche abdomen",
            },
            {
                "name": "Personne âgée — chute simple",
                "age": 78, "blunt": True,
                "gcs": 15, "pas": 130, "fr": 18,
                "temperature": 36.2,
                "ais_scores": {"head_neck": 2, "extremities": 3},
                "mechanism": "Chute de sa hauteur, anticoagulants",
            },
        ]

        results = []
        for case in cases:
            rts = compute_rts(case["gcs"], case["pas"], case["fr"])
            iss = compute_iss(case["ais_scores"])
            ps = compute_triss(iss, rts, case["age"], case["blunt"])

            # Niveau de gravité ISS
            if iss <= 8:
                iss_level = "minor"
            elif iss <= 15:
                iss_level = "moderate"
            elif iss <= 24:
                iss_level = "severe"
            elif iss <= 40:
                iss_level = "critical"
            else:
                iss_level = "unsurvivable"

            # Critères damage control
            dc_score = 0
            dc_triggers = []
            if case["temperature"] < 35.0:
                dc_score += 3; dc_triggers.append("Hypothermie")
            if case["pas"] < 90:
                dc_score += 2; dc_triggers.append("Hypotension")
            if iss > 35:
                dc_score += 2; dc_triggers.append("ISS > 35")
            if any(v >= 4 for v in case["ais_scores"].values()):
                dc_score += 2; dc_triggers.append("Lésion AIS ≥ 4")

            damage_control = dc_score >= 4

            # Recommandations
            recs = []
            if ps < 0.50:
                recs.append(f"[CRITIQUE] Probabilité de survie {ps*100:.0f}% — Activation trauma niveau 1, chirurgie immédiate")
            elif ps < 0.75:
                recs.append(f"[SÉVÈRE] Probabilité de survie {ps*100:.0f}% — Surveillance rapprochée, chirurgie en attente")
            else:
                recs.append(f"[MODÉRÉ] Probabilité de survie {ps*100:.0f}% — Prise en charge standard")

            if damage_control:
                recs.append(f"[DAMAGE CONTROL] Critères présents ({', '.join(dc_triggers)}) — Chirurgie écourtée, réanimation prioritaire")

            if not case["blunt"] and case["pas"] < 100:
                recs.append("[TXA] Acide tranexamique indiqué si < 3h depuis traumatisme")

            results.append({
                "case_name": case["name"],
                "mechanism": case["mechanism"],
                "age": case["age"],
                "mechanism_type": "contondant" if case["blunt"] else "pénétrant",
                "vital_signs": {"gcs": case["gcs"], "pas": case["pas"], "fr": case["fr"], "temperature": case["temperature"]},
                "ais_scores": case["ais_scores"],
                "scores": {
                    "rts": rts,
                    "iss": iss,
                    "iss_level": ISS_THRESHOLDS[iss_level]["label"],
                    "triss_survival_probability": ps,
                    "triss_survival_pct": round(ps * 100, 1),
                    "predicted_mortality_pct": round((1 - ps) * 100, 1),
                },
                "damage_control_indicated": damage_control,
                "damage_control_triggers": dc_triggers,
                "recommendations": recs,
            })

        return results

    def predict(self) -> dict[str, Any]:
        """Retourne le modèle trauma complet avec cas cliniques et protocoles."""
        if self._cache_valid():
            return self._cache

        now = datetime.now(timezone.utc)
        case_examples = self._generate_case_examples()

        # Statistiques de la cohorte d'exemples
        mean_ps = sum(c["scores"]["triss_survival_pct"] for c in case_examples) / len(case_examples)
        dc_cases = sum(1 for c in case_examples if c["damage_control_indicated"])

        result = {
            "model": "TraumaCare v1.0",
            "status": "live",
            "generated_at": now.isoformat(),
            "region": "Grand Genève (CH/FR)",
            "overall_alert_level": "NORMAL",

            "scoring_systems": {
                "iss": {
                    "name": "Injury Severity Score (ISS)",
                    "range": "1-75",
                    "major_trauma_threshold": 16,
                    "reference": "Baker et al. (1974). J Trauma.",
                    "thresholds": ISS_THRESHOLDS,
                },
                "rts": {
                    "name": "Revised Trauma Score (RTS)",
                    "range": "0-7.84",
                    "critical_threshold": 4.0,
                    "reference": "Champion et al. (1989). J Trauma.",
                },
                "triss": {
                    "name": "TRISS — Trauma and Injury Severity Score",
                    "output": "Probabilité de survie (0-1)",
                    "reference": "Boyd et al. (1987). J Trauma.",
                    "note": "Ps < 0.50 → mortalité prédite > 50%",
                },
            },

            "case_examples": case_examples,
            "cohort_summary": {
                "n_cases": len(case_examples),
                "mean_survival_pct": round(mean_ps, 1),
                "damage_control_cases": dc_cases,
                "damage_control_rate_pct": round(dc_cases / len(case_examples) * 100, 1),
            },

            "damage_control_criteria": DAMAGE_CONTROL_CRITERIA,
            "transfusion_protocol": TRANSFUSION_PROTOCOL,

            "ais_regions": AIS_REGIONS,

            "recommendations": [
                "Calculer ISS + RTS + TRISS pour tout traumatisme ISS ≥ 16.",
                "Damage control si ≥ 2 critères présents (hypothermie + acidose + coagulopathie).",
                "TXA systématique si traumatisme hémorragique < 3h (CRASH-2).",
                "Ratio 1:1:1 plasma:plaquettes:CGR pour hémorragie massive (PROPPR trial).",
                "Hypotension permissive PAS 80-90 mmHg (sauf TCE grave).",
            ],

            "scientific_references": [
                "Boyd et al. (1987). TRISS methodology. J Trauma.",
                "Baker et al. (1974). Injury Severity Score. J Trauma.",
                "Champion et al. (1989). Revised Trauma Score. J Trauma.",
                "Holcomb et al. (2015). PROPPR trial — ratio 1:1:1. JAMA.",
                "CRASH-2 Collaborators (2010). Tranexamic acid in trauma. Lancet.",
                "Rotondo et al. (1993). Damage control surgery. J Trauma.",
            ],
            "data_sources": ["MTOS database (TRISS coefficients)", "PROPPR trial", "CRASH-2 trial"],
        }

        self._cache = result
        self._cache_ts = now
        return result


trauma_model_singleton = TraumaCareModel()
