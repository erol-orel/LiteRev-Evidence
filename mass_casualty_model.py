"""
mass_casualty_model.py — Simulation et aide à la décision pour événements à victimes multiples
==========================================================================
Scénario GESICA : mass-casualty
Priorité : 3

Base scientifique :
- Lerner et al. (2011) : SALT triage — Sort, Assess, Lifesaving interventions, Treatment/Transport
- Kahn et al. (2009) : Systematic review of mass casualty triage systems
- Frykberg et al. (2002) : Medical management of disasters and mass casualties from terrorist bombings
- Hick et al. (2012) : Crisis standards of care — systematic review
- Auf der Heide (2006) : Common misconceptions about disasters
- Sacco et al. (2005) : STM (Sacco Triage Method) — evidence-based MCI triage
- Jenkins et al. (2008) : Explosions and blast injuries

Algorithme :
- Simulation Monte-Carlo pour estimer la distribution des victimes
- Triage SALT (Sort, Assess, Lifesaving interventions, Treatment/Transport)
- Calcul des besoins en ressources (ambulances, médecins, lits)
- Identification des hôpitaux de destination selon la capacité
- Recommandations d'activation du plan ORSEC/NOVI

Données utilisées :
- Type d'événement (explosion, accident transport, intoxication collective, etc.)
- Nombre estimé de victimes
- Localisation (distance aux hôpitaux)
- Ressources disponibles (ambulances, SMUR, hélicoptères)
- Capacité hospitalière en temps réel (simulée)
"""

from __future__ import annotations
import logging
import math
import random
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("mass-casualty")

# ─── Types d'événements MCI ───────────────────────────────────────────────────
MCI_TYPES = {
    "explosion": {
        "label": "Explosion / Attentat",
        "distribution": {"immediate": 0.20, "delayed": 0.35, "minimal": 0.30, "expectant": 0.10, "deceased": 0.05},
        "blast_injuries": True,
        "contamination_risk": False,
        "reference": "Frykberg (2002), Jenkins et al. (2008)",
    },
    "transport_accident": {
        "label": "Accident transport (bus, train, avion)",
        "distribution": {"immediate": 0.15, "delayed": 0.30, "minimal": 0.40, "expectant": 0.08, "deceased": 0.07},
        "blast_injuries": False,
        "contamination_risk": False,
        "reference": "Auf der Heide (2006)",
    },
    "chemical": {
        "label": "Intoxication chimique collective",
        "distribution": {"immediate": 0.25, "delayed": 0.40, "minimal": 0.25, "expectant": 0.07, "deceased": 0.03},
        "blast_injuries": False,
        "contamination_risk": True,
        "reference": "Hick et al. (2012)",
    },
    "building_collapse": {
        "label": "Effondrement de bâtiment",
        "distribution": {"immediate": 0.30, "delayed": 0.25, "minimal": 0.25, "expectant": 0.12, "deceased": 0.08},
        "blast_injuries": False,
        "contamination_risk": False,
        "reference": "Auf der Heide (2006)",
    },
    "mass_shooting": {
        "label": "Fusillade / Attaque armée",
        "distribution": {"immediate": 0.35, "delayed": 0.30, "minimal": 0.25, "expectant": 0.05, "deceased": 0.05},
        "blast_injuries": False,
        "contamination_risk": False,
        "reference": "Frykberg (2002)",
    },
    "industrial_accident": {
        "label": "Accident industriel",
        "distribution": {"immediate": 0.20, "delayed": 0.30, "minimal": 0.35, "expectant": 0.08, "deceased": 0.07},
        "blast_injuries": False,
        "contamination_risk": True,
        "reference": "Hick et al. (2012)",
    },
}

# ─── Niveaux SALT ─────────────────────────────────────────────────────────────
SALT_CATEGORIES = {
    "immediate": {
        "label": "IMMÉDIAT (Rouge)",
        "color": "red",
        "description": "Pronostic vital engagé — intervention < 1 heure",
        "interventions": ["Contrôle hémorragie", "Ouverture voies aériennes", "Décompression pneumothorax"],
        "transport_priority": 1,
    },
    "delayed": {
        "label": "DIFFÉRÉ (Jaune)",
        "color": "yellow",
        "description": "Blessé grave mais stable — peut attendre 1-4 heures",
        "interventions": ["Immobilisation fractures", "Analgésie", "Perfusion"],
        "transport_priority": 2,
    },
    "minimal": {
        "label": "MINIMAL (Vert)",
        "color": "green",
        "description": "Blessé léger — peut attendre ou se déplacer seul",
        "interventions": ["Pansements", "Soutien psychologique"],
        "transport_priority": 3,
    },
    "expectant": {
        "label": "EXPECTANT (Noir rayé)",
        "color": "gray",
        "description": "Pronostic fatal malgré soins — ressources limitées",
        "interventions": ["Soins de confort", "Analgésie"],
        "transport_priority": 4,
    },
    "deceased": {
        "label": "DÉCÉDÉ (Noir)",
        "color": "black",
        "description": "Décès constaté",
        "interventions": [],
        "transport_priority": 5,
    },
}

# ─── Hôpitaux Grand Genève ────────────────────────────────────────────────────
HOSPITALS = [
    {"name": "HUG", "city": "Genève", "lat": 46.1936, "lon": 6.1487,
     "capacity_trauma": 8, "capacity_surgery": 6, "capacity_icu": 20, "has_burn_unit": True, "country": "CH"},
    {"name": "CHUV", "city": "Lausanne", "lat": 46.5247, "lon": 6.6147,
     "capacity_trauma": 6, "capacity_surgery": 5, "capacity_icu": 18, "has_burn_unit": True, "country": "CH"},
    {"name": "CH Annecy-Genevois", "city": "Annecy", "lat": 45.9167, "lon": 6.1333,
     "capacity_trauma": 4, "capacity_surgery": 4, "capacity_icu": 10, "has_burn_unit": False, "country": "FR"},
    {"name": "CHU Grenoble", "city": "Grenoble", "lat": 45.1885, "lon": 5.7245,
     "capacity_trauma": 5, "capacity_surgery": 5, "capacity_icu": 15, "has_burn_unit": True, "country": "FR"},
    {"name": "Hôpital de la Tour", "city": "Meyrin", "lat": 46.2333, "lon": 6.0833,
     "capacity_trauma": 2, "capacity_surgery": 3, "capacity_icu": 6, "has_burn_unit": False, "country": "CH"},
    {"name": "CH Pays de Gex", "city": "Gex", "lat": 46.3333, "lon": 6.0667,
     "capacity_trauma": 1, "capacity_surgery": 1, "capacity_icu": 3, "has_burn_unit": False, "country": "FR"},
]

# ─── Ressources EMS disponibles ───────────────────────────────────────────────
EMS_RESOURCES = {
    "ambulances_amu": {"label": "Ambulances AMU", "count": 12, "capacity": 1, "response_min": 8},
    "smur_units": {"label": "Unités SMUR", "count": 4, "capacity": 1, "response_min": 10},
    "helicopters": {"label": "Hélicoptères médicaux", "count": 2, "capacity": 1, "response_min": 15},
    "fire_rescue": {"label": "Pompiers secours", "count": 8, "capacity": 4, "response_min": 7},
    "civil_protection": {"label": "Protection civile", "count": 20, "capacity": 4, "response_min": 30},
}


class MassCasualtyModel:
    """Modèle de simulation et aide à la décision pour événements à victimes multiples."""

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._cache_ts: datetime | None = None
        self._cache_ttl = 300  # 5 minutes (données dynamiques)
        random.seed(42)

    def _cache_valid(self) -> bool:
        if self._cache_ts is None:
            return False
        return (datetime.now(timezone.utc) - self._cache_ts).total_seconds() < self._cache_ttl

    def _monte_carlo_simulation(
        self,
        n_victims: int,
        event_type: str,
        n_simulations: int = 1000,
    ) -> dict[str, Any]:
        """
        Simulation Monte-Carlo de la distribution SALT selon le type d'événement.
        Retourne la distribution moyenne et les intervalles de confiance à 95%.
        """
        dist = MCI_TYPES[event_type]["distribution"]
        categories = list(dist.keys())
        probs = list(dist.values())

        # Normaliser les probabilités
        total = sum(probs)
        probs = [p / total for p in probs]

        # Simulations
        results = {cat: [] for cat in categories}
        for _ in range(n_simulations):
            counts = {cat: 0 for cat in categories}
            for _ in range(n_victims):
                # Tirage aléatoire selon la distribution
                r = random.random()
                cumul = 0
                for cat, prob in zip(categories, probs):
                    cumul += prob
                    if r <= cumul:
                        counts[cat] += 1
                        break
            for cat in categories:
                results[cat].append(counts[cat])

        # Statistiques
        summary = {}
        for cat in categories:
            vals = sorted(results[cat])
            n = len(vals)
            summary[cat] = {
                "mean": round(sum(vals) / n, 1),
                "median": vals[n // 2],
                "ci95_low": vals[int(n * 0.025)],
                "ci95_high": vals[int(n * 0.975)],
                "label": SALT_CATEGORIES[cat]["label"],
                "color": SALT_CATEGORIES[cat]["color"],
            }

        return summary

    def _compute_resource_needs(self, salt_distribution: dict[str, Any]) -> dict[str, Any]:
        """
        Calcule les besoins en ressources selon la distribution SALT.
        Basé sur les ratios recommandés par l'INSERM/SAMU de France.
        """
        immediate = salt_distribution["immediate"]["mean"]
        delayed = salt_distribution["delayed"]["mean"]
        minimal = salt_distribution["minimal"]["mean"]

        # Besoins en ambulances (1 SMUR pour 2 immédiats, 1 AMU pour 1 différé)
        smur_needed = math.ceil(immediate / 2)
        amu_needed = math.ceil(delayed)
        transport_needed = math.ceil(minimal / 4)  # 4 minimaux par véhicule léger

        # Besoins en médecins (1 médecin pour 5 immédiats, 1 infirmier pour 3 différés)
        doctors_needed = math.ceil(immediate / 5) + math.ceil(delayed / 10)
        nurses_needed = math.ceil(immediate / 3) + math.ceil(delayed / 5)

        # Besoins en lits hospitaliers
        icu_beds_needed = math.ceil(immediate * 0.6)
        surgery_needed = math.ceil(immediate * 0.4 + delayed * 0.2)
        regular_beds_needed = math.ceil(delayed * 0.8 + minimal * 0.1)

        # Ressources disponibles vs besoins
        available_smur = EMS_RESOURCES["smur_units"]["count"]
        available_amu = EMS_RESOURCES["ambulances_amu"]["count"]
        available_heli = EMS_RESOURCES["helicopters"]["count"]

        deficit_smur = max(0, smur_needed - available_smur)
        deficit_amu = max(0, amu_needed - available_amu)

        return {
            "transport": {
                "smur_needed": smur_needed,
                "amu_needed": amu_needed,
                "transport_light_needed": transport_needed,
                "helicopters_recommended": min(2, math.ceil(immediate / 5)),
                "deficit_smur": deficit_smur,
                "deficit_amu": deficit_amu,
            },
            "personnel": {
                "doctors_needed": doctors_needed,
                "nurses_needed": nurses_needed,
                "paramedics_needed": math.ceil((immediate + delayed) / 3),
            },
            "hospital_capacity": {
                "icu_beds_needed": icu_beds_needed,
                "surgery_rooms_needed": surgery_needed,
                "regular_beds_needed": regular_beds_needed,
                "total_hospital_capacity_available": sum(h["capacity_trauma"] for h in HOSPITALS),
                "total_icu_available": sum(h["capacity_icu"] for h in HOSPITALS),
            },
            "mutual_aid_required": deficit_smur > 0 or deficit_amu > 2,
        }

    def _plan_hospital_distribution(self, salt_distribution: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Planifie la distribution des victimes vers les hôpitaux selon leur capacité.
        Priorise les immédiats vers les centres de niveau 1.
        """
        immediate = int(salt_distribution["immediate"]["mean"])
        delayed = int(salt_distribution["delayed"]["mean"])
        minimal = int(salt_distribution["minimal"]["mean"])

        # Trier les hôpitaux par capacité trauma décroissante
        sorted_hospitals = sorted(HOSPITALS, key=lambda h: h["capacity_trauma"], reverse=True)

        distribution = []
        remaining_immediate = immediate
        remaining_delayed = delayed
        remaining_minimal = minimal

        for hospital in sorted_hospitals:
            if remaining_immediate <= 0 and remaining_delayed <= 0 and remaining_minimal <= 0:
                break

            assigned_immediate = min(remaining_immediate, hospital["capacity_trauma"])
            remaining_immediate -= assigned_immediate

            assigned_delayed = min(remaining_delayed, hospital["capacity_surgery"])
            remaining_delayed -= assigned_delayed

            assigned_minimal = min(remaining_minimal, max(0, hospital["capacity_icu"] - assigned_immediate))
            remaining_minimal -= assigned_minimal

            # Distance depuis Genève centre
            dlat = hospital["lat"] - 46.2044
            dlon = hospital["lon"] - 6.1432
            dist_km = math.sqrt(dlat**2 + dlon**2) * 111.0
            transport_min = round(dist_km / 60.0 * 60, 0)

            if assigned_immediate + assigned_delayed + assigned_minimal > 0:
                distribution.append({
                    "hospital": hospital["name"],
                    "city": hospital["city"],
                    "country": hospital["country"],
                    "distance_km": round(dist_km, 1),
                    "transport_time_min": transport_min,
                    "assigned_immediate": assigned_immediate,
                    "assigned_delayed": assigned_delayed,
                    "assigned_minimal": assigned_minimal,
                    "total_assigned": assigned_immediate + assigned_delayed + assigned_minimal,
                    "has_burn_unit": hospital["has_burn_unit"],
                })

        return distribution

    def _generate_activation_checklist(self, n_victims: int, event_type: str) -> list[dict[str, Any]]:
        """Génère la checklist d'activation du plan ORSEC/NOVI."""
        event_info = MCI_TYPES[event_type]
        checklist = [
            {"step": 1, "action": "Déclencher le plan NOVI (Nombreuses Victimes)", "responsible": "Médecin régulateur SAMU", "time_target": "T+0"},
            {"step": 2, "action": "Alerter le Préfet / Directeur des Opérations de Secours", "responsible": "SAMU / SDIS", "time_target": "T+2 min"},
            {"step": 3, "action": "Établir le Poste Médical Avancé (PMA)", "responsible": "Médecin DSM", "time_target": "T+10 min"},
            {"step": 4, "action": "Activer le triage SALT — Tri primaire", "responsible": "Équipes EMS sur site", "time_target": "T+15 min"},
            {"step": 5, "action": "Alerter les hôpitaux de la zone (plan blanc)", "responsible": "SAMU coordinateur", "time_target": "T+15 min"},
            {"step": 6, "action": "Mettre en place le circuit de transport (rouge → niveau 1)", "responsible": "Médecin coordinateur PMA", "time_target": "T+20 min"},
        ]

        if event_info["contamination_risk"]:
            checklist.insert(2, {
                "step": "2b",
                "action": "Activer le plan NRBC — Zone d'exclusion + décontamination",
                "responsible": "SDIS / SAMU",
                "time_target": "T+5 min",
            })

        if n_victims >= 50:
            checklist.append({
                "step": 7,
                "action": "Demander renforts régionaux (SAMU voisins, réserve sanitaire)",
                "responsible": "ARS / Préfecture",
                "time_target": "T+30 min",
            })

        if n_victims >= 100:
            checklist.append({
                "step": 8,
                "action": "Activer le plan ORSEC zonal — renforts nationaux",
                "responsible": "Préfet de zone",
                "time_target": "T+45 min",
            })

        return checklist

    def predict(self, n_victims: int = 50, event_type: str = "transport_accident") -> dict[str, Any]:
        """Exécute la simulation MCI et retourne les résultats."""
        cache_key = f"{n_victims}_{event_type}"
        if self._cache_valid() and self._cache.get("_key") == cache_key:
            return self._cache

        now = datetime.now(timezone.utc)
        event_info = MCI_TYPES.get(event_type, MCI_TYPES["transport_accident"])

        # Simulation Monte-Carlo
        salt_distribution = self._monte_carlo_simulation(n_victims, event_type, n_simulations=500)

        # Ressources nécessaires
        resource_needs = self._compute_resource_needs(salt_distribution)

        # Distribution hospitalière
        hospital_distribution = self._plan_hospital_distribution(salt_distribution)

        # Checklist d'activation
        activation_checklist = self._generate_activation_checklist(n_victims, event_type)

        # Niveau d'alerte
        if n_victims >= 100 or resource_needs["mutual_aid_required"]:
            alert_level = "CRITIQUE"
        elif n_victims >= 30:
            alert_level = "ÉLEVÉ"
        elif n_victims >= 10:
            alert_level = "MODÉRÉ"
        else:
            alert_level = "VIGILANCE"

        # Recommandations
        recommendations = [
            f"[{alert_level}] Événement {event_info['label']} — {n_victims} victimes estimées.",
            f"Immédiats estimés : {salt_distribution['immediate']['mean']:.0f} (IC95% : {salt_distribution['immediate']['ci95_low']}-{salt_distribution['immediate']['ci95_high']}).",
        ]

        if resource_needs["mutual_aid_required"]:
            recommendations.append(
                f"[RENFORTS NÉCESSAIRES] Déficit SMUR : {resource_needs['transport']['deficit_smur']}, "
                f"Déficit AMU : {resource_needs['transport']['deficit_amu']}. Activer aide mutuelle."
            )

        if event_info["contamination_risk"]:
            recommendations.append(
                "[NRBC] Risque de contamination — Activer protocole NRBC, zone d'exclusion obligatoire avant accès."
            )

        if event_info["blast_injuries"]:
            recommendations.append(
                "[BLAST] Explosion — Rechercher lésions blast (tympans, poumons, intestin). "
                "Victimes asymptomatiques à surveiller 6h minimum."
            )

        result = {
            "model": "MassCasualty v1.0",
            "status": "live",
            "generated_at": now.isoformat(),
            "region": "Grand Genève (CH/FR)",
            "overall_alert_level": alert_level,
            "_key": cache_key,

            "scenario": {
                "n_victims": n_victims,
                "event_type": event_type,
                "event_label": event_info["label"],
                "contamination_risk": event_info["contamination_risk"],
                "blast_injuries": event_info["blast_injuries"],
            },

            "salt_distribution": salt_distribution,
            "salt_categories": SALT_CATEGORIES,
            "resource_needs": resource_needs,
            "ems_resources_available": EMS_RESOURCES,
            "hospital_distribution": hospital_distribution,
            "activation_checklist": activation_checklist,
            "mci_types": {k: {"label": v["label"], "contamination_risk": v["contamination_risk"], "blast_injuries": v["blast_injuries"]} for k, v in MCI_TYPES.items()},

            "recommendations": recommendations,
            "scientific_references": [
                "Lerner et al. (2011). SALT triage. Disaster Med Public Health Prep.",
                "Kahn et al. (2009). Systematic review of mass casualty triage. Prehosp Emerg Care.",
                "Frykberg et al. (2002). Medical management of disasters from terrorist bombings. J Trauma.",
                "Hick et al. (2012). Crisis standards of care. Ann Emerg Med.",
                "Jenkins et al. (2008). Explosions and blast injuries. Ann Emerg Med.",
            ],
            "data_sources": ["SALT triage algorithm", "ORSEC/NOVI (France)", "SIAM (Suisse)", "Capacités hospitalières Grand Genève"],
        }

        self._cache = result
        self._cache_ts = now
        return result


mass_casualty_model_singleton = MassCasualtyModel()
