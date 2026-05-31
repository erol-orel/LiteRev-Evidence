"""
emergency_call_qualification_model.py
Qualification des Appels d'Urgence & Priorisation (NLP + scoring multi-critères)
Couvre : emergency-call-qualification + call-prioritization
Basé sur : MPDS (Medical Priority Dispatch System), SFMU référentiel CRRA15,
           Clawson JEMS (2007), Sporer AEM (2006), Lerner Prehosp Emerg Care (2012)
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Any

# ─── Dictionnaire de mots-clés cliniques ─────────────────────────────────────
# Chaque catégorie est associée à un score de gravité (1-5) et à une ressource

KEYWORD_RULES: list[dict[str, Any]] = [
    # Niveau 5 — Engagement immédiat SMUR
    {"keywords": ["arrêt cardiaque", "cardiac arrest", "no pulse", "pas de pouls", "inconscient ne respire pas"], "priority": 5, "resource": "SMUR + Ambulance", "category": "cardiac-arrest"},
    {"keywords": ["ne respire plus", "apnée", "arrêt respiratoire", "stopped breathing"], "priority": 5, "resource": "SMUR + Ambulance", "category": "respiratory-arrest"},
    {"keywords": ["hémorragie massive", "saignement abondant", "sang partout", "massive bleeding"], "priority": 5, "resource": "SMUR + Ambulance", "category": "hemorrhage"},
    {"keywords": ["noyade", "drowning", "submersion"], "priority": 5, "resource": "SMUR + Ambulance + GRIMP", "category": "drowning"},
    {"keywords": ["pendaison", "strangulation", "hanging"], "priority": 5, "resource": "SMUR + Police", "category": "hanging"},
    {"keywords": ["polytraumatisme", "accident grave", "éjection", "écrasement"], "priority": 5, "resource": "SMUR + Ambulance", "category": "major-trauma"},
    # Niveau 4 — SMUR probable
    {"keywords": ["douleur thoracique", "chest pain", "douleur poitrine", "oppression thoracique"], "priority": 4, "resource": "SMUR + Ambulance", "category": "chest-pain"},
    {"keywords": ["avc", "stroke", "paralysie", "hémiplégie", "déviation bouche", "face drooping"], "priority": 4, "resource": "SMUR + Ambulance", "category": "stroke"},
    {"keywords": ["détresse respiratoire", "dyspnée sévère", "ne peut plus parler", "respiratory distress"], "priority": 4, "resource": "SMUR + Ambulance", "category": "respiratory-distress"},
    {"keywords": ["convulsions", "crise épileptique", "seizure", "tonic-clonic"], "priority": 4, "resource": "SMUR + Ambulance", "category": "seizure"},
    {"keywords": ["inconscient", "perte de connaissance", "unresponsive", "unconscious"], "priority": 4, "resource": "SMUR + Ambulance", "category": "unconscious"},
    {"keywords": ["anaphylaxie", "choc allergique", "anaphylactic", "allergie sévère"], "priority": 4, "resource": "SMUR + Ambulance", "category": "anaphylaxis"},
    {"keywords": ["accouchement imminent", "naissance imminente", "crowning", "tête visible"], "priority": 4, "resource": "SMUR + Sage-femme", "category": "obstetric"},
    # Niveau 3 — Ambulance urgente
    {"keywords": ["fracture ouverte", "déformation membre", "open fracture"], "priority": 3, "resource": "Ambulance ASSU", "category": "fracture"},
    {"keywords": ["brûlure étendue", "burn", "brûlure visage", "brûlure voies aériennes"], "priority": 3, "resource": "Ambulance ASSU", "category": "burns"},
    {"keywords": ["hypoglycémie", "diabétique inconscient", "hypoglycemia", "sucre bas"], "priority": 3, "resource": "Ambulance ASSU", "category": "hypoglycemia"},
    {"keywords": ["douleur abdominale intense", "ventre dur", "abdomen rigide"], "priority": 3, "resource": "Ambulance ASSU", "category": "abdominal-pain"},
    {"keywords": ["fièvre élevée", "température 40", "hyperthermie", "high fever"], "priority": 3, "resource": "Ambulance ASSU", "category": "fever"},
    {"keywords": ["malaise", "syncope", "évanouissement", "fainting", "perte connaissance brève"], "priority": 3, "resource": "Ambulance ASSU", "category": "syncope"},
    {"keywords": ["intoxication", "overdose", "empoisonnement", "ingestion médicaments"], "priority": 3, "resource": "Ambulance ASSU + SAMU", "category": "poisoning"},
    # Niveau 2 — Ambulance non urgente
    {"keywords": ["douleur modérée", "plaie simple", "chute sans perte connaissance"], "priority": 2, "resource": "Ambulance VSL", "category": "minor-trauma"},
    {"keywords": ["vomissements", "diarrhée", "nausées persistantes"], "priority": 2, "resource": "Médecin de garde", "category": "gastro"},
    {"keywords": ["maux de tête", "headache", "céphalées"], "priority": 2, "resource": "Médecin de garde", "category": "headache"},
    # Niveau 1 — Conseil médical
    {"keywords": ["conseil", "question médicale", "ordonnance", "renouvellement"], "priority": 1, "resource": "Conseil médical téléphonique", "category": "advice"},
]

# ─── Red flags — facteurs aggravants ─────────────────────────────────────────
RED_FLAGS = [
    ("nourrisson", "Nourrisson < 1 an — risque aggravé"),
    ("bébé", "Nourrisson — risque aggravé"),
    ("nouveau-né", "Nouveau-né — risque aggravé"),
    ("personne âgée", "Patient âgé — risque de sous-estimation"),
    ("anticoagulant", "Traitement anticoagulant — risque hémorragique"),
    ("immunodéprimé", "Immunodépression — risque infectieux aggravé"),
    ("dialyse", "Insuffisance rénale chronique — risque métabolique"),
    ("grossesse", "Grossesse — prise en charge spécifique"),
    ("enceinte", "Grossesse — prise en charge spécifique"),
    ("diabétique", "Diabète — risque hypoglycémie/acidocétose"),
    ("pacemaker", "Porteur de pacemaker — risque rythmique"),
    ("seul", "Patient seul — pas de témoin disponible"),
    ("inaccessible", "Accès difficile — délai de secours augmenté"),
]

def _normalize(text: str) -> str:
    """Normalise le texte pour la recherche de mots-clés."""
    return text.lower().strip()

def qualify_call(complaint_text: str, age: int | None = None, caller_type: str = "patient") -> dict[str, Any]:
    """
    Qualifie un appel d'urgence à partir du texte de la plainte.
    Retourne la priorité, la ressource recommandée et les drapeaux rouges.
    """
    text_norm = _normalize(complaint_text)
    matched_rules = []

    for rule in KEYWORD_RULES:
        for kw in rule["keywords"]:
            if kw in text_norm:
                matched_rules.append(rule)
                break

    # Prendre la règle de plus haute priorité
    if matched_rules:
        best_rule = max(matched_rules, key=lambda r: r["priority"])
        priority = best_rule["priority"]
        resource = best_rule["resource"]
        category = best_rule["category"]
        matched_keywords = [kw for r in matched_rules for kw in r["keywords"] if kw in text_norm]
    else:
        priority = 2
        resource = "Médecin de garde"
        category = "unclassified"
        matched_keywords = []

    # Ajustement par l'âge
    age_flag = None
    if age is not None:
        if age < 1:
            priority = min(5, priority + 2)
            age_flag = "Nourrisson < 1 an — priorité augmentée"
        elif age < 5:
            priority = min(5, priority + 1)
            age_flag = "Enfant < 5 ans — priorité augmentée"
        elif age > 80:
            priority = min(5, priority + 1)
            age_flag = "Patient > 80 ans — priorité augmentée"

    # Détection des red flags
    detected_flags = []
    for flag_kw, flag_msg in RED_FLAGS:
        if flag_kw in text_norm:
            detected_flags.append(flag_msg)
    if age_flag:
        detected_flags.append(age_flag)

    # Niveau de priorité MPDS
    priority_labels = {
        5: ("OMEGA — Engagement immédiat", "red"),
        4: ("ALPHA — Urgence vitale", "red"),
        3: ("BRAVO — Urgence relative", "orange"),
        2: ("CHARLIE — Semi-urgent", "yellow"),
        1: ("DELTA — Non urgent", "green"),
    }
    label, color = priority_labels.get(priority, ("INCONNU", "gray"))

    return {
        "priority": priority,
        "priority_label": label,
        "priority_color": color,
        "recommended_resource": resource,
        "clinical_category": category,
        "matched_keywords": matched_keywords[:5],
        "red_flags": detected_flags,
        "caller_type": caller_type,
        "age": age,
    }


def prioritize_queue(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Priorise une file d'attente d'appels en attente de régulation.
    Trie par priorité décroissante, puis par durée d'attente.
    """
    qualified = []
    for i, call in enumerate(calls):
        q = qualify_call(
            complaint_text=call.get("complaint", ""),
            age=call.get("age"),
            caller_type=call.get("caller_type", "patient"),
        )
        q["call_id"] = call.get("id", f"CALL-{i+1:03d}")
        q["wait_seconds"] = call.get("wait_seconds", 0)
        # Bonus d'urgence si attente > 3 min
        if q["wait_seconds"] > 180 and q["priority"] >= 3:
            q["priority"] = min(5, q["priority"] + 1)
            q["priority_label"] += " (↑ attente > 3 min)"
        qualified.append(q)

    # Tri : priorité décroissante, puis attente décroissante
    sorted_calls = sorted(qualified, key=lambda x: (-x["priority"], -x["wait_seconds"]))

    critical = [c for c in sorted_calls if c["priority"] >= 4]
    urgent = [c for c in sorted_calls if c["priority"] == 3]
    non_urgent = [c for c in sorted_calls if c["priority"] <= 2]

    return {
        "model": "EmergencyCallQualification v1.0",
        "status": "live",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queue_summary": {
            "total_calls": len(calls),
            "critical_immediate": len(critical),
            "urgent": len(urgent),
            "non_urgent": len(non_urgent),
        },
        "prioritized_queue": sorted_calls,
        "immediate_actions": [c for c in sorted_calls if c["priority"] >= 4],
        "evidence_base": [
            "MPDS — Medical Priority Dispatch System (Clawson 2007)",
            "SFMU Référentiel CRRA15 (2020)",
            "Lerner Prehosp Emerg Care (2012) — Dispatch triage validation",
        ]
    }


# ─── Démo pour l'API ─────────────────────────────────────────────────────────
class EmergencyCallQualificationModel:
    def predict_demo(self) -> dict[str, Any]:
        demo_calls = [
            {"id": "CALL-001", "complaint": "Mon mari ne respire plus, il est inconscient, pas de pouls", "age": 67, "wait_seconds": 0},
            {"id": "CALL-002", "complaint": "Douleur thoracique intense depuis 20 minutes, transpiration", "age": 55, "wait_seconds": 45},
            {"id": "CALL-003", "complaint": "Déviation de la bouche, paralysie bras droit, parole difficile", "age": 72, "wait_seconds": 30},
            {"id": "CALL-004", "complaint": "Enfant fièvre élevée 40 degrés, convulsions", "age": 3, "wait_seconds": 120},
            {"id": "CALL-005", "complaint": "Chute, douleur modérée au poignet, pas de perte de connaissance", "age": 45, "wait_seconds": 300},
            {"id": "CALL-006", "complaint": "Vomissements depuis hier soir, demande conseil médical", "age": 28, "wait_seconds": 600},
        ]
        return prioritize_queue(demo_calls)


emergency_call_model_singleton = EmergencyCallQualificationModel()
