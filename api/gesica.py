"""Built-in (GESICA) catalogue: metadata, stats, listing and detail.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from sqlalchemy import text

from gesica_i18n import localize_gesica as _localize_gesica

from .core import _msg, app, engine, logger
from .documents import _extract_gesica_evidence
from .corpus import _canonical_source

# GESICA_ENRICHED et GESICA_SCENARIO_METADATA sont désormais stockés en base de données (user_scenarios is_system=TRUE)
# Les imports statiques ci-dessous sont conservés pour compatibilité ascendante uniquement
try:
    from gesica_scenario_enriched_metadata import GESICA_ENRICHED as _GESICA_ENRICHED_LEGACY
except ImportError:
    _GESICA_ENRICHED_LEGACY: dict = {}

# Cache du dashboard /gesica/stats. Le calcul balaie TOUT le corpus literev puis
# applique _extract_gesica_evidence (regex) PAR document - O(N) Python, > 45 s sur
# un corpus de dizaines de milliers de docs (→ la requête HTTP dépassait le délai).
# On met en cache avec TTL et on rafraîchit EN ARRIÈRE-PLAN : la requête ne bloque
# jamais (elle sert le cache, éventuellement périmé, ou une réponse froide légère).
_GESICA_STATS_CACHE: dict[str, Any] = {"ts": 0.0, "data": None, "computing": False}
_GESICA_STATS_TTL = 900  # 15 min


def _compute_gesica_stats() -> dict[str, Any]:
    """Calcul LOURD des stats globales (balayage complet + regex par document)."""
    with engine.connect() as conn:
        docs = conn.execute(text(
            "SELECT id, title, abstract FROM literature_document WHERE project_context = 'literev'"
        )).mappings().all()
    horizons_count: dict[str, int] = {}
    uncertainty_count: dict[str, int] = {}
    evidence_strengths = {"weak": 0, "moderate": 0, "strong": 0}
    for doc in docs:
        signals = _extract_gesica_evidence(doc["title"], doc["abstract"], [])
        strength = signals["evidence_strength"]
        evidence_strengths[strength] = evidence_strengths.get(strength, 0) + 1
        for m in signals["uncertainty_handling"]:
            uncertainty_count[m] = uncertainty_count.get(m, 0) + 1
        if signals["forecast_horizon"]:
            h = signals["forecast_horizon"]
            horizons_count[h] = horizons_count.get(h, 0) + 1
    return {
        "total_documents": len(docs),
        "evidence_strength_distribution": evidence_strengths,
        "uncertainty_methods": dict(sorted(uncertainty_count.items(), key=lambda x: x[1], reverse=True)),
        "forecast_horizons": dict(sorted(horizons_count.items(), key=lambda x: x[1], reverse=True)),
    }


@app.get("/gesica/stats")
def get_gesica_stats() -> dict[str, Any]:
    """Statistiques globales du corpus LiteRev (cache TTL + rafraîchissement de fond)."""
    import time as _t
    import threading as _th
    now = _t.time()
    cached = _GESICA_STATS_CACHE["data"]
    if cached is not None and (now - _GESICA_STATS_CACHE["ts"]) < _GESICA_STATS_TTL:
        return cached
    # Périmé ou froid → déclenche UN rafraîchissement de fond (pas de calcul concurrent).
    if not _GESICA_STATS_CACHE["computing"]:
        _GESICA_STATS_CACHE["computing"] = True

        def _bg():
            try:
                r = _compute_gesica_stats()
                _GESICA_STATS_CACHE["data"] = r
                _GESICA_STATS_CACHE["ts"] = _t.time()
            except Exception as _e:
                logger.warning(f"gesica/stats refresh: {_e}")
            finally:
                _GESICA_STATS_CACHE["computing"] = False

        _th.Thread(target=_bg, daemon=True).start()
    if cached is not None:
        return cached  # sert le cache périmé pendant le rafraîchissement
    # Démarrage à froid : réponse légère immédiate (COUNT rapide) ; distributions à venir.
    try:
        with engine.connect() as conn:
            total = conn.execute(text(
                "SELECT COUNT(*) FROM literature_document WHERE project_context = 'literev'"
            )).scalar() or 0
    except Exception:
        total = 0
    return {
        "total_documents": int(total),
        "evidence_strength_distribution": {"weak": 0, "moderate": 0, "strong": 0},
        "uncertainty_methods": {},
        "forecast_horizons": {},
        "computing": True,
    }

@app.get("/geoai4ei/stats")
def get_geoai4ei_stats() -> dict[str, Any]:
    """Statistiques globales du corpus Urgences Hospitalières."""
    sql_diseases = text("""
        SELECT disease_or_condition, COUNT(*) as count
        FROM literature_document
        WHERE project_context = 'literev' AND disease_or_condition IS NOT NULL
        GROUP BY disease_or_condition
        ORDER BY count DESC
    """)
    sql_geo = text("""
        SELECT geographic_scope, COUNT(*) as count
        FROM literature_document
        WHERE project_context = 'literev' AND geographic_scope IS NOT NULL
        GROUP BY geographic_scope
        ORDER BY count DESC
    """)
    with engine.connect() as conn:
        diseases = {r["disease_or_condition"]: r["count"] for r in conn.execute(sql_diseases).mappings().all()}
        geo = {r["geographic_scope"]: r["count"] for r in conn.execute(sql_geo).mappings().all()}

    return {
        "diseases": diseases,
        "geographic_scopes": geo,
    }

# ─────────────────────────────────────────────────────────────────────────────
# GESICA Scenarios Metadata : 31 scénarios fins issus de la revue systématique
# ─────────────────────────────────────────────────────────────────────────────

GESICA_SCENARIO_METADATA: dict[str, dict[str, Any]] = {
    "cardiac-arrest-prediction": {
        "hidden": False,
        "title": "Prédiction de l'Arrêt Cardiaque Extra-Hospitalier (OHCA)",
        "description": "Modèles de prédiction spatio-temporelle de l'incidence des arrêts cardiorespiratoires (OHCA) basés sur l'apprentissage automatique, les rythmes circadiens, les données climatiques et météorologiques, visant à optimiser la chaîne de survie et le positionnement préventif des ressources.",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Déployer des algorithmes de détection acoustique de l'agonie respiratoire (gasping) assistés par IA au Centre 15/144",
            "Optimiser la couverture et le dispatch des premiers répondants (citizen responders) équipés de DEA via géolocalisation dynamique",
            "Ajuster préventivement le positionnement des SMUR et des ambulances de réanimation dans les zones à haut risque d'OHCA",
            "Intégrer les données de défibrillateurs connectés (IoT) pour une cartographie temps réel de l'accessibilité des DEA"
        ]
    },
    "stroke-detection": {
        "hidden": False,
        "title": "Détection Préhospitalière de l'AVC",
        "description": "Systèmes d'aide à la décision clinique pour l'identification précoce des accidents vasculaires cérébraux (AVC) sur le terrain, l'évaluation de la sévérité via des scores automatisés (FAST, NIHSS, LVO) et l'orientation optimale et directe vers les centres de reperfusion (thrombolyse/thrombectomie).",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Intégrer des échelles cliniques d'AVC automatisées et guidées par IA dans le dossier patient embarqué des ambulanciers",
            "Orienter directement et sans transit par les urgences générales vers l'Unité Neurovasculaire (UNV) de référence (HUG/CHUV)",
            "Déclencher une pré-alerte automatique pour l'équipe de neuroradiologie interventionnelle en cas de forte suspicion d'occlusion de gros vaisseau (LVO)",
            "Optimiser le délai porte-aiguille (door-to-needle) par la transmission préhospitalière sécurisée des données cliniques"
        ]
    },
    "trauma-severity-assessment": {
        "hidden": False,
        "title": "Évaluation de la Gravité des Traumatismes",
        "description": "Modèles prédictifs et scores de stratification du risque (ISS, RTS, TRISS) pour l'évaluation immédiate des traumatisés graves (accidents de la route, chutes, traumatismes de montagne) afin d'orienter sans délai vers les Trauma Centers de niveau adapté.",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Déployer des modèles prédictifs de besoin de transfusion massive (score de choc, score d'hémorragie) dès la prise en charge terrain",
            "Orienter systématiquement les traumatismes sévères (ISS > 15) vers un Trauma Center de niveau 1 agréé (HUG ou CHUV)",
            "Partager en flux continu et en temps réel les constantes vitales et l'échographie FAST avec la salle de déchocage hospitalière",
            "Implémenter des protocoles de réanimation de contrôle des dommages (damage control resuscitation) guidés par des algorithmes d'aide à la décision"
        ]
    },
    "clinical-deterioration-prediction": {
        "hidden": True,
        "title": "Prédiction de la Détérioration Clinique en Transit",
        "description": "Surveillance intelligente des patients critiques durant leur transport en ambulance ou hélicoptère.",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Activer des alertes de détérioration basées sur la tendance des constantes vitales multi-paramétriques",
            "Préparer des protocoles de réanimation avancée en lien avec le médecin régulateur du SMUR",
            "Ajuster la vitesse de transfert ou envisager un rendez-vous SMUR/Héli-SMUR si nécessaire"
        ]
    },
    "patient-pathway-optimization": {
        "hidden": True,
        "title": "Optimisation du Parcours Patient Transfrontalier",
        "description": "Planification du transfert des patients vers les structures de soins appropriées en optimisant les capacités des deux côtés de la frontière.",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Vérifier la disponibilité des lits spécialisés en temps réel en France (ROR) et en Suisse",
            "Fluidifier les démarches administratives douanières pour les ambulances de transfert",
            "Établir un protocole de retour à domicile ou de soins de suite de proximité"
        ]
    },
    "mci-victim-estimation": {
        "hidden": True,
        "title": "Estimation des Victimes en Situation de Catastrophe (MCI)",
        "description": "Évaluation rapide du nombre et de la gravité des victimes lors d'événements majeurs pour dimensionner la réponse.",
        "cluster": "Patient-centered prehospital critical care",
        "recommended_actions": [
            "Activer le Plan Blanc (FR) / Plan ORCA (CH) de manière coordonnée",
            "Utiliser des outils de tri connectés (smart glasses, bracelets IoT) pour un inventaire en temps réel",
            "Répartir les flux de victimes de manière équilibrée entre les hôpitaux de la région"
        ]
    },
    "environmental-risk-forecasting": {
        "hidden": True,
        "title": "Prévision des Risques Environnementaux",
        "description": "Anticipation des pics de pollution de l'air, d'ozone ou d'allergènes et de leur impact direct sur les urgences respiratoires.",
        "cluster": "Environmental & Disaster Risk Forecasting",
        "recommended_actions": [
            "Croiser les données d'AirGenève et d'Atmo Auvergne-Rhône-Alpes avec les appels pour asthme/BPCO",
            "Diffuser des messages de prévention ciblés aux patients vulnérables enregistrés",
            "Anticiper une hausse de 15% des appels pour détresse respiratoire dans les 48 heures"
        ]
    },
    "disaster-risk-assessment": {
        "hidden": True,
        "title": "Évaluation des Risques de Catastrophes Naturelles",
        "description": "Modélisation de l'impact sanitaire des inondations, séismes locaux, ou glissements de terrain sur les infrastructures EMS.",
        "cluster": "Environmental & Disaster Risk Forecasting",
        "recommended_actions": [
            "Identifier les casernes et voies d'accès ambulances situées en zone inondable (crues de l'Arve/Rhône)",
            "Établir des points de rassemblement des secours hors des zones à risque",
            "Simuler des scénarios de rupture d'alimentation électrique ou de télécommunications"
        ]
    },
    "heatwave-ems-impact": {
        "hidden": True,
        "title": "Impact des Canicules sur les EMS",
        "description": "Modélisation de l'impact des vagues de chaleur extrêmes sur la demande EMS et les pathologies liées à la chaleur (coup de chaleur, hyperthermie).",
        "cluster": "Environmental & Disaster Risk Forecasting",
        "recommended_actions": [
            "Anticiper une hausse de 20-40% des appels EMS lors des épisodes de canicule (UTCI > 38°C)",
            "Activer les protocoles de prise en charge préhospitalière des coups de chaleur",
            "Renforcer les équipages avec du matériel de refroidissement rapide (poches de glace, brumisateurs)"
        ]
    },
    "climate-impact-on-ems": {
        "hidden": True,
        "title": "Impact du Changement Climatique sur les EMS",
        "description": "Analyse à long terme et saisonnière de l'évolution des pathologies d'urgence liées au réchauffement climatique.",
        "cluster": "Environmental & Disaster Risk Forecasting",
        "recommended_actions": [
            "Adapter les plannings de garde estivaux pour faire face à des vagues de chaleur plus fréquentes",
            "Intégrer les projections climatiques de Copernicus dans le schéma directeur de santé transfrontalier",
            "Former le personnel aux pathologies émergentes (maladies à vecteur comme la dengue en Europe)"
        ]
    },
    "emergency-call-qualification": {
        "hidden": False,
        "title": "Qualification Automatisée des Appels d'Urgence",
        "description": "Outils de traitement du langage naturel (NLP) et de reconnaissance vocale en temps réel pour assister les assistants de régulation médicale (ARM) dans la transcription, la détection automatique de mots-clés cliniques et la qualification rapide des motifs d'appels d'urgence.",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Activer la transcription vocale continue à faible latence (Speech-to-Text) intégrée au système de téléphonie de régulation",
            "Utiliser des modèles NLP spécialisés (type CamemBERT médical) pour extraire automatiquement les entités cliniques et les symptômes clés",
            "Analyser les caractéristiques acoustiques et les bruits de fond de l'appel pour détecter la détresse respiratoire ou la panique",
            "Suggérer de manière adaptative et dynamique les questions de protocoles de régulation selon les premiers mots transcrits"
        ]
    },
    "call-prioritization": {
        "hidden": False,
        "title": "Priorisation des Appels de Régulation",
        "description": "Algorithmes d'apprentissage automatique pour le tri et la priorisation dynamique de la file d'attente des appels entrants en centrale de régulation médicale, garantissant une prise en charge immédiate des détresses vitales et minimisant le risque de sous-triage.",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Placer automatiquement en priorité absolue de file d'attente les appels identifiés comme suspicion d'arrêt cardiorespiratoire ou d'étouffement",
            "Ajuster dynamiquement les seuils de tri et les files d'attente lors de situations de saturation de la centrale (pics d'appels)",
            "Fournir aux régulateurs un tableau de bord prédictif du niveau de risque clinique estimé pour chaque appel en attente",
            "Mesurer en continu le taux d'adéquation de la priorisation pour minimiser le sous-triage sous la barre stricte de 5%"
        ]
    },
    "mass-casualty-triage": {
        "hidden": True,
        "title": "Tri en Situation de Nombreuses Victimes",
        "description": "Algorithmes d'aide au tri de masse sur le terrain pour classer rapidement les victimes (Urgence Absolue, Urgence Relative).",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Appliquer les critères de tri standardisés (START/SALT) via une interface mobile simplifiée",
            "Générer des codes QR uniques pour chaque victime afin de suivre leur parcours",
            "Visualiser la répartition des catégories de gravité sur la cartographie du PMA"
        ]
    },
    "undertriage-detection": {
        "hidden": True,
        "title": "Détection du Sous-Tri (Undertriage)",
        "description": "Algorithmes de contrôle qualité pour identifier les patients graves classés à tort en faible priorité.",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Analyser rétrospectivement les dossiers de régulation pour identifier les écarts de tri",
            "Alerter en temps réel si les constantes saisies contredisent le niveau de priorité attribué",
            "Ajuster les arbres de décision cliniques pour réduire le taux de sous-tri sous le seuil de 5%"
        ]
    },
    "dispatch-decision-support": {
        "hidden": False,
        "title": "Aide à la Décision de Dispatch",
        "description": "Systèmes experts et modèles prédictifs d'aide à la décision pour recommander instantanément le moyen de secours préhospitalier optimal (ambulance de soins d'urgence, équipe médicale SMUR, hélicoptère ou médecin généraliste de garde) en fonction de la gravité clinique suspectée et des ressources disponibles.",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Suggérer automatiquement l'envoi d'un SMUR transfrontalier (FR/CH) si son délai d'arrivée estimé est inférieur à la ressource nationale",
            "Intégrer les données de géolocalisation live (GPS) et le statut opérationnel des véhicules pour proposer la ressource la plus rapide",
            "Suggérer des alternatives de régulation libérale, de conseil médical ou de transport sanitaire non urgent pour les motifs de faible gravité",
            "Implémenter un modèle d'adéquation d'envoi pour réduire les envois inutiles d'équipes médicalisées (over-dispatch) tout en sécurisant les patients"
        ]
    },
    "triage-support": {
        "hidden": False,
        "title": "Support au Tri Clinique aux Urgences",
        "description": "Algorithmes de classification clinique pour assister le personnel infirmier d'accueil (IOA) dans la détermination rapide du niveau de gravité des patients aux urgences selon des échelles validées (Échelle Suisse de Tri, Échelle de Rouen), optimisant les délais d'accès aux soins.",
        "cluster": "Prehospital Emergency Triage & Risk Stratification",
        "recommended_actions": [
            "Calculer automatiquement le niveau de gravité clinique théorique en intégrant les constantes vitales et le motif de consultation saisi",
            "Prédire dès l'accueil le risque d'hospitalisation d'aval ou de passage en réanimation pour anticiper l'orientation des patients",
            "Générer des alertes visuelles et sonores immédiates pour l'infirmier d'accueil en cas d'anomalie physiologique majeure",
            "Mesurer la concordance inter-observateur (Kappa de Cohen) entre le tri assisté par IA et l'évaluation finale par le médecin"
        ]
    },
    "response-time-optimization": {
        "hidden": False,
        "title": "Optimisation des Temps de Réponse EMS",
        "description": "Modèles de routage prédictif intégrant les conditions de trafic en temps réel, la météorologie et la topologie urbaine pour guider les véhicules d'urgence par l'itinéraire le plus rapide et minimiser le délai d'accès aux soins critiques.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Calculer des itinéraires d'urgence dynamiques intégrant les données de congestion du trafic en temps réel et l'historique de circulation",
            "Interfacer le système de navigation des ambulances avec la gestion des feux tricolores (priorité de passage) sur les axes critiques",
            "Modéliser spécifiquement les délais de passage transfrontaliers (douanes du Grand Genève, ponts sur le lac) pour adapter les trajets",
            "Évaluer en continu la courbe d'efficacité temps-dépendante du temps de réponse réel sur la survie des détresses vitales"
        ]
    },
    "ambulance-dispatch-optimization": {
        "hidden": False,
        "title": "Optimisation de la Flotte d'Ambulances",
        "description": "Modèles mathématiques de couverture spatio-temporelle maximale (MCLP, DSM) pour la gestion et le repositionnement préventif et dynamique de la flotte d'ambulances, garantissant une couverture territoriale optimale en fonction des risques prédictifs.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Repositionner dynamiquement et de manière préventive les ambulances disponibles en attente pour combler les failles de couverture",
            "Prédire les micro-zones à haut risque d'appels d'urgence à l'échelle horaire pour y pré-positionner des équipages",
            "Coordonner de manière transparente sur une plateforme unique le dispatch des ambulances publiques, privées et associatives",
            "Suivre en temps réel le taux de couverture de la population cible à moins de 10 minutes d'une ambulance disponible"
        ]
    },
    "staffing-level-prediction": {
        "hidden": True,
        "title": "Prévision des Effectifs Requis",
        "description": "Modèles prédictifs pour dimensionner les équipes de régulation et les équipages d'ambulances selon la charge attendue.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Ajuster le nombre d'ARM de garde en fonction des prévisions de charge à 7 jours",
            "Planifier des renforts pour les périodes de grands événements (fêtes de Genève, manifestations)",
            "Prendre en compte les taux d'absentéisme saisonniers (pandémies hivernales du personnel)"
        ]
    },
    "hospital-capacity-forecasting": {
        "hidden": False,
        "title": "Prévision de la Capacité Hospitalière",
        "description": "Modèles prédictifs de séries temporelles pour anticiper la saturation des services d'urgences et l'occupation des lits de réanimation, de soins continus et d'hospitalisation conventionnelle (lits d'aval), facilitant la gestion proactive des flux de patients.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Prédire l'afflux de patients aux urgences et le taux d'occupation des lits à 24h et 48h (score NEDOCS prédictif)",
            "Coordonner en temps réel les sorties de patients hospitalisés et les transferts vers les unités de soins de suite et de réadaptation (SSR)",
            "Déclencher des alertes automatiques et des cellules de crise de gestion des lits (Bed Management) transfrontalières en cas de tension",
            "Modéliser l'impact de la saturation des urgences (overcrowding) sur les délais de libération et de transfert des ambulances (ambulance diversion)"
        ]
    },
    "demand-forecasting": {
        "hidden": False,
        "title": "Prévision de la Demande EMS",
        "description": "Modèles de prévision hybrides (Prophet, LightGBM, LSTM) intégrant les données météorologiques, le calendrier, les vacances scolaires et la surveillance épidémiologique pour estimer avec précision le volume d'appels d'urgence et dimensionner les équipes.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Intégrer des flux météorologiques locaux (Open-Meteo) et épidémiques (Réseau Sentinelles) en temps réel pour affiner les prévisions",
            "Visualiser la prévision de la demande EMS à l'échelle horaire par secteur géographique sur un horizon de J+1 à J+7",
            "Alerter automatiquement les cadres opérationnels en cas d'écart significatif (> 15%) entre le volume réel d'appels et la prévision de base",
            "Utiliser les prévisions de demande pour adapter dynamiquement la planification des gardes et le nombre de véhicules opérationnels"
        ]
    },
    "resource-allocation": {
        "hidden": True,
        "title": "Allocation Optimisée des Ressources",
        "description": "Distribution des moyens humains et matériels de manière à maximiser l'efficacité de la réponse d'urgence.",
        "cluster": "Demand Forecasting, Response Time & Resource Management",
        "recommended_actions": [
            "Allouer les ambulances de réanimation (SMUR) prioritairement aux urgences vitales",
            "Optimiser la répartition des stocks de matériel d'urgence entre sites",
            "Suivre en temps réel le statut d'activité de chaque équipage"
        ]
    },
    "epidemic-early-warning": {
        "hidden": True,
        "title": "Alerte Précoce Épidémique",
        "description": "Détection précoce des signaux faibles épidémiques à partir des motifs d'appels de régulation médicale.",
        "cluster": "Surveillance & Epidemic Management",
        "recommended_actions": [
            "Surveiller l'évolution des appels pour syndrome grippal, gastro-entérite ou détresse respiratoire",
            "Déclencher une alerte si un seuil d'incidence statistique est dépassé dans un district",
            "Partager les alertes précoces avec les autorités sanitaires (OFSP, ARS) pour action coordonnée"
        ]
    },
    "surveillance": {
        "hidden": True,
        "title": "Surveillance Syndromique Active",
        "description": "Suivi continu des indicateurs de santé de la population pour identifier des anomalies ou des clusters inhabituels.",
        "cluster": "Surveillance & Epidemic Management",
        "recommended_actions": [
            "Analyser les données de passage aux urgences (SOS Médecins, hôpitaux) en temps réel",
            "Identifier géographiquement des regroupements anormaux de cas présentant des symptômes similaires",
            "Adapter les seuils de détection en fonction de la saisonnalité et du contexte local"
        ]
    },
    "surge-management": {
        "hidden": True,
        "title": "Gestion des Pics d'Afflux (Surge)",
        "description": "Stratégies opérationnelles pour faire face à une hausse soudaine et massive de la demande de soins d'urgence.",
        "cluster": "Surveillance & Epidemic Management",
        "recommended_actions": [
            "Activer des lignes de régulation médicale supplémentaires au Centre 15/144",
            "Mettre en place des structures d'accueil temporaires (tentes de tri) devant les urgences",
            "Reporter les hospitalisations non urgentes (programmées) pour libérer des capacités"
        ]
    },
    "pandemic-preparedness": {
        "hidden": True,
        "title": "Préparation aux Pandémies",
        "description": "Planification stratégique et modélisation à long terme pour renforcer la résilience du système de santé face à des crises globales.",
        "cluster": "Surveillance & Epidemic Management",
        "recommended_actions": [
            "Établir des plans de continuité d'activité (PCA) pour les services d'urgence et de régulation",
            "Dimensionner les stocks stratégiques de contre-mesures médicales (masques, antiviraux, vaccins)",
            "Organiser des exercices de simulation de crise pandémique à l'échelle transfrontalière"
        ]
    },
    "cross-border-coordination": {
        "hidden": True,
        "title": "Coordination Sanitaire Transfrontalière",
        "description": "Protocoles et outils de communication pour harmoniser la réponse d'urgence entre la France et la Suisse (Grand Genève).",
        "cluster": "Cross-border & Operational Coordination",
        "recommended_actions": [
            "Interconnecter les systèmes de régulation TECHWAN SAGA (France) et l'équivalent suisse",
            "Établir des conventions de libre passage des ambulances et hélicoptères de secours",
            "Organiser des réunions de coordination régulières entre les directions des HUG, du CHUV et des SAMU"
        ]
    },
    "situational-awareness": {
        "hidden": True,
        "title": "Conscience Situationnelle Opérationnelle",
        "description": "Tableau de bord en temps réel intégrant toutes les sources de données pour une vue unifiée de la situation d'urgence.",
        "cluster": "Cross-border & Operational Coordination",
        "recommended_actions": [
            "Afficher en temps réel la position de toutes les unités mobiles (ambulances, SMUR, hélicoptères)",
            "Intégrer les flux météo, épidémiques et de trafic dans une carte opérationnelle unifiée",
            "Partager la vue opérationnelle avec les partenaires transfrontaliers en temps réel"
        ]
    },
    "unassigned": {
        "hidden": True,
        "title": "Scénarios Non Classés",
        "description": "Documents en attente de classification dans un scénario spécifique.",
        "cluster": "Non classé",
        "recommended_actions": [
            "Relancer le script de backfill pour réassigner ces documents",
            "Examiner manuellement les titres et résumés pour une classification manuelle"
        ]
    }
}


GESICA_FOLDER_ID = "fld-gesica-main"


def _gesica_title(meta: dict[str, Any]) -> str:
    return str(meta.get("title") or meta.get("name") or meta.get("id") or "Scénario")


def _gesica_actions(meta: dict[str, Any]) -> list[str]:
    actions = meta.get("recommended_actions")
    if actions is None:
        actions = meta.get("recommended_action")
    if isinstance(actions, list):
        return actions
    if isinstance(actions, str) and actions.strip():
        return [actions]
    return []


def _get_db_gesica_scenario_or_404(scenario_id: str, conn=None) -> dict[str, Any]:
    sql = text("""
        SELECT *
        FROM user_scenarios
        WHERE id = :sid
          AND is_system = TRUE
          AND folder_id = :folder_id
    """)
    params = {"sid": scenario_id, "folder_id": GESICA_FOLDER_ID}
    if conn is None:
        with engine.connect() as _conn:
            row = _conn.execute(sql, params).mappings().first()
    else:
        row = conn.execute(sql, params).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Scénario '{scenario_id}' non trouvé")
    return dict(row)


def _list_db_gesica_scenarios(conn) -> list[dict[str, Any]]:
    rows = conn.execute(text("""
        SELECT *
        FROM user_scenarios
        WHERE is_system = TRUE
          AND folder_id = :folder_id
          AND id <> 'unassigned'
          AND COALESCE(hidden, FALSE) = FALSE
        ORDER BY COALESCE(title, name, id) ASC
    """), {"folder_id": GESICA_FOLDER_ID}).mappings().all()
    return [dict(r) for r in rows]


@app.get("/gesica/scenarios")
def get_gesica_scenarios(lang: str | None = Query(None)) -> list[dict[str, Any]]:
    """
    Scénarios GESICA dynamiques : retourne les scénarios système stockés en base,
    enrichis avec les articles scientifiques associés depuis la DB (living evidence review).
    Les scénarios sont triés par nombre d'articles décroissant, puis alphabétiquement.
    `lang=en` rend le catalogue (titre, description, actions) en anglais.
    """
    with engine.connect() as conn:
        scenario_rows = _list_db_gesica_scenarios(conn)
        # Compteurs limités AUX scénarios listés : article_scenarios contient aussi les
        # liens de tous les scénarios utilisateur (des centaines de milliers de lignes
        # quand une recherche a « matché » large), qui n'ont rien à faire ici.
        sys_ids = [str(m["id"]) for m in scenario_rows]

        sql_counts = text("""
            SELECT ars.scenario_id, COUNT(DISTINCT ars.document_id) as article_count
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = ANY(CAST(:ids AS text[]))
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            GROUP BY ars.scenario_id;
        """)
        db_counts = {row["scenario_id"]: row["article_count"]
                     for row in conn.execute(sql_counts, {"ids": sys_ids}).mappings().all()}

        sql_screening = text("""
            SELECT
                ars.scenario_id,
                COUNT(CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'included' THEN 1 END) as included_count,
                COUNT(CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'excluded' THEN 1 END) as excluded_count
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = ANY(CAST(:ids AS text[]))
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            GROUP BY ars.scenario_id;
        """)
        screening_counts = {
            row["scenario_id"]: {"included": row["included_count"], "excluded": row["excluded_count"]}
            for row in conn.execute(sql_screening, {"ids": sys_ids}).mappings().all()
        }

        # Kappa par scénario : il n'existe pas de table `scenario_kappa_cache`
        # (aucun écrivain). Le kappa live est servi par l'endpoint dédié
        # /double-blind/kappa ; ici on ne fournit pas de valeur agrégée.
        kappa_scores: dict[str, Any] = {}

        # `relevant_articles` est VIDE, à dessein. Cette liste chargeait autrefois TOUS
        # les articles de TOUS les scénarios (y compris les scénarios utilisateur, non
        # listés ici) avec leur résumé, en UNE requête : un seul scénario dont la
        # recherche lexicale avait apparié 238 000 documents suffisait à faire dépasser
        # la minute - et la mémoire - à cette route, que l'interface appelle au
        # chargement de CHAQUE page. Réponse observée en production : 502 sur
        # /api/gesica/scenarios, « Failed to load scenarios ». Le front n'affiche
        # d'ailleurs pas cette liste (App.tsx : `false && scenario.relevantArticles…`) ;
        # les articles d'un scénario se lisent, paginés, sur /gesica/scenarios/{id}/corpus.
        result = []
        for meta in scenario_rows:
            meta = _localize_gesica(meta, lang)
            scenario_id = str(meta["id"])
            title = _gesica_title(meta)
            article_count = int(db_counts.get(scenario_id, 0) or 0)
            sc = screening_counts.get(scenario_id, {"included": 0, "excluded": 0})

            articles: list[dict[str, Any]] = []

            result.append({
                "id": scenario_id,
                "name": title,
                "title": title,
                "label_short": meta.get("label_short"),
                "description": meta.get("description") or "",
                "cluster": meta.get("cluster") or "",
                "article_count": article_count,
                "included_count": int(sc["included"] or 0),
                "excluded_count": int(sc["excluded"] or 0),
                "kappa_score": kappa_scores.get(scenario_id),
                "hidden": bool(meta.get("hidden", False)),
                "recommended_actions": _gesica_actions(meta),
                "relevant_articles": articles,
                "living_evidence_note": (
                    _msg(lang,
                         f"Living Evidence Review · {article_count} articles indexés. Mis à jour automatiquement à chaque ingestion.",
                         f"Living Evidence Review · {article_count} articles indexed. Updated automatically with every ingestion.")
                    if article_count > 0
                    else _msg(lang,
                              "Aucun article indexé pour ce scénario. En attente d'ingestion de nouvelles sources.",
                              "No article indexed for this scenario yet. Waiting for new sources to be ingested.")
                )
            })

        result.sort(key=lambda x: (-x["article_count"], x["title"]))
    return result


# ─── Endpoint : scénarios multiples d'un article ────────────────────────────
@app.get("/documents/{doc_id}/scenarios")
def get_document_scenarios(doc_id: int) -> dict[str, Any]:
    """
    Retourne tous les scénarios auxquels un article est assigné (relation N:N).
    Utile pour l'affichage multi-scénario dans l'interface.
    """
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ars.scenario_id, ars.similarity_score, ars.assigned_at
            FROM article_scenarios ars
            WHERE ars.document_id = :doc_id
            ORDER BY ars.similarity_score DESC NULLS LAST
        """), {"doc_id": doc_id}).mappings().all()
    scenarios = []
    # Charger les titres GESICA depuis la DB
    with engine.connect() as _conn:
        _gesica_rows = _conn.execute(text("""
            SELECT id, title, name FROM user_scenarios
            WHERE is_system = TRUE AND folder_id = :fid
        """), {"fid": GESICA_FOLDER_ID}).mappings().all()
    _gesica_title_map = {str(r["id"]): (r.get("title") or r.get("name") or str(r["id"])) for r in _gesica_rows}
    for r in rows:
        sid = r["scenario_id"]
        scenarios.append({
            "scenario_id": sid,
            "title": _gesica_title_map.get(sid, sid),
            "similarity_score": float(r["similarity_score"]) if r["similarity_score"] else None,
            "assigned_at": r["assigned_at"].isoformat() if r["assigned_at"] else None,
        })
    return {"document_id": doc_id, "scenarios": scenarios, "count": len(scenarios)}


SCENARIO_LIVING_REVIEW_IDS = [
    # Cluster 1 : Patient-centered prehospital critical care
    "cardiac-arrest-prediction",
    "stroke-detection",
    "trauma-severity-assessment",
    "clinical-deterioration-prediction",
    "patient-pathway-optimization",
    "mci-victim-estimation",
    # Cluster 2 : Environmental & Disaster Risk
    "environmental-risk-forecasting",
    "disaster-risk-assessment",
    "climate-impact-on-ems",
    # Cluster 3 : Prehospital Triage & Risk Stratification
    "emergency-call-qualification",
    "call-prioritization",
    "mass-casualty-triage",
    "undertriage-detection",
    "dispatch-decision-support",
    "triage-support",
    # Cluster 4 : EMS Operations & Resource Management
    "response-time-optimization",
    "ambulance-dispatch-optimization",
    "staffing-level-prediction",
    "hospital-capacity-forecasting",
    "demand-forecasting",
    "resource-allocation",
    # Cluster 5 : Epidemiological & Strategic Surveillance
    "epidemic-early-warning",
    "surveillance",
    "surge-management",
    "pandemic-preparedness",
    "cross-border-coordination",
    "situational-awareness",
]

# ─────────────────────────────────────────────────────────────────────────────
# GESICA Scenario Detail Endpoints (Phase 2 : Refonte interface)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/gesica/scenarios/{scenario_id}/detail")
def get_scenario_detail(scenario_id: str, lang: str | None = Query(None)) -> dict[str, Any]:
    """
    Retourne toutes les informations enrichies d'un scénario :
    - Métadonnées de base (titre, description, cluster, actions recommandées)
    - Queries booléennes PubMed et requêtes NL pour la recherche sémantique
    - Prompt d'extraction d'évidence spécifique au scénario
    - Informations sur le modèle IA (algorithme, variables, fréquence de mise à jour)
    - Seuils d'alerte vert/orange/rouge
    `lang=en` rend le titre, la description et les actions du catalogue en anglais.
    """
    with engine.connect() as conn:
        meta = _localize_gesica(_get_db_gesica_scenario_or_404(scenario_id, conn), lang)
        stats = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                ) THEN 1 ELSE 0 END) AS with_fulltext,
                COUNT(DISTINCT d.year) AS years_covered,
                COUNT(DISTINCT d.journal) AS journals_count,
                MIN(d.year) FILTER (WHERE d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_min,
                MAX(d.year) FILTER (WHERE d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_max
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
        """), {"sid": scenario_id}).mappings().first()
    return {
        "id": scenario_id,
        "title": _gesica_title(meta),
        "description": meta.get("description") or "",
        "cluster": meta.get("cluster") or "",
        "recommended_actions": _gesica_actions(meta),
        "boolean_queries": meta.get("boolean_queries") or [],
        "nl_queries": meta.get("nl_queries") or [],
        "evidence_extraction_prompt": meta.get("evidence_extraction_prompt") or "",
        "model_info": meta.get("model_info") or {},
        "alert_thresholds": meta.get("alert_thresholds") or {},
        "databases": meta.get("required_databases") or [],
        "outcome_definition": meta.get("outcome_definition") or "",
        "variables_detail": meta.get("variables_detail") or {},
        "keywords": meta.get("keywords") or [],
        "clinical_rationale": meta.get("clinical_rationale") or "",
        "corpus_stats": {
            "total": int(stats["total"] or 0),
            "with_fulltext": int(stats["with_fulltext"] or 0),
            "years_covered": int(stats["years_covered"] or 0),
            "journals_count": int(stats["journals_count"] or 0),
            "year_min": stats["year_min"],
            "year_max": stats["year_max"],
        },
    }


@app.get("/gesica/scenarios/{scenario_id}/corpus")
def get_scenario_corpus(
    scenario_id: str,
    limit: int = 100000,
    offset: int = 0,
    year_from: int | None = None,
    year_to: int | None = None,
    fulltext_only: bool = False,
    source: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    from .scenarios import get_user_scenario_corpus  # lazy: scenarios is loaded after this module
    return get_user_scenario_corpus(scenario_id, limit, offset, year_from, year_to, fulltext_only, source, threshold)


def _get_scenario_name(scenario_id: str) -> str:
    """Retourne le nom lisible d'un scénario (user ou GESICA)."""
    if scenario_id.startswith("usr-"):
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT name FROM user_scenarios WHERE id = :id"
            ), {"id": scenario_id}).mappings().first()
        return row["name"] if row else scenario_id
    # GESICA : utiliser la DB
    try:
        meta = _get_db_gesica_scenario_or_404(scenario_id)
        return _gesica_title(meta)
    except Exception:
        return scenario_id


# ─── HEATMAP AVEC VRAIS NOMS ─────────────────────────────────────────────────

@app.get("/corpus/stats/by-year/named")
def get_corpus_stats_by_year_named() -> dict[str, Any]:
    """
    Comme /corpus/stats/by-year mais avec les vrais noms des scénarios
    (GESICA et user_scenarios) dans la heatmap.
    """
    with engine.connect() as conn:
        # Articles par année (1800 → année courante)
        rows_year = conn.execute(text("""
            SELECT year, COUNT(*) as count
            FROM literature_document
            WHERE year >= 1800 AND year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
            GROUP BY year ORDER BY year ASC
        """)).mappings().all()

        # Articles par scénario ET par source (heatmap) + nombre en texte intégral.
        rows_heatmap = conn.execute(text("""
            SELECT ars.scenario_id, d.source,
                   COUNT(*) AS count,
                   COUNT(*) FILTER (WHERE EXISTS (
                       SELECT 1 FROM document_chunk c
                       WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                   )) AS fulltext
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE d.is_duplicate IS NOT TRUE
            GROUP BY ars.scenario_id, d.source
            ORDER BY ars.scenario_id, count DESC
        """)).mappings().all()

        # Articles par scénario ET par année
        rows_scenario_year = conn.execute(text("""
            SELECT d.year, ars.scenario_id, COUNT(*) as count
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE d.year >= 1800 AND d.year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
            GROUP BY d.year, ars.scenario_id ORDER BY d.year ASC
        """)).mappings().all()

        # Noms des user_scenarios (only non-deleted ones)
        user_names = conn.execute(text("""
            SELECT id, name FROM user_scenarios
        """)).mappings().all()

    user_name_map = {r["id"]: r["name"] for r in user_names}
    # Valid GESICA scenarios (not hidden) - depuis la DB
    with engine.connect() as _hm_conn:
        _gesica_db_rows = _list_db_gesica_scenarios(_hm_conn)
    _gesica_name_map = {str(r["id"]): _gesica_title(r) for r in _gesica_db_rows}
    valid_gesica_ids = set(_gesica_name_map.keys())
    # All valid scenario IDs: existing user scenarios + non-hidden GESICA ones
    valid_sids = set(user_name_map.keys()) | valid_gesica_ids

    def _resolve_name(sid: str) -> str | None:
        if sid in user_name_map:
            return user_name_map[sid]
        if sid in _gesica_name_map:
            return _gesica_name_map[sid]
        return None  # deleted or hidden - exclude from heatmap

    by_year = {str(r["year"]): r["count"] for r in rows_year}

    # Clé = scenario_id (stable, évite la fusion de scénarios homonymes qui
    # faisait apparaître MOINS de scénarios qu'en réalité). On porte le nom et,
    # par source CANONIQUE, le total ET le nombre en texte intégral.
    heatmap: dict[str, dict] = {}
    for r in rows_heatmap:
        if r["scenario_id"] not in valid_sids:
            continue
        name = _resolve_name(r["scenario_id"])
        if not name:
            continue
        src = _canonical_source(r["source"])
        entry = heatmap.setdefault(str(r["scenario_id"]), {"name": name, "sources": {}})
        cell = entry["sources"].setdefault(src, {"total": 0, "fulltext": 0})
        cell["total"] += int(r["count"] or 0)
        cell["fulltext"] += int(r["fulltext"] or 0)

    scenario_year: dict[str, dict[str, int]] = {}
    for r in rows_scenario_year:
        if r["scenario_id"] not in valid_sids:
            continue
        name = _resolve_name(r["scenario_id"])
        if not name:
            continue
        yr = str(r["year"])
        if name not in scenario_year:
            scenario_year[name] = {}
        scenario_year[name][yr] = scenario_year[name].get(yr, 0) + r["count"]

    return {
        "by_year": by_year,
        "scenario_by_year": scenario_year,
        "heatmap_scenario_source": heatmap,
    }
