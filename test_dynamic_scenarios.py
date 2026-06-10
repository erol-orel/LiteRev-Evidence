import os
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DB_URL (or DATABASE_URL) environment variable is required")
engine = create_engine(DB_URL)

# Mapping des identifiants de scénarios vers des titres et descriptions en français
SCENARIO_METADATA = {
    "cardiac-arrest-prediction": {
        "title": "Prédiction de l'Arrêt Cardiaque Extra-Hospitalier (OHCA)",
        "description": "Modèles de prédiction et d'identification précoce des arrêts cardiorespiratoires pour optimiser la chaîne de survie.",
        "recommended_actions": [
            "Déployer des algorithmes de détection acoustique de l'agonie respiratoire (gasping) au Centre 15/144",
            "Optimiser le dispatch des premiers répondants équipés de DEA via l'application locale",
            "Ajuster le positionnement des SMUR en fonction des zones à forte probabilité d'OHCA"
        ]
    },
    "stroke-detection": {
        "title": "Détection Préhospitalière de l'AVC",
        "description": "Outils d'aide à la décision pour identifier les AVC (ischémiques vs hémorragiques) et orienter vers la bonne filière (thrombolyse/thrombectomie).",
        "recommended_actions": [
            "Intégrer des scores cliniques préhospitaliers automatisés dans le dossier patient embarqué",
            "Orienter directement vers l'unité de soins intensifs neurovasculaires (UNV) des HUG ou du CHUV",
            "Pré-alerter l'équipe d'angioradiologie dès la confirmation de suspicion d'occlusion de gros vaisseau (LVO)"
        ]
    },
    "trauma-severity-assessment": {
        "title": "Évaluation de la Gravité des Traumatismes",
        "description": "Stratification du risque pour les traumatisés graves (accidents de la route, chutes) afin d'orienter vers les trauma centers adaptés.",
        "recommended_actions": [
            "Utiliser des modèles prédictifs de transfusion massive dès la prise en charge terrain",
            "Orienter les traumatismes sévères vers le Trauma Center de niveau 1 (HUG ou CHUV)",
            "Partager en temps réel les constantes vitales (score de Glasgow, pression artérielle) avec la salle de déchocage"
        ]
    },
    "clinical-deterioration-prediction": {
        "title": "Prédiction de la Détérioration Clinique en Transit",
        "description": "Surveillance intelligente des patients critiques durant leur transport en ambulance ou hélicoptère pour anticiper les défaillances d'organes.",
        "recommended_actions": [
            "Activer des alertes de détérioration basées sur la tendance des constantes vitales multi-paramétriques",
            "Préparer des protocoles de réanimation avancée en lien avec le médecin régulateur du SMUR",
            "Ajuster la vitesse de transfert ou envisager un rendez-vous SMUR/Héli-SMUR si nécessaire"
        ]
    },
    "patient-pathway-optimization": {
        "title": "Optimisation du Parcours Patient Transfrontalier",
        "description": "Planification du transfert des patients vers les structures de soins appropriées en optimisant les capacités des deux côtés de la frontière.",
        "recommended_actions": [
            "Vérifier la disponibilité des lits spécialisés en temps réel en France (ROR) et en Suisse",
            "Fluidifier les démarches administratives douanières pour les ambulances de transfert",
            "Établir un protocole de retour à domicile ou de soins de suite de proximité après la phase aiguë"
        ]
    },
    "mci-victim-estimation": {
        "title": "Estimation des Victimes en Situation de Catastrophe (MCI)",
        "description": "Évaluation rapide du nombre et de la gravité des victimes lors d'événements majeurs (accidents collectifs, attentats) pour dimensionner la réponse.",
        "recommended_actions": [
            "Activer le Plan Blanc (FR) / Plan ORCA (CH) de manière coordonnée",
            "Utiliser des outils de tri connectés (smart glasses, bracelets IoT) pour un inventaire en temps réel",
            "Répartir les flux de victimes de manière équilibrée entre les hôpitaux de la région transfrontalière"
        ]
    },
    "environmental-risk-forecasting": {
        "title": "Prévision des Risques Environnementaux",
        "description": "Anticipation des pics de pollution de l'air, d'ozone ou d'allergènes et de leur impact direct sur les urgences respiratoires.",
        "recommended_actions": [
            "Croiser les données d'AirGenève et d'Atmo Auvergne-Rhône-Alpes avec les appels pour asthme/BPCO",
            "Diffuser des messages de prévention ciblés aux patients vulnérables enregistrés",
            "Anticiper une hausse de 15% des appels pour détresse respiratoire dans les 48 heures"
        ]
    },
    "disaster-risk-assessment": {
        "title": "Évaluation des Risques de Catastrophes Naturelles",
        "description": "Modélisation de l'impact sanitaire des inondations, séismes locaux, ou glissements de terrain sur les infrastructures EMS.",
        "recommended_actions": [
            "Identifier les casernes et voies d'accès ambulances situées en zone inondable (crues de l'Arve/Rhône)",
            "Établir des points de rassemblement des secours hors des zones à risque",
            "Simuler des scénarios de rupture d'alimentation électrique ou de télécommunications"
        ]
    },
    "climate-impact-on-ems": {
        "title": "Impact du Changement Climatique sur les EMS",
        "description": "Analyse à long terme et saisonnière de l'évolution des pathologies d'urgence liées au réchauffement climatique.",
        "recommended_actions": [
            "Adapter les plannings de garde estivaux pour faire face à des vagues de chaleur plus fréquentes et plus longues",
            "Intégrer les projections climatiques de Copernicus dans le schéma directeur de santé transfrontalier",
            "Former le personnel aux pathologies émergentes (maladies à vecteur comme la dengue en Europe)"
        ]
    },
    "emergency-call-qualification": {
        "title": "Qualification Automatisée des Appels d'Urgence",
        "description": "Analyse sémantique et acoustique des appels au Centre 15/144 pour assister l'assistant de régulation médicale (ARM).",
        "recommended_actions": [
            "Activer la transcription vocale en temps réel avec détection des mots-clés critiques (douleur thoracique, paralysie)",
            "Analyser les bruits de fond et les signaux acoustiques pour détecter le stress ou l'inconscience",
            "Suggérer des protocoles de questionnement adaptés au profil de l'appelant"
        ]
    },
    "call-prioritization": {
        "title": "Priorisation des Appels de Régulation",
        "description": "Algorithmes de tri pour classer les appels d'urgence par niveau de gravité et réduire le temps d'attente des cas critiques.",
        "recommended_actions": [
            "Placer automatiquement en tête de file les appels suspects d'arrêt cardiaque ou d'obstruction des voies aériennes",
            "Ajuster dynamiquement les seuils de priorisation en période de forte surcharge du centre d'appels",
            "Fournir un tableau de bord visuel des appels en attente avec un score de risque estimé"
        ]
    },
    "mass-casualty-triage": {
        "title": "Tri en Situation de Nombreuses Victimes",
        "description": "Algorithmes d'aide au tri de masse sur le terrain pour classer rapidement les victimes (Urgence Absolue, Urgence Relative).",
        "recommended_actions": [
            "Appliquer les critères de tri standardisés (START/SALT) via une interface mobile simplifiée",
            "Générer des codes QR uniques pour chaque victime afin de suivre leur parcours de l'évacuation à l'admission",
            "Visualiser la répartition des catégories de gravité sur la cartographie du poste médical avancé (PMA)"
        ]
    },
    "undertriage-detection": {
        "title": "Détection du Sous-Tri (Undertriage)",
        "description": "Algorithmes de contrôle qualité pour identifier les patients graves classés à tort en faible priorité.",
        "recommended_actions": [
            "Analyser rétrospectivement les dossiers de régulation pour identifier les écarts de tri",
            "Alerter en temps réel si les constantes saisies contredisent le niveau de priorité attribué",
            "Ajuster les arbres de décision cliniques pour réduire le taux de sous-tri sous le seuil de 5%"
        ]
    },
    "dispatch-decision-support": {
        "title": "Aide à la Décision de Dispatch",
        "description": "Recommandation du moyen de secours le plus adapté (VSAV, SMUR, hélicoptère, médecin généraliste) selon le motif d'appel.",
        "recommended_actions": [
            "Suggérer l'envoi d'un SMUR transfrontalier si le temps de trajet est inférieur à celui du SMUR national",
            "Prendre en compte la disponibilité et la spécialisation des équipes de garde",
            "Proposer une régulation libérale ou un conseil médical pour les motifs non urgents"
        ]
    },
    "triage-support": {
        "title": "Support au Tri Clinique aux Urgences",
        "description": "Systèmes d'aide à la décision pour orienter et prioriser les patients dès leur arrivée dans le service des urgences.",
        "recommended_actions": [
            "Calculer automatiquement le score d'orientation (ex: French Emergency Nurses Association ou suisse)",
            "Estimer le risque de réadmission ou d'hospitalisation dès l'accueil",
            "Alerter l'infirmier organisateur d'accueil (IOA) en cas de constantes vitales anormales"
        ]
    },
    "response-time-optimization": {
        "title": "Optimisation des Temps de Réponse EMS",
        "description": "Algorithmes de routage dynamique et de prépositionnement pour réduire le délai d'arrivée des secours sur les lieux.",
        "recommended_actions": [
            "Utiliser les données de trafic en temps réel (HERE/OSRM) pour calculer l'itinéraire le plus rapide",
            "Activer la priorité aux feux tricolores pour les véhicules d'urgence sur les axes majeurs",
            "Analyser les goulets d'étranglement transfrontaliers (douanes, ponts) pour adapter les trajets"
        ]
    },
    "ambulance-dispatch-optimization": {
        "title": "Optimisation de la Flotte d'Ambulances",
        "description": "Gestion dynamique de la couverture opérationnelle en déplaçant préventivement des ambulances vers les zones à risque.",
        "recommended_actions": [
            "Repositionner temporairement une ambulance si une zone se retrouve sans couverture",
            "Prédire les pics de demande par secteur géographique pour y pré-positionner des moyens",
            "Coordonner le dispatch des ambulances privées et publiques sur une plateforme unique"
        ]
    },
    "staffing-level-prediction": {
        "title": "Prévision des Effectifs Requis",
        "description": "Modèles prédictifs pour dimensionner les équipes de régulation et les équipages d'ambulances selon la charge attendue.",
        "recommended_actions": [
            "Ajuster le nombre d'ARM de garde en fonction des prévisions de charge à 7 jours",
            "Planifier des renforts pour les périodes de grands événements (fêtes de Genève, manifestations)",
            "Prendre en compte les taux d'absentéisme saisonniers (épidémies hivernales du personnel)"
        ]
    },
    "hospital-capacity-forecasting": {
        "title": "Prévision de la Capacité Hospitalière",
        "description": "Anticipation de la saturation des lits de réanimation, de soins continus et d'hospitalisation conventionnelle.",
        "recommended_actions": [
            "Prédire le taux d'occupation des lits à 24h/48h pour anticiper les tensions",
            "Coordonner les sorties d'hospitalisation et les transferts vers les soins de suite (SSR)",
            "Déclencher des cellules de crise de gestion des lits (Bed Management) transfrontalières"
        ]
    },
    "demand-forecasting": {
        "title": "Prévision de la Demande EMS",
        "description": "Modèles de séries temporelles et de machine learning pour prévoir le volume d'appels d'urgence à court et moyen terme.",
        "recommended_actions": [
            "Intégrer les prévisions météo et épidémiques dans les modèles de prévision de charge",
            "Visualiser les tendances d'appels par tranche horaire et par motif d'appel",
            "Alerter si le volume d'appels réel s'écarte significativement de la prévision de base"
        ]
    },
    "resource-allocation": {
        "title": "Allocation Optimisée des Ressources",
        "description": "Distribution des moyens humains et matériels de manière à maximiser l'efficacité de la réponse d'urgence.",
        "recommended_actions": [
            "Allouer les ambulances de réanimation (SMUR) prioritairement aux urgences vitales",
            "Optimiser la répartition des stocks de matériel d'urgence (ventilateurs, consommables) entre sites",
            "Suivre en temps réel le statut d'activité de chaque équipage (disponible, en route, sur les lieux, à l'hôpital)"
        ]
    },
    "epidemic-early-warning": {
        "title": "Alerte Précoce Épidémique",
        "description": "Détection précoce des signaux faibles épidémiques à partir des motifs d'appels de régulation médicale.",
        "recommended_actions": [
            "Surveiller l'évolution des appels pour syndrome grippal, gastro-entérite ou détresse respiratoire",
            "Déclencher une alerte si un seuil d'incidence statistique est dépassé dans un district",
            "Partager les alertes précoces avec les autorités sanitaires (OFSP, ARS) pour action coordonnée"
        ]
    },
    "surveillance": {
        "title": "Surveillance Syndromique Active",
        "description": "Suivi continu des indicateurs de santé de la population pour identifier des anomalies ou des clusters inhabituels.",
        "recommended_actions": [
            "Analyser les données de passage aux urgences (SOS Médecins, hôpitaux) en temps réel",
            "Identifier géographiquement des regroupements anormaux de cas présentant des symptômes similaires",
            "Adapter les seuils de détection en fonction de la saisonnalité et du contexte local"
        ]
    },
    "surge-management": {
        "title": "Gestion des Pics d'Afflux (Surge)",
        "description": "Stratégies opérationnelles pour faire face à une hausse soudaine et massive de la demande de soins d'urgence.",
        "recommended_actions": [
            "Activer des lignes de régulation médicale supplémentaires au Centre 15/144",
            "Mettre en place des structures d'accueil temporaires (tentes de tri) devant les urgences",
            "Reporter les hospitalisations non urgentes (programmées) pour libérer des capacités"
        ]
    },
    "pandemic-preparedness": {
        "title": "Préparation aux Pandémies",
        "description": "Planification stratégique et modélisation à long terme pour renforcer la résilience du système de santé face à des crises globales.",
        "recommended_actions": [
            "Établir des plans de continuité d'activité (PCA) pour les services d'urgence et de régulation",
            "Dimensionner les stocks stratégiques de contre-mesures médicales (masques, antiviraux, vaccins)",
            "Organiser des exercices de simulation de crise pandémique à l'échelle transfrontalière"
        ]
    },
    "cross-border-coordination": {
        "title": "Coordination Sanitaire Transfrontalière",
        "description": "Protocoles et outils de communication pour harmoniser la réponse d'urgence entre la France et la Suisse (Grand Genève).",
        "recommended_actions": [
            "Interconnecter les systèmes de régulation TECHWAN SAGA (France) et l'équivalent suisse",
            "Établir des conventions de libre passage des ambulances et hélicoptères de secours",
            "Organiser des réunions de coordination régulières entre les directions des HUG, du CHUV et des SAMU limitrophes"
        ]
    }
}

def test_scenarios():
    with engine.connect() as conn:
        # Récupérer tous les scénarios uniques présents dans la DB pour GESICA
        sql_scenarios = text("""
            SELECT DISTINCT scenario_type 
            FROM literature_document 
            WHERE project_context = 'gesica' AND scenario_type IS NOT NULL AND scenario_type != 'unassigned'
            ORDER BY scenario_type;
        """)
        db_scenarios = [r[0] for r in conn.execute(sql_scenarios).fetchall()]
        print(f"Scénarios uniques trouvés dans la DB ({len(db_scenarios)}) :")
        for s in db_scenarios:
            meta = SCENARIO_METADATA.get(s, {
                "title": s.replace("-", " ").title(),
                "description": f"Scénario d'urgence de type {s}.",
                "recommended_actions": ["Consulter la littérature scientifique pour des recommandations spécifiques."]
            })
            
            # Compter les articles associés
            sql_count = text("""
                SELECT COUNT(*) FROM literature_document 
                WHERE project_context = 'gesica' AND scenario_type = :scenario
            """)
            count = conn.execute(sql_count, {"scenario": s}).scalar()
            
            # Récupérer les 3 articles les plus pertinents
            sql_articles = text("""
                SELECT id, title, abstract, year, source, authors
                FROM literature_document
                WHERE project_context = 'gesica' AND scenario_type = :scenario
                ORDER BY year DESC, title ASC
                LIMIT 3
            """)
            articles = [dict(r) for r in conn.execute(sql_articles, {"scenario": s}).mappings().all()]
            
            print(f" - {s} : {meta['title']} ({count} articles)")
            if articles:
                print(f"    Premier article : {articles[0]['title']} ({articles[0]['year']})")

if __name__ == "__main__":
    test_scenarios()
