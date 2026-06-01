"""
gesica_scenario_enriched_metadata.py
Métadonnées enrichies pour les 27 scénarios GESICA :
  - boolean_queries  : requêtes PubMed booléennes
  - nl_queries       : requêtes en langage naturel pour la recherche sémantique
  - evidence_extraction_prompt : prompt spécifique pour l'extraction d'évidence via RAG
  - model_info       : algorithme, variables, dernière valeur live (description)
  - alert_thresholds : seuils vert/orange/rouge
"""

GESICA_ENRICHED: dict = {

    # ──────────────────────────────────────────────────────────────────────────
    # Cluster 1 — Patient-centered prehospital critical care
    # ──────────────────────────────────────────────────────────────────────────

    "cardiac-arrest-prediction": {
        "boolean_queries": [
            '("out-of-hospital cardiac arrest" OR OHCA) AND (prediction OR forecasting OR "machine learning") AND (2020:2026[dp])',
            '(OHCA OR "cardiac arrest") AND (weather OR temperature OR season OR circadian) AND (incidence OR risk)',
            '("cardiac arrest" OR OHCA) AND ("survival" OR ROSC) AND (EMS OR prehospital) AND (2022:2026[dp])',
        ],
        "nl_queries": [
            "prédiction arrêt cardiaque hors hôpital facteurs de risque météo",
            "out-of-hospital cardiac arrest prediction machine learning EMS",
            "OHCA incidence circadian seasonal patterns Grand Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles scientifiques sur la prédiction de l'arrêt cardiaque extra-hospitalier (OHCA). "
            "Extrayez : (1) les facteurs de risque identifiés (météo, heure, saison, comorbidités), "
            "(2) les algorithmes de prédiction utilisés et leur performance (AUC, sensibilité, spécificité), "
            "(3) les recommandations opérationnelles pour les EMS. "
            "Citez les chiffres clés et les intervalles de confiance. "
            "Contexte : Grand Genève (CH/FR), population ~1M, SMUR + ambulances privées."
        ),
        "model_info": {
            "algorithm": "LightGBM + météo Open-Meteo + circadien",
            "variables": ["température", "humidité", "heure_du_jour", "jour_semaine", "saison", "historique_OHCA_7j"],
            "output": "Probabilité OHCA par zone géographique (J+1)",
            "update_frequency": "Toutes les 6h (données météo temps réel)",
        },
        "alert_thresholds": {
            "green": {"label": "Risque normal", "condition": "score < 0.3"},
            "orange": {"label": "Risque modéré — vigilance", "condition": "0.3 ≤ score < 0.6"},
            "red": {"label": "Risque élevé — alerte", "condition": "score ≥ 0.6"},
        },
    },

    "stroke-detection": {
        "boolean_queries": [
            '(stroke OR "cerebrovascular accident") AND ("door-to-needle" OR thrombolysis OR thrombectomy) AND (prehospital OR EMS) AND (2020:2026[dp])',
            '("FAST score" OR "NIHSS" OR "Cincinnati prehospital stroke scale") AND (EMS OR prehospital)',
            '("stroke unit" OR "comprehensive stroke center") AND ("transport time") AND (outcome OR survival)',
        ],
        "nl_queries": [
            "détection AVC préhospitalière score FAST NIHSS orientation filière",
            "prehospital stroke detection thrombolysis thrombectomy transport time",
            "AVC orientation HUG CHUV délai porte-aiguille",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la détection préhospitalière de l'AVC. "
            "Extrayez : (1) les scores cliniques validés (FAST, NIHSS, CPSS) et leur performance diagnostique, "
            "(2) les délais porte-aiguille et porte-ponction recommandés, "
            "(3) les critères d'orientation vers une UNV vs un centre de thrombectomie. "
            "Contexte : Grand Genève, HUG (Genève) et CHUV (Lausanne) comme centres de référence."
        ),
        "model_info": {
            "algorithm": "XGBoost + NIHSS proxy + OSRM routing",
            "variables": ["score_FAST", "age", "pression_arterielle", "glycemie", "heure_debut_symptomes", "distance_UNV"],
            "output": "Probabilité AVC ischémique + recommandation orientation (UNV/thrombectomie)",
            "update_frequency": "À chaque appel (temps réel)",
        },
        "alert_thresholds": {
            "green": {"label": "Suspicion faible", "condition": "score < 0.4"},
            "orange": {"label": "Suspicion modérée — activer protocole AVC", "condition": "0.4 ≤ score < 0.7"},
            "red": {"label": "Suspicion forte — pré-alerte UNV immédiate", "condition": "score ≥ 0.7"},
        },
    },

    "trauma-severity-assessment": {
        "boolean_queries": [
            '("injury severity score" OR ISS OR TRISS OR RTS) AND (prehospital OR EMS) AND (outcome OR survival) AND (2020:2026[dp])',
            '("damage control" OR "damage control resuscitation") AND (prehospital OR EMS) AND (2020:2026[dp])',
            '("major trauma" OR "polytrauma") AND ("trauma center") AND (EMS OR prehospital) AND (outcome)',
        ],
        "nl_queries": [
            "évaluation gravité traumatisme ISS RTS TRISS orientation trauma center",
            "prehospital trauma severity assessment damage control resuscitation",
            "polytrauma triage EMS orientation niveau 1 HUG CHUV",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'évaluation de la gravité des traumatismes en préhospitalier. "
            "Extrayez : (1) les scores validés (ISS, RTS, TRISS, GCS) et leur valeur pronostique, "
            "(2) les critères de triage vers un Trauma Center de niveau 1, "
            "(3) les protocoles de réanimation damage control préhospitalière. "
            "Contexte : Grand Genève, accidents de la route A1/A40, sports de montagne."
        ),
        "model_info": {
            "algorithm": "Cox Survival + ISS + RTS + TRISS",
            "variables": ["GCS", "pression_arterielle_systolique", "frequence_respiratoire", "mecanisme_traumatisme", "age", "ISS_estime"],
            "output": "Probabilité de survie + recommandation orientation Trauma Center",
            "update_frequency": "À chaque bilan préhospitalier",
        },
        "alert_thresholds": {
            "green": {"label": "Traumatisme mineur", "condition": "RTS > 7.8"},
            "orange": {"label": "Traumatisme modéré — surveillance rapprochée", "condition": "5.0 ≤ RTS ≤ 7.8"},
            "red": {"label": "Traumatisme grave — Trauma Center niveau 1", "condition": "RTS < 5.0"},
        },
    },

    "clinical-deterioration-prediction": {
        "boolean_queries": [
            '("clinical deterioration" OR "early warning score" OR NEWS2 OR MEWS) AND (prehospital OR ambulance OR transport) AND (2020:2026[dp])',
            '("vital signs" OR "physiological monitoring") AND (ambulance OR "air ambulance") AND (deterioration OR alert)',
            '("LSTM" OR "recurrent neural network") AND ("vital signs" OR "patient monitoring") AND (emergency OR prehospital)',
        ],
        "nl_queries": [
            "détérioration clinique en transit NEWS2 MEWS ambulance surveillance",
            "clinical deterioration prediction transport prehospital LSTM vital signs",
            "alerte précoce dégradation patient SMUR hélicoptère",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la prédiction de la détérioration clinique en transport préhospitalier. "
            "Extrayez : (1) les scores d'alerte précoce validés (NEWS2, MEWS) et leurs seuils, "
            "(2) les algorithmes de surveillance continue (LSTM, séries temporelles), "
            "(3) les protocoles de réponse à la détérioration en transit. "
            "Contexte : transport SMUR/hélicoptère Grand Genève, patients critiques."
        ),
        "model_info": {
            "algorithm": "NEWS2 + MEWS + LSTM séries temporelles",
            "variables": ["frequence_cardiaque", "SpO2", "pression_arterielle", "frequence_respiratoire", "temperature", "conscience_AVPU"],
            "output": "Score NEWS2 + tendance + probabilité détérioration dans 30 min",
            "update_frequency": "Toutes les 2 minutes (monitoring continu)",
        },
        "alert_thresholds": {
            "green": {"label": "Stable (NEWS2 0-4)", "condition": "NEWS2 < 5"},
            "orange": {"label": "Surveillance rapprochée (NEWS2 5-6)", "condition": "5 ≤ NEWS2 ≤ 6"},
            "red": {"label": "Urgence — réponse immédiate (NEWS2 ≥ 7)", "condition": "NEWS2 ≥ 7"},
        },
    },

    "patient-pathway-optimization": {
        "boolean_queries": [
            '("patient transfer" OR "interfacility transfer") AND ("cross-border" OR transfrontier) AND (emergency OR EMS) AND (2018:2026[dp])',
            '("bed availability" OR "hospital capacity") AND (real-time OR dashboard) AND (emergency OR ICU) AND (France OR Switzerland)',
            '("patient flow" OR "patient pathway") AND (optimization OR routing) AND (emergency OR prehospital)',
        ],
        "nl_queries": [
            "optimisation parcours patient transfrontalier transfert inter-hospitalier Grand Genève",
            "cross-border patient transfer hospital capacity France Switzerland EMS",
            "disponibilité lits temps réel ROR Suisse orientation patient",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'optimisation du parcours patient transfrontalier. "
            "Extrayez : (1) les modèles d'optimisation de transfert inter-hospitalier, "
            "(2) les outils de suivi de la disponibilité des lits en temps réel, "
            "(3) les protocoles de coordination transfrontalière France-Suisse. "
            "Contexte : Grand Genève, HUG + hôpitaux français (Annecy, Annemasse)."
        ),
        "model_info": {
            "algorithm": "OSRM routing + Programmation Linéaire transfrontalière",
            "variables": ["disponibilite_lits_HUG", "disponibilite_lits_FR", "temps_trajet_OSRM", "type_pathologie", "niveau_urgence"],
            "output": "Hôpital optimal + temps de trajet estimé + disponibilité confirmée",
            "update_frequency": "Toutes les 15 min (synchronisation ROR/SIUR)",
        },
        "alert_thresholds": {
            "green": {"label": "Capacité disponible", "condition": "taux_occupation < 80%"},
            "orange": {"label": "Capacité tendue", "condition": "80% ≤ taux_occupation < 95%"},
            "red": {"label": "Saturation — redirection obligatoire", "condition": "taux_occupation ≥ 95%"},
        },
    },

    "mci-victim-estimation": {
        "boolean_queries": [
            '("mass casualty" OR "multiple casualty" OR MCI) AND (EMS OR ambulance OR prehospital) AND (triage OR management) AND (2018:2026[dp])',
            '("SALT triage" OR "START triage" OR "METHANE") AND (mass casualty OR disaster OR MCI)',
            '("disaster medicine" OR "mass casualty incident") AND (simulation OR planning OR preparedness)',
        ],
        "nl_queries": [
            "estimation victimes catastrophe MCI triage SALT START préhospitalier",
            "mass casualty incident victim estimation EMS prehospital triage",
            "événement à victimes multiples Grand Genève Plan Blanc ORCA",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la gestion des événements à victimes multiples (MCI). "
            "Extrayez : (1) les méthodes d'estimation rapide du nombre de victimes, "
            "(2) les algorithmes de triage (SALT, START) et leur performance, "
            "(3) les modèles de dimensionnement de la réponse EMS. "
            "Contexte : Grand Genève, Plan Blanc (FR) / Plan ORCA (CH)."
        ),
        "model_info": {
            "algorithm": "Monte-Carlo + SALT triage simulation",
            "variables": ["type_evenement", "nombre_victimes_estime", "gravite_distribution", "ressources_disponibles", "distance_hopitaux"],
            "output": "Distribution victimes par catégorie SALT + ressources nécessaires",
            "update_frequency": "À chaque mise à jour terrain (temps réel MCI)",
        },
        "alert_thresholds": {
            "green": {"label": "Situation gérée (< 10 victimes)", "condition": "n_victimes < 10"},
            "orange": {"label": "MCI modéré — renfort nécessaire (10-50)", "condition": "10 ≤ n_victimes < 50"},
            "red": {"label": "MCI majeur — Plan Blanc/ORCA (≥ 50)", "condition": "n_victimes ≥ 50"},
        },
    },

    # ──────────────────────────────────────────────────────────────────────────
    # Cluster 2 — Environmental & Disaster Risk Forecasting
    # ──────────────────────────────────────────────────────────────────────────

    "environmental-risk-forecasting": {
        "boolean_queries": [
            '(pollution OR "air quality" OR ozone OR particulate) AND ("emergency medical services" OR EMS OR ambulance) AND (respiratory OR asthma OR COPD) AND (2020:2026[dp])',
            '("AQI" OR "PM2.5" OR "PM10" OR "NO2") AND (health OR emergency OR hospital) AND (prediction OR forecast)',
            '("pollen" OR "allergen") AND (emergency OR EMS) AND (asthma OR "allergic reaction")',
        ],
        "nl_queries": [
            "prévision risques environnementaux pollution air urgences respiratoires EMS",
            "air quality index AQI emergency services respiratory asthma prediction",
            "pollution ozone PM2.5 impact urgences Grand Genève AirGenève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'impact des risques environnementaux sur les urgences. "
            "Extrayez : (1) les corrélations entre indices de qualité de l'air et appels EMS, "
            "(2) les modèles de prévision de la demande liée à la pollution, "
            "(3) les seuils d'alerte recommandés pour les EMS. "
            "Contexte : Grand Genève, AirGenève + Atmo AURA."
        ),
        "model_info": {
            "algorithm": "DLNM (Distributed Lag Non-linear Model) + AirGenève API",
            "variables": ["AQI", "PM2.5", "PM10", "ozone", "NO2", "pollen_count", "temperature"],
            "output": "Risque d'augmentation appels EMS respiratoires (J+1 à J+3)",
            "update_frequency": "Toutes les heures (données AirGenève temps réel)",
        },
        "alert_thresholds": {
            "green": {"label": "Qualité de l'air bonne", "condition": "AQI < 50"},
            "orange": {"label": "Qualité dégradée — vigilance EMS", "condition": "50 ≤ AQI < 100"},
            "red": {"label": "Pollution élevée — alerte EMS", "condition": "AQI ≥ 100"},
        },
    },

    "disaster-risk-assessment": {
        "boolean_queries": [
            '("natural disaster" OR flood OR earthquake OR landslide) AND ("emergency medical services" OR EMS) AND (impact OR response OR preparedness) AND (2018:2026[dp])',
            '("flood risk" OR "seismic risk") AND (hospital OR EMS OR ambulance) AND (vulnerability OR resilience)',
            '("climate change" OR "extreme weather") AND (EMS OR emergency) AND (infrastructure OR disruption)',
        ],
        "nl_queries": [
            "évaluation risques catastrophes naturelles inondations EMS infrastructure",
            "natural disaster flood earthquake EMS impact response preparedness",
            "risque inondation Arve Rhône casernes ambulances Grand Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'évaluation des risques de catastrophes naturelles pour les EMS. "
            "Extrayez : (1) les méthodes d'évaluation de la vulnérabilité des infrastructures EMS, "
            "(2) les modèles de simulation d'impact (inondations, séismes), "
            "(3) les stratégies de résilience et de continuité d'activité. "
            "Contexte : Grand Genève, risques inondation Arve/Rhône, sismicité modérée."
        ),
        "model_info": {
            "algorithm": "Modèle géospatial + analyse de vulnérabilité",
            "variables": ["zone_inondable", "distance_caserne_zone_risque", "altitude", "acces_routier", "alimentation_electrique"],
            "output": "Score de vulnérabilité des infrastructures EMS + recommandations",
            "update_frequency": "Mensuelle (données météo + hydrologie)",
        },
        "alert_thresholds": {
            "green": {"label": "Risque faible", "condition": "score_risque < 0.3"},
            "orange": {"label": "Risque modéré — planification préventive", "condition": "0.3 ≤ score_risque < 0.6"},
            "red": {"label": "Risque élevé — activation plan d'urgence", "condition": "score_risque ≥ 0.6"},
        },
    },

    "climate-impact-on-ems": {
        "boolean_queries": [
            '("climate change" OR "global warming") AND ("emergency medical services" OR EMS OR ambulance) AND (impact OR adaptation OR projection) AND (2020:2026[dp])',
            '("heat wave" OR heatwave OR "extreme heat") AND (EMS OR emergency OR hospital) AND (demand OR calls OR mortality)',
            '("vector-borne disease" OR dengue OR "West Nile") AND (Europe OR Switzerland OR France) AND (emergence OR risk)',
        ],
        "nl_queries": [
            "impact changement climatique EMS urgences pathologies émergentes adaptation",
            "climate change emergency medical services demand heat waves projection",
            "maladies vectorielles émergentes Europe dengue EMS surveillance",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'impact à long terme du changement climatique sur les EMS. "
            "Extrayez : (1) les projections d'évolution de la demande EMS liée au climat, "
            "(2) les nouvelles pathologies émergentes en Europe (maladies vectorielles, chaleur), "
            "(3) les stratégies d'adaptation des services d'urgence. "
            "Contexte : Grand Genève, projections climatiques Copernicus."
        ),
        "model_info": {
            "algorithm": "Modèle DLNM + projections climatiques Copernicus",
            "variables": ["temperature_max_journaliere", "jours_chaleur_extreme_annuels", "indice_UTCI", "tendance_climatique_10ans"],
            "output": "Projection demande EMS +5/+10 ans par pathologie",
            "update_frequency": "Annuelle (mise à jour projections climatiques)",
        },
        "alert_thresholds": {
            "green": {"label": "Impact climatique faible", "condition": "delta_demande < 5%"},
            "orange": {"label": "Impact modéré — adaptation planifiée", "condition": "5% ≤ delta_demande < 15%"},
            "red": {"label": "Impact fort — réorganisation nécessaire", "condition": "delta_demande ≥ 15%"},
        },
    },

    "heatwave-ems-impact": {
        "boolean_queries": [
            '(heatwave OR "heat wave" OR "extreme heat") AND ("emergency medical services" OR EMS OR ambulance) AND (impact OR demand OR calls) AND (2020:2026[dp])',
            '("heat-related illness" OR hyperthermia OR "heat stroke") AND (prehospital OR EMS OR ambulance)',
            '(UTCI OR "universal thermal climate index" OR "wet bulb globe temperature") AND (health OR mortality OR morbidity)',
        ],
        "nl_queries": [
            "impact canicule EMS demande appels urgences hyperthermie coup de chaleur",
            "heatwave emergency medical services demand heat stroke prehospital",
            "UTCI indice thermique universel santé mortalité Grand Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'impact des canicules sur les EMS. "
            "Extrayez : (1) la relation dose-réponse entre température et appels EMS, "
            "(2) les seuils de température déclenchant une augmentation significative de la demande, "
            "(3) les protocoles de prise en charge préhospitalière des coups de chaleur. "
            "Contexte : Grand Genève, étés de plus en plus chauds, population âgée vulnérable."
        ),
        "model_info": {
            "algorithm": "DLNM + UTCI + Open-Meteo API",
            "variables": ["temperature_max", "UTCI", "humidite_relative", "nuits_tropicales", "population_age_65plus"],
            "output": "Excès de demande EMS prévu (%) + catégories pathologies",
            "update_frequency": "Toutes les 3h (données météo Open-Meteo)",
        },
        "alert_thresholds": {
            "green": {"label": "Conditions normales", "condition": "UTCI < 32°C"},
            "orange": {"label": "Stress thermique fort — vigilance EMS", "condition": "32°C ≤ UTCI < 38°C"},
            "red": {"label": "Stress thermique très fort — alerte canicule", "condition": "UTCI ≥ 38°C"},
        },
    },

    # ──────────────────────────────────────────────────────────────────────────
    # Cluster 3 — Prehospital Emergency Triage & Risk Stratification
    # ──────────────────────────────────────────────────────────────────────────

    "emergency-call-qualification": {
        "boolean_queries": [
            '("emergency call" OR "dispatch" OR "call center") AND ("natural language processing" OR NLP OR "speech recognition") AND (emergency OR EMS) AND (2020:2026[dp])',
            '("automatic speech recognition" OR ASR) AND (emergency OR "medical dispatch") AND (accuracy OR performance)',
            '("call triage" OR "telephone triage") AND (EMS OR "medical dispatch") AND ("machine learning" OR AI)',
        ],
        "nl_queries": [
            "qualification automatique appels urgence NLP reconnaissance vocale Centre 15 144",
            "emergency call NLP speech recognition dispatch qualification machine learning",
            "analyse sémantique appels ARM assistant régulation médicale IA",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la qualification automatisée des appels d'urgence. "
            "Extrayez : (1) les techniques NLP/ASR appliquées aux appels d'urgence, "
            "(2) les performances des systèmes automatisés (précision, rappel, F1), "
            "(3) les recommandations d'intégration dans les centres de régulation. "
            "Contexte : Centre 15 (SAMU Haute-Savoie) + 144 (Genève), ARM bilingues FR/DE."
        ),
        "model_info": {
            "algorithm": "NLP + TF-IDF + scoring sémantique",
            "variables": ["transcription_appel", "mots_cles_critiques", "niveau_stress_vocal", "duree_appel", "localisation"],
            "output": "Catégorie d'urgence + score de priorité + protocole suggéré",
            "update_frequency": "Temps réel (à chaque appel)",
        },
        "alert_thresholds": {
            "green": {"label": "Urgence non vitale", "condition": "score_urgence < 0.4"},
            "orange": {"label": "Urgence potentielle — envoi ambulance", "condition": "0.4 ≤ score_urgence < 0.75"},
            "red": {"label": "Urgence vitale — SMUR immédiat", "condition": "score_urgence ≥ 0.75"},
        },
    },

    "call-prioritization": {
        "boolean_queries": [
            '("call prioritization" OR "dispatch priority" OR "triage by phone") AND (EMS OR "emergency dispatch") AND (2018:2026[dp])',
            '("ProQA" OR "MPDS" OR "AMPDS") AND (emergency OR dispatch OR EMS) AND (accuracy OR performance)',
            '("priority dispatch" OR "advanced medical priority dispatch") AND (outcome OR mortality OR response)',
        ],
        "nl_queries": [
            "priorisation appels urgence dispatch EMS AMPDS ProQA performance",
            "call prioritization emergency dispatch system accuracy outcomes",
            "aide décision ARM priorisation appels Centre 15 144",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la priorisation des appels d'urgence. "
            "Extrayez : (1) les systèmes de priorisation validés (AMPDS, ProQA) et leur performance, "
            "(2) les taux d'erreur de priorisation et leurs conséquences cliniques, "
            "(3) les améliorations apportées par l'IA. "
            "Contexte : Centre 15/144 Grand Genève, volume ~500 appels/jour."
        ),
        "model_info": {
            "algorithm": "Arbre de décision + scoring AMPDS",
            "variables": ["motif_appel", "age_patient", "symptomes_declares", "antecedents_connus", "localisation"],
            "output": "Niveau de priorité (P1-P4) + ressource recommandée",
            "update_frequency": "Temps réel",
        },
        "alert_thresholds": {
            "green": {"label": "P3-P4 — Urgence relative", "condition": "priorite >= 3"},
            "orange": {"label": "P2 — Urgence médicale", "condition": "priorite == 2"},
            "red": {"label": "P1 — Urgence vitale immédiate", "condition": "priorite == 1"},
        },
    },

    "mass-casualty-triage": {
        "boolean_queries": [
            '("mass casualty triage" OR "SALT triage" OR "START triage") AND (EMS OR prehospital) AND (2018:2026[dp])',
            '("triage accuracy" OR "triage performance") AND ("mass casualty" OR MCI OR disaster)',
            '("electronic triage" OR "digital triage" OR "smart triage") AND (mass casualty OR disaster)',
        ],
        "nl_queries": [
            "triage catastrophe SALT START victimes multiples EMS performance",
            "mass casualty triage accuracy electronic smart triage EMS",
            "triage numérique MCI Grand Genève exercice simulation",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur le triage en situation de catastrophe. "
            "Extrayez : (1) les algorithmes de triage (SALT, START, SIEVE) et leur précision, "
            "(2) les outils numériques de triage et leur impact sur les résultats, "
            "(3) les recommandations pour les exercices de simulation. "
            "Contexte : Grand Genève, Plan ORCA (CH) + Plan Rouge (FR)."
        ),
        "model_info": {
            "algorithm": "Monte-Carlo + SALT triage simulation",
            "variables": ["n_victimes", "distribution_gravite", "ressources_SMUR", "ressources_ambulances", "temps_arrivee_renforts"],
            "output": "Recommandation triage + allocation ressources optimale",
            "update_frequency": "Temps réel (mise à jour terrain)",
        },
        "alert_thresholds": {
            "green": {"label": "Situation contrôlée", "condition": "ratio_ressources_victimes > 0.5"},
            "orange": {"label": "Ressources insuffisantes — renfort nécessaire", "condition": "0.2 ≤ ratio < 0.5"},
            "red": {"label": "Dépassement capacités — Plan Blanc", "condition": "ratio < 0.2"},
        },
    },

    "undertriage-detection": {
        "boolean_queries": [
            '(undertriage OR "under-triage") AND (EMS OR ambulance OR prehospital) AND (risk OR prediction OR detection) AND (2018:2026[dp])',
            '("triage accuracy" OR "triage error") AND (prehospital OR EMS) AND (outcome OR mortality)',
            '("missed diagnosis" OR undertriage) AND (trauma OR cardiac OR stroke) AND (ambulance OR EMS)',
        ],
        "nl_queries": [
            "détection sous-triage EMS ambulance risque prédiction erreur triage",
            "undertriage detection prehospital EMS risk prediction machine learning",
            "erreur triage ambulance Grand Genève conséquences mortalité",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la détection et la prévention du sous-triage préhospitalier. "
            "Extrayez : (1) les facteurs de risque de sous-triage identifiés, "
            "(2) les algorithmes de détection (Random Forest, régression logistique), "
            "(3) les interventions efficaces pour réduire le sous-triage. "
            "Contexte : EMS Grand Genève, patients âgés, pathologies atypiques."
        ),
        "model_info": {
            "algorithm": "Random Forest + Régression Logistique",
            "variables": ["age", "sexe", "comorbidites", "constantes_vitales", "motif_appel", "heure_intervention"],
            "output": "Score de risque de sous-triage + recommandation réévaluation",
            "update_frequency": "À chaque bilan préhospitalier",
        },
        "alert_thresholds": {
            "green": {"label": "Triage probablement correct", "condition": "score_undertriage < 0.3"},
            "orange": {"label": "Risque modéré — réévaluation recommandée", "condition": "0.3 ≤ score < 0.6"},
            "red": {"label": "Risque élevé — réévaluation immédiate", "condition": "score ≥ 0.6"},
        },
    },

    "triage-support": {
        "boolean_queries": [
            '(triage OR "CCMU" OR "NEWS2" OR "Manchester triage") AND ("emergency medical services" OR prehospital) AND (2020:2026[dp])',
            '("artificial intelligence" OR "machine learning" OR NLP) AND (triage OR "clinical decision support") AND (emergency OR prehospital)',
            '("undertriage" OR "overtriage") AND (EMS OR ambulance) AND (accuracy OR sensitivity OR specificity)',
        ],
        "nl_queries": [
            "aide au triage EMS CCMU NEWS2 Manchester IA décision clinique",
            "AI triage support prehospital emergency clinical decision machine learning",
            "triage préhospitalier outil aide décision Grand Genève SMUR",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur les outils d'aide au triage préhospitalier. "
            "Extrayez : (1) les scores de triage validés (CCMU, NEWS2, Manchester) et leur performance, "
            "(2) les systèmes d'aide à la décision clinique basés sur l'IA, "
            "(3) les recommandations d'implémentation dans les EMS. "
            "Contexte : Grand Genève, SMUR + ambulances, dossier patient embarqué."
        ),
        "model_info": {
            "algorithm": "CCMU + NEWS2 + LLM NLP",
            "variables": ["constantes_vitales", "motif_appel", "age", "antecedents", "score_CCMU", "score_NEWS2"],
            "output": "Catégorie CCMU + recommandation destination + niveau urgence",
            "update_frequency": "Temps réel",
        },
        "alert_thresholds": {
            "green": {"label": "CCMU 1-2 — Urgence relative", "condition": "CCMU <= 2"},
            "orange": {"label": "CCMU 3 — Urgence médicale", "condition": "CCMU == 3"},
            "red": {"label": "CCMU 4-5 — Urgence vitale", "condition": "CCMU >= 4"},
        },
    },

    "dispatch-decision-support": {
        "boolean_queries": [
            '("dispatch decision" OR "dispatch support" OR "computer-aided dispatch") AND (EMS OR ambulance) AND (2020:2026[dp])',
            '("CAD system" OR "computer aided dispatch") AND (emergency OR EMS) AND (performance OR accuracy OR outcome)',
            '("resource allocation" OR "unit selection") AND (EMS OR ambulance OR dispatch) AND (optimization)',
        ],
        "nl_queries": [
            "aide décision dispatch EMS CAD système informatisé allocation ressources",
            "computer aided dispatch EMS decision support optimization performance",
            "dispatch ambulance SMUR aide décision Grand Genève TECHWAN SAGA",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur les systèmes d'aide à la décision de dispatch EMS. "
            "Extrayez : (1) les algorithmes de sélection des ressources (arbre de décision, VRP), "
            "(2) les performances des systèmes CAD et leur impact sur les délais, "
            "(3) les recommandations d'intégration avec les systèmes existants. "
            "Contexte : Centre 15/144 Grand Genève, TECHWAN SAGA (FR) + système CH."
        ),
        "model_info": {
            "algorithm": "Arbre de décision + VRP (Vehicle Routing Problem)",
            "variables": ["position_unites", "type_urgence", "disponibilite_SMUR", "temps_trajet_OSRM", "competences_equipage"],
            "output": "Unité recommandée + temps d'arrivée estimé + alternatives",
            "update_frequency": "Temps réel (GPS + disponibilité)",
        },
        "alert_thresholds": {
            "green": {"label": "Ressource disponible < 8 min", "condition": "ETA < 8"},
            "orange": {"label": "Délai modéré 8-15 min", "condition": "8 ≤ ETA < 15"},
            "red": {"label": "Délai critique > 15 min — renfort transfrontalier", "condition": "ETA ≥ 15"},
        },
    },

    # ──────────────────────────────────────────────────────────────────────────
    # Cluster 4 — EMS Operations & Resource Management
    # ──────────────────────────────────────────────────────────────────────────

    "response-time-optimization": {
        "boolean_queries": [
            '("response time" OR "dispatch time") AND ("emergency medical services" OR ambulance) AND (optimization OR routing OR GIS) AND (2020:2026[dp])',
            '("cross-border" OR transfrontier OR "mutual aid") AND (EMS OR ambulance OR prehospital) AND (response OR coordination)',
            '("ambulance placement" OR "base location" OR "coverage optimization") AND (emergency OR EMS)',
        ],
        "nl_queries": [
            "optimisation temps de réponse EMS ambulance routage GIS couverture",
            "EMS response time optimization routing cross-border ambulance placement",
            "positionnement ambulances Grand Genève couverture transfrontalière OSRM",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'optimisation des temps de réponse EMS. "
            "Extrayez : (1) les modèles de positionnement dynamique des ambulances, "
            "(2) les algorithmes de routage (OSRM, Dijkstra) et leur précision, "
            "(3) les stratégies de coopération transfrontalière pour réduire les délais. "
            "Contexte : Grand Genève, objectif < 8 min pour urgences vitales."
        ),
        "model_info": {
            "algorithm": "OSRM + optimisation couverture spatiale",
            "variables": ["position_ambulances", "demande_historique", "trafic_temps_reel", "zones_couverture", "population_densite"],
            "output": "Positionnement optimal ambulances + temps de réponse estimé par zone",
            "update_frequency": "Toutes les 30 min (données trafic + position GPS)",
        },
        "alert_thresholds": {
            "green": {"label": "Couverture optimale (< 8 min)", "condition": "temps_reponse_median < 8"},
            "orange": {"label": "Couverture dégradée (8-12 min)", "condition": "8 ≤ temps_reponse_median < 12"},
            "red": {"label": "Couverture insuffisante (> 12 min)", "condition": "temps_reponse_median ≥ 12"},
        },
    },

    "ambulance-dispatch-optimization": {
        "boolean_queries": [
            '("ambulance dispatch" OR "fleet management") AND (optimization OR "vehicle routing") AND (EMS OR emergency) AND (2020:2026[dp])',
            '("dynamic ambulance" OR "real-time dispatch") AND (optimization OR algorithm OR AI)',
            '("VRP" OR "vehicle routing problem") AND (ambulance OR EMS OR emergency)',
        ],
        "nl_queries": [
            "optimisation dispatch ambulances flotte VRP temps réel EMS",
            "ambulance fleet management dispatch optimization vehicle routing EMS",
            "gestion flotte ambulances Grand Genève optimisation ressources",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'optimisation du dispatch des ambulances. "
            "Extrayez : (1) les algorithmes de VRP appliqués aux ambulances, "
            "(2) les systèmes de dispatch dynamique en temps réel, "
            "(3) les gains mesurés en termes de temps de réponse et de couverture. "
            "Contexte : Grand Genève, ~30 ambulances, couverture transfrontalière."
        ),
        "model_info": {
            "algorithm": "VRP + couverture spatiale dynamique",
            "variables": ["position_GPS_ambulances", "statut_disponibilite", "demande_prevue", "priorite_appel", "competences_equipage"],
            "output": "Affectation optimale ambulance-appel + repositionnement préventif",
            "update_frequency": "Temps réel (GPS + CAD)",
        },
        "alert_thresholds": {
            "green": {"label": "Flotte disponible > 70%", "condition": "disponibilite > 0.7"},
            "orange": {"label": "Flotte tendue 40-70%", "condition": "0.4 ≤ disponibilite ≤ 0.7"},
            "red": {"label": "Flotte saturée < 40%", "condition": "disponibilite < 0.4"},
        },
    },

    "staffing-level-prediction": {
        "boolean_queries": [
            '("staffing" OR "workforce") AND ("emergency medical services" OR EMS OR ambulance) AND (prediction OR optimization OR planning) AND (2020:2026[dp])',
            '("Erlang" OR "queuing theory") AND (EMS OR ambulance OR emergency) AND (staffing OR capacity)',
            '("nurse staffing" OR "paramedic staffing") AND (emergency OR EMS) AND (outcome OR quality)',
        ],
        "nl_queries": [
            "prédiction niveaux staffing EMS ambulance planification ressources humaines",
            "EMS staffing prediction optimization Erlang queuing theory workforce",
            "planification effectifs ambulanciers Grand Genève saisonnalité",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la prédiction et l'optimisation des effectifs EMS. "
            "Extrayez : (1) les modèles de prévision des besoins en personnel (Erlang, files d'attente), "
            "(2) les facteurs influençant la demande de personnel (saisonnalité, événements), "
            "(3) les recommandations pour la planification des gardes. "
            "Contexte : Grand Genève, ambulanciers CH + FR, conventions collectives différentes."
        ),
        "model_info": {
            "algorithm": "Prophet + NEDOCS + Erlang C",
            "variables": ["demande_historique", "saison", "evenements_prevus", "taux_absenteisme", "formation_requise"],
            "output": "Effectif optimal par quart + probabilité de saturation",
            "update_frequency": "Hebdomadaire (planification) + quotidienne (ajustement)",
        },
        "alert_thresholds": {
            "green": {"label": "Effectif suffisant", "condition": "NEDOCS < 60"},
            "orange": {"label": "Effectif tendu — heures supplémentaires", "condition": "60 ≤ NEDOCS < 100"},
            "red": {"label": "Effectif insuffisant — rappel de personnel", "condition": "NEDOCS ≥ 100"},
        },
    },

    "hospital-capacity-forecasting": {
        "boolean_queries": [
            '("hospital capacity" OR "bed availability" OR "ICU capacity") AND (forecasting OR prediction OR "machine learning") AND (emergency OR EMS) AND (2020:2026[dp])',
            '("emergency department crowding" OR "ED overcrowding" OR NEDOCS) AND (prediction OR management)',
            '("hospital surge" OR "capacity management") AND (emergency OR EMS) AND (2020:2026[dp])',
        ],
        "nl_queries": [
            "prévision capacité hospitalière lits disponibles urgences saturation",
            "hospital capacity forecasting bed availability emergency department crowding",
            "saturation urgences HUG Grand Genève prévision NEDOCS",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la prévision de la capacité hospitalière. "
            "Extrayez : (1) les modèles de prévision de la saturation des urgences (NEDOCS, EDWIN), "
            "(2) les algorithmes de prédiction de la disponibilité des lits, "
            "(3) les stratégies de gestion des pics d'affluence. "
            "Contexte : HUG Genève + hôpitaux français, coordination transfrontalière."
        ),
        "model_info": {
            "algorithm": "Prophet + NEDOCS + Erlang C",
            "variables": ["nb_patients_urgences", "nb_lits_disponibles", "duree_sejour_moyenne", "admissions_prevues", "sorties_prevues"],
            "output": "Score NEDOCS + probabilité saturation dans 4h/8h/24h",
            "update_frequency": "Toutes les heures (données SIHF/PMSI)",
        },
        "alert_thresholds": {
            "green": {"label": "Capacité normale (NEDOCS < 60)", "condition": "NEDOCS < 60"},
            "orange": {"label": "Capacité tendue (NEDOCS 60-100)", "condition": "60 ≤ NEDOCS < 100"},
            "red": {"label": "Saturation (NEDOCS ≥ 100)", "condition": "NEDOCS ≥ 100"},
        },
    },

    "demand-forecasting": {
        "boolean_queries": [
            '("emergency medical services" OR EMS OR ambulance) AND (demand OR call OR volume) AND (forecasting OR prediction OR "time series") AND (2020:2026[dp])',
            '("ambulance demand" OR "EMS call volume") AND (machine learning OR "random forest" OR XGBoost OR LightGBM OR Prophet)',
            '(weather OR temperature OR season) AND ("EMS demand" OR "ambulance calls") AND (prediction OR forecast)',
        ],
        "nl_queries": [
            "prévision demande EMS ambulance volume appels séries temporelles météo",
            "EMS demand forecasting ambulance call volume machine learning Prophet LightGBM",
            "prédiction demande SAMU 74 Grand Genève saisonnalité météo",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la prévision de la demande EMS. "
            "Extrayez : (1) les modèles de prévision validés (Prophet, SARIMA, XGBoost) et leur précision (MAPE, RMSE), "
            "(2) les facteurs influençant la demande (météo, saison, épidémies, événements), "
            "(3) les recommandations pour l'implémentation opérationnelle. "
            "Contexte : Grand Genève, ~500 appels/jour, saisonnalité marquée."
        ),
        "model_info": {
            "algorithm": "Prophet + LightGBM + données météo Open-Meteo",
            "variables": ["historique_appels_7j", "temperature", "precipitations", "jour_semaine", "feries", "incidence_grippale"],
            "output": "Prévision demande EMS J+1 à J+7 par heure et par zone",
            "update_frequency": "Quotidienne (6h UTC)",
        },
        "alert_thresholds": {
            "green": {"label": "Demande normale (± 10%)", "condition": "delta_demande < 10%"},
            "orange": {"label": "Demande élevée (+10 à +30%)", "condition": "10% ≤ delta_demande < 30%"},
            "red": {"label": "Pic de demande (> +30%)", "condition": "delta_demande ≥ 30%"},
        },
    },

    "resource-allocation": {
        "boolean_queries": [
            '("resource allocation" OR "resource management") AND ("emergency medical services" OR EMS) AND (optimization OR efficiency) AND (2020:2026[dp])',
            '("dynamic resource" OR "real-time resource") AND (EMS OR ambulance OR emergency) AND (allocation OR management)',
            '("supply chain" OR "inventory management") AND (EMS OR ambulance OR emergency) AND (optimization)',
        ],
        "nl_queries": [
            "allocation ressources EMS optimisation efficience gestion temps réel",
            "EMS resource allocation optimization dynamic real-time management",
            "gestion ressources EMS Grand Genève matériel stocks optimisation",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'allocation des ressources EMS. "
            "Extrayez : (1) les modèles d'allocation dynamique des ressources (PL, métaheuristiques), "
            "(2) les indicateurs de performance (taux d'utilisation, délais), "
            "(3) les stratégies de mutualisation transfrontalière. "
            "Contexte : Grand Genève, ressources CH + FR, conventions de mutualisation."
        ),
        "model_info": {
            "algorithm": "Programmation Linéaire + Isolation Forest",
            "variables": ["disponibilite_ressources", "demande_prevue", "cout_ressource", "priorite_urgence", "distance_depot"],
            "output": "Allocation optimale ressources + indicateurs d'efficience",
            "update_frequency": "Toutes les 4h",
        },
        "alert_thresholds": {
            "green": {"label": "Ressources bien allouées", "condition": "taux_utilisation < 75%"},
            "orange": {"label": "Ressources tendues", "condition": "75% ≤ taux_utilisation < 90%"},
            "red": {"label": "Ressources saturées", "condition": "taux_utilisation ≥ 90%"},
        },
    },

    # ──────────────────────────────────────────────────────────────────────────
    # Cluster 5 — Surveillance & Epidemic Management
    # ──────────────────────────────────────────────────────────────────────────

    "epidemic-early-warning": {
        "boolean_queries": [
            '(influenza OR "respiratory syncytial virus" OR gastroenteritis) AND (surveillance OR "early warning" OR forecasting) AND (EMS OR prehospital)',
            '("epidemic threshold" OR "sentinel surveillance" OR "syndromic surveillance") AND (machine learning OR SARIMAX OR Prophet) AND (2022:2026[dp])',
            '("Réseau Sentinelles" OR "ILI surveillance" OR "influenza-like illness") AND (prediction OR forecast OR "time series")',
        ],
        "nl_queries": [
            "alerte précoce épidémique surveillance sentinelles EMS signaux faibles",
            "epidemic early warning surveillance EMS SARIMAX Prophet machine learning",
            "détection précoce épidémie grippe gastro Grand Genève Sentinelles",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur les systèmes d'alerte précoce épidémique. "
            "Extrayez : (1) les modèles de détection précoce (SARIMAX, Prophet, Farrington), "
            "(2) les sources de données utilisées (Sentinelles, urgences, EMS), "
            "(3) les seuils épidémiques et leur validation. "
            "Contexte : Grand Genève, Réseau Sentinelles FR + surveillance OFSP CH."
        ),
        "model_info": {
            "algorithm": "Ensemble SARIMAX + Prophet + XGBoost + Farrington Flexible",
            "variables": ["incidence_ILI_sentinelles", "appels_EMS_syndrome_grippal", "temperature", "semaine_iso", "seuil_epidemique"],
            "output": "Niveau alerte épidémique (pré-épidémique/épidémique/post) + prévision J+14",
            "update_frequency": "Hebdomadaire (données Sentinelles J-3)",
        },
        "alert_thresholds": {
            "green": {"label": "Niveau basal — surveillance normale", "condition": "incidence < seuil_pre_epidemique"},
            "orange": {"label": "Pré-épidémique — vigilance renforcée", "condition": "seuil_pre_epidemique ≤ incidence < seuil_epidemique"},
            "red": {"label": "Épidémie déclarée — alerte OFSP/ARS", "condition": "incidence ≥ seuil_epidemique"},
        },
    },

    "surveillance": {
        "boolean_queries": [
            '("syndromic surveillance" OR "emergency department surveillance") AND (EMS OR ambulance OR emergency) AND (2020:2026[dp])',
            '("anomaly detection" OR "cluster detection") AND (health OR emergency OR EMS) AND (machine learning OR statistical)',
            '("real-time surveillance" OR "biosurveillance") AND (emergency OR EMS OR hospital)',
        ],
        "nl_queries": [
            "surveillance syndromique active urgences anomalies clusters géographiques",
            "syndromic surveillance emergency department real-time anomaly detection",
            "surveillance épidémiologique Grand Genève urgences SOS Médecins",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la surveillance syndromique active. "
            "Extrayez : (1) les méthodes de détection d'anomalies (CUSUM, EARS, SaTScan), "
            "(2) les sources de données utilisées (urgences, pharmacies, EMS), "
            "(3) les seuils de déclenchement d'alerte validés. "
            "Contexte : Grand Genève, données urgences HUG + hôpitaux français."
        ),
        "model_info": {
            "algorithm": "Isolation Forest + CUSUM + SaTScan géospatial",
            "variables": ["passages_urgences", "motifs_consultation", "localisation_patient", "age", "symptomes"],
            "output": "Score d'anomalie + clusters géographiques suspects + niveau alerte",
            "update_frequency": "Quotidienne",
        },
        "alert_thresholds": {
            "green": {"label": "Activité normale", "condition": "score_anomalie < 2.0"},
            "orange": {"label": "Anomalie détectée — investigation", "condition": "2.0 ≤ score_anomalie < 3.5"},
            "red": {"label": "Cluster confirmé — alerte sanitaire", "condition": "score_anomalie ≥ 3.5"},
        },
    },

    "surge-management": {
        "boolean_queries": [
            '("surge capacity" OR "surge management") AND (emergency OR hospital OR EMS) AND (2020:2026[dp])',
            '("emergency department surge" OR "hospital surge") AND (management OR strategy OR response)',
            '("demand surge" OR "patient surge") AND (EMS OR ambulance OR emergency) AND (management OR planning)',
        ],
        "nl_queries": [
            "gestion pics afflux surge urgences EMS stratégies opérationnelles capacité",
            "surge capacity management emergency department hospital EMS strategies",
            "gestion afflux massif urgences Grand Genève tentes tri lignes régulation",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la gestion des pics d'afflux aux urgences. "
            "Extrayez : (1) les stratégies de gestion du surge (files d'attente M/M/c, PL), "
            "(2) les indicateurs de saturation (NEDOCS, EDWIN, READI), "
            "(3) les interventions efficaces pour réduire la congestion. "
            "Contexte : Grand Genève, HUG + hôpitaux français, coordination transfrontalière."
        ),
        "model_info": {
            "algorithm": "M/M/c files d'attente + Programmation Linéaire",
            "variables": ["taux_arrivee_patients", "capacite_traitement", "nb_lits_disponibles", "duree_sejour", "ressources_humaines"],
            "output": "Temps d'attente estimé + recommandations activation ressources supplémentaires",
            "update_frequency": "Toutes les 30 min",
        },
        "alert_thresholds": {
            "green": {"label": "Flux normal", "condition": "taux_occupation_urgences < 80%"},
            "orange": {"label": "Flux tendu — mesures préventives", "condition": "80% ≤ taux_occupation < 95%"},
            "red": {"label": "Saturation — activation plan surge", "condition": "taux_occupation ≥ 95%"},
        },
    },

    "pandemic-preparedness": {
        "boolean_queries": [
            '("pandemic preparedness" OR "pandemic planning") AND (EMS OR ambulance OR emergency) AND (2020:2026[dp])',
            '("COVID-19" OR SARS-CoV-2 OR influenza) AND ("emergency medical services" OR EMS) AND (preparedness OR response OR resilience)',
            '("business continuity" OR "continuity of operations") AND (EMS OR emergency OR hospital) AND (pandemic OR epidemic)',
        ],
        "nl_queries": [
            "préparation pandémie EMS planification résilience continuité activité",
            "pandemic preparedness EMS emergency services resilience COVID-19 planning",
            "plan continuité activité EMS Grand Genève pandémie stocks stratégiques",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la préparation aux pandémies pour les EMS. "
            "Extrayez : (1) les modèles épidémiques (SEIR) et leurs paramètres pour les EMS, "
            "(2) les stratégies de continuité d'activité validées, "
            "(3) les recommandations pour les stocks stratégiques. "
            "Contexte : Grand Genève, leçons COVID-19, OFSP + ARS Auvergne-Rhône-Alpes."
        ),
        "model_info": {
            "algorithm": "SEIR + Monte-Carlo + graphe de décision",
            "variables": ["R0", "taux_hospitalisation", "capacite_EMS", "stocks_EPI", "taux_vaccination"],
            "output": "Scénarios pandémiques + recommandations dimensionnement ressources",
            "update_frequency": "Hebdomadaire (données épidémiques OFSP/ARS)",
        },
        "alert_thresholds": {
            "green": {"label": "Phase inter-pandémique", "condition": "R0 < 1.0"},
            "orange": {"label": "Alerte pandémique — activation PCA", "condition": "1.0 ≤ R0 < 2.0"},
            "red": {"label": "Pandémie déclarée — plan d'urgence", "condition": "R0 ≥ 2.0"},
        },
    },

    # ──────────────────────────────────────────────────────────────────────────
    # Cluster 6 — Cross-border & Operational Coordination
    # ──────────────────────────────────────────────────────────────────────────

    "cross-border-coordination": {
        "boolean_queries": [
            '("cross-border" OR transfrontier OR "cross-border healthcare") AND (EMS OR ambulance OR emergency) AND (coordination OR protocol OR agreement) AND (2018:2026[dp])',
            '(France OR Switzerland OR "Franco-Swiss") AND (EMS OR ambulance OR "emergency medical") AND (cooperation OR agreement OR protocol)',
            '("mutual aid" OR "aid agreement") AND (EMS OR ambulance) AND (cross-border OR international)',
        ],
        "nl_queries": [
            "coordination sanitaire transfrontalière France Suisse EMS protocoles accords",
            "cross-border EMS coordination France Switzerland mutual aid agreements",
            "coopération transfrontalière ambulances SMUR Grand Genève TECHWAN SAGA",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la coordination sanitaire transfrontalière. "
            "Extrayez : (1) les modèles de coordination EMS transfrontaliers et leur efficacité, "
            "(2) les barrières juridiques et administratives identifiées, "
            "(3) les meilleures pratiques de protocoles de coopération. "
            "Contexte : Grand Genève, France (Haute-Savoie, Ain) + Suisse (Genève, Vaud)."
        ),
        "model_info": {
            "algorithm": "Modèle géospatial + graphe de décision transfrontalier",
            "variables": ["position_unites_CH", "position_unites_FR", "protocoles_actifs", "temps_passage_frontiere", "disponibilite_ressources"],
            "output": "Recommandation ressource transfrontalière + délai estimé",
            "update_frequency": "Temps réel",
        },
        "alert_thresholds": {
            "green": {"label": "Coordination normale", "condition": "protocoles_actifs == True"},
            "orange": {"label": "Coordination dégradée — contact direct", "condition": "delai_coordination > 5 min"},
            "red": {"label": "Rupture coordination — escalade hiérarchique", "condition": "protocoles_inactifs == True"},
        },
    },

    "situational-awareness": {
        "boolean_queries": [
            '("situational awareness" OR "common operating picture") AND (EMS OR emergency OR prehospital) AND (2020:2026[dp])',
            '("real-time dashboard" OR "operational dashboard") AND (EMS OR emergency OR ambulance) AND (performance OR monitoring)',
            '("data integration" OR "data fusion") AND (EMS OR emergency) AND (situational OR operational OR awareness)',
        ],
        "nl_queries": [
            "conscience situationnelle opérationnelle tableau de bord EMS temps réel",
            "situational awareness EMS real-time dashboard operational monitoring",
            "tableau de bord opérationnel Grand Genève EMS météo épidémie trafic",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la conscience situationnelle opérationnelle pour les EMS. "
            "Extrayez : (1) les architectures de tableaux de bord opérationnels validés, "
            "(2) les sources de données intégrées et leur valeur ajoutée, "
            "(3) les indicateurs clés de performance (KPI) recommandés. "
            "Contexte : Grand Genève, intégration GPS + météo + épidémie + trafic."
        ),
        "model_info": {
            "algorithm": "Intégration multi-sources + alertes temps réel",
            "variables": ["position_unites_GPS", "statut_disponibilite", "alertes_meteo", "alertes_epidemiques", "trafic_temps_reel"],
            "output": "Vue opérationnelle unifiée + alertes prioritaires",
            "update_frequency": "Temps réel (< 30 secondes)",
        },
        "alert_thresholds": {
            "green": {"label": "Situation normale", "condition": "nb_alertes_actives == 0"},
            "orange": {"label": "Alertes en cours — surveillance renforcée", "condition": "1 ≤ nb_alertes_actives < 3"},
            "red": {"label": "Situation critique — cellule de crise", "condition": "nb_alertes_actives ≥ 3"},
        },
    },
}
