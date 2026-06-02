#!/usr/bin/env python3
"""
gesica_scenario_enriched_metadata.py
Métadonnées enrichies pour les 28 scénarios GESICA :
  - boolean_queries  : requêtes PubMed booléennes
  - nl_queries       : requêtes en langage naturel pour la recherche sémantique
  - evidence_extraction_prompt : prompt spécifique pour l'extraction d'évidence via RAG
  - model_info       : algorithme, variables, dernière valeur live (description)
  - alert_thresholds : seuils vert/orange/rouge
  - databases        : liste des bases de données réelles nécessaires
  - variables_detail : dictionnaire des variables utilisées avec définition exacte et statut de connexion (plugged)
  - outcome_definition : définition de l'outcome ou de l'indicateur clé de performance surveillé
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
        "databases": [
            "Registre National des Arrêts Cardiaques (ex: RéAC en France, registre cantonal à Genève)",
            "Données météorologiques historiques et temps réel (Open-Meteo API)",
            "Données de régulation des appels EMS (SAGA / Techwan / SIS)"
        ],
        "outcome_definition": "Incidence quotidienne d'arrêts cardiaques extra-hospitaliers (OHCA) confirmés par zone géographique (cellule de 1km²).",
        "variables_detail": {
            "température": {
                "definition": "Température horaire de l'air à 2 mètres du sol en degrés Celsius.",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            },
            "humidité": {
                "definition": "Humidité relative de l'air en pourcentage à 2 mètres du sol.",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            },
            "heure_du_jour": {
                "definition": "Heure locale (0-23) pour capturer le rythme circadien de l'incidence des OHCA.",
                "plugged": True,
                "source": "Horloge système"
            },
            "jour_semaine": {
                "definition": "Jour de la semaine (0=Lundi, 6=Dimanche) pour modéliser les variations hebdomadaires.",
                "plugged": True,
                "source": "Horloge système"
            },
            "saison": {
                "definition": "Saison astronomique (Hiver, Printemps, Été, Automne) liée aux variations de température.",
                "plugged": True,
                "source": "Horloge système"
            },
            "historique_OHCA_7j": {
                "definition": "Nombre cumulé d'arrêts cardiaques enregistrés dans la zone au cours des 7 derniers jours.",
                "plugged": False,
                "source": "Registre Cantonal OHCA (Manquant - Dataset requis)"
            }
        },
        "keywords": ["Arrêt cardiaque", "OHCA", "Chaîne de survie", "DEA", "Premiers répondants", "Gasping", "Réanimation cardiopulmonaire"],
        "clinical_rationale": (
            "L'arrêt cardiaque extra-hospitalier (OHCA) reste une cause majeure de mortalité. Chaque minute sans réanimation réduit les chances de survie de 10%. La prédiction spatio-temporelle et l'identification précoce (notamment via l'analyse acoustique des appels) permettent de déployer les premiers répondants et d'ajuster le positionnement des ambulances pour intervenir dans la 'minute d'or'."
        ),
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
        "databases": [
            "Dossier Patient Informatisé (DPI) des urgences (HUG/CHUV)",
            "Registre des AVC (Stroke Unit HUG)",
            "API de calcul d'itinéraire routier (OSRM / OpenStreetMap)"
        ],
        "outcome_definition": "Délai de prise en charge porte-aiguille (door-to-needle) et porte-ponction (door-to-puncture) optimal pour la reperfusion cérébrale.",
        "variables_detail": {
            "score_FAST": {
                "definition": "Score clinique préhospitalier évaluant l'asymétrie faciale (Face), la faiblesse des bras (Arm), les troubles de la parole (Speech) et le temps d'apparition (Time).",
                "plugged": False,
                "source": "Fiche d'intervention ambulancière (Manquant - Dataset requis)"
            },
            "age": {
                "definition": "Âge du patient en années.",
                "plugged": False,
                "source": "Régulation EMS (Manquant - Dataset requis)"
            },
            "pression_arterielle": {
                "definition": "Pression artérielle systolique mesurée en mmHg par l'équipage préhospitalier.",
                "plugged": False,
                "source": "Moniteur multiparamétrique (Manquant - Dataset requis)"
            },
            "glycemie": {
                "definition": "Taux de glucose sanguin en mmol/L pour éliminer une hypoglycémie mimant un AVC.",
                "plugged": False,
                "source": "Glucomètre préhospitalier (Manquant - Dataset requis)"
            },
            "heure_debut_symptomes": {
                "definition": "Heure exacte d'apparition des premiers symptômes ou heure du 'dernier état normal connu' (Last Seen Well).",
                "plugged": False,
                "source": "Interrogatoire régulation/famille (Manquant - Dataset requis)"
            },
            "distance_UNV": {
                "definition": "Distance en mètres et temps de trajet en minutes jusqu'à l'Unité Neurovasculaire (UNV) la plus proche.",
                "plugged": True,
                "source": "OSRM API (Live)"
            }
        },
        "keywords": ["AVC", "Thrombolyse", "Thrombectomie", "Score FAST", "Score NIHSS", "Occlusion grand vaisseau", "Time is brain"],
        "clinical_rationale": (
            "Dans l'accident vasculaire cérébral (AVC), chaque minute perdue équivaut à la perte de 1,9 million de neurones. Une détection préhospitalière fiable permet de contourner les urgences générales pour orienter le patient directement vers une Unité Neurovasculaire (UNV) apte à réaliser une thrombolyse ou une thrombectomie mécanique."
        ),
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
            "output": "Score de mortalité à 30 jours + recommandation d'orientation vers Trauma Center niveau 1",
            "update_frequency": "Temps réel à l'intervention",
        },
        "alert_thresholds": {
            "green": {"label": "Traumatisme mineur", "condition": "TRISS_survival ≥ 0.90"},
            "orange": {"label": "Traumatisme modéré — surveillance étroite", "condition": "0.75 ≤ TRISS_survival < 0.90"},
            "red": {"label": "Traumatisme grave — transfert Trauma Center Niveau 1", "condition": "TRISS_survival < 0.75"},
        },
        "databases": [
            "Registre National des Traumatismes Graves (ex: Traumabase)",
            "Fiches d'intervention SMUR / Hélicoptère (REGA / HUG)",
            "Dossiers de réanimation des urgences (Trauma Center HUG)"
        ],
        "outcome_definition": "Score ISS (Injury Severity Score) réel supérieur à 15, caractérisant un traumatisme majeur nécessitant un Trauma Center de niveau 1.",
        "variables_detail": {
            "GCS": {
                "definition": "Score de Glasgow (3 à 15) évaluant l'état de conscience du patient.",
                "plugged": False,
                "source": "Examen clinique SMUR (Manquant - Dataset requis)"
            },
            "pression_arterielle_systolique": {
                "definition": "Pression artérielle systolique en mmHg.",
                "plugged": False,
                "source": "Moniteur multiparamétrique (Manquant - Dataset requis)"
            },
            "frequence_respiratoire": {
                "definition": "Fréquence respiratoire en cycles par minute.",
                "plugged": False,
                "source": "Examen clinique SMUR (Manquant - Dataset requis)"
            },
            "mecanisme_traumatisme": {
                "definition": "Catégorie du mécanisme lésionnel (chute de hauteur, accident haute cinétique, arme blanche, etc.).",
                "plugged": False,
                "source": "Régulation/Fiche d'intervention (Manquant - Dataset requis)"
            },
            "age": {
                "definition": "Âge du patient en années.",
                "plugged": False,
                "source": "Régulation (Manquant - Dataset requis)"
            },
            "ISS_estime": {
                "definition": "Estimation préhospitalière du score anatomique de gravité ISS.",
                "plugged": False,
                "source": "Algorithme clinique SMUR (Manquant - Dataset requis)"
            }
        },
        "keywords": ["Traumatisme grave", "Polytraumatisé", "Trauma Center", "Score ISS", "Transfusion massive", "Damage Control"],
        "clinical_rationale": (
            "Les traumatismes graves (accidents de la route, de montagne) nécessitent une orientation immédiate vers un Trauma Center de niveau 1 pour réduire la mortalité évitable. Une stratification précoce du risque permet de déclencher les protocoles de transfusion massive et de damage control dès la phase préhospitalière."
        ),
    },

    "clinical-deterioration-prediction": {
        "boolean_queries": [
            '("clinical deterioration" OR "early warning score" OR NEWS2 OR MEWS) AND (prehospital OR EMS OR emergency) AND (2020:2026[dp])',
            '("machine learning" OR "deep learning") AND ("clinical deterioration") AND (prehospital OR emergency)',
            '("vital signs" OR "physiological telemetry") AND (prediction OR forecasting) AND ("clinical deterioration")',
        ],
        "nl_queries": [
            "prédiction détérioration clinique NEWS2 préhospitalier urgences",
            "prehospital clinical deterioration prediction early warning score NEWS2",
            "télémétrie paramètres vitaux détérioration patient ambulance",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la prédiction de la détérioration clinique en préhospitalier. "
            "Extrayez : (1) les scores d'alerte précoce validés (NEWS2, MEWS, qSOFA) et leur performance, "
            "(2) les modèles prédictifs basés sur l'IA utilisant les séries temporelles de constantes vitales, "
            "(3) l'impact clinique de l'implémentation de ces scores en ambulance. "
            "Contexte : Grand Genève, transferts inter-hospitaliers et urgences primaires."
        ),
        "model_info": {
            "algorithm": "LSTM Recurrent Neural Network + NEWS2",
            "variables": ["frequence_cardiaque", "pression_arterielle_systolique", "saturation_O2", "frequence_respiratoire", "temperature", "statut_neurologique"],
            "output": "Probabilité de détérioration clinique (arrêt cardiorespiratoire ou admission soins intensifs) à H+2",
            "update_frequency": "Continu (télémétrie toutes les 5 minutes)",
        },
        "alert_thresholds": {
            "green": {"label": "Patient stable", "condition": "NEWS2 < 5"},
            "orange": {"label": "Risque modéré — surveillance continue", "condition": "5 ≤ NEWS2 < 7"},
            "red": {"label": "Risque critique — alerte réanimation", "condition": "NEWS2 ≥ 7"},
        },
        "databases": [
            "Données de monitorage continu préhospitalier (télémétrie Lifepak/Corpuls)",
            "Dossier Patient Informatisé des Soins Intensifs (HUG)",
            "Registre des réanimations aux Urgences"
        ],
        "outcome_definition": "Survenue d'un arrêt cardiorespiratoire ou d'une admission directe non programmée en Soins Intensifs dans les 24 heures suivant la prise en charge.",
        "variables_detail": {
            "frequence_cardiaque": {
                "definition": "Fréquence cardiaque en battements par minute.",
                "plugged": False,
                "source": "Télémétrie Corpuls (Manquant - Dataset requis)"
            },
            "pression_arterielle_systolique": {
                "definition": "Pression artérielle systolique en mmHg.",
                "plugged": False,
                "source": "Télémétrie Corpuls (Manquant - Dataset requis)"
            },
            "saturation_O2": {
                "definition": "Saturation en oxygène du sang en pourcentage (SpO2).",
                "plugged": False,
                "source": "Télémétrie Corpuls (Manquant - Dataset requis)"
            },
            "frequence_respiratoire": {
                "definition": "Fréquence respiratoire en cycles par minute.",
                "plugged": False,
                "source": "Examen clinique (Manquant - Dataset requis)"
            },
            "temperature": {
                "definition": "Température corporelle en degrés Celsius.",
                "plugged": False,
                "source": "Thermomètre préhospitalier (Manquant - Dataset requis)"
            },
            "statut_neurologique": {
                "definition": "Niveau de conscience évalué selon l'échelle ACVPU (Alerte, Confusion, Voix, Douleur, Inconscient).",
                "plugged": False,
                "source": "Examen clinique (Manquant - Dataset requis)"
            }
        }
    },

    "patient-pathway-optimization": {
        "boolean_queries": [
            '("patient pathway" OR "patient flow" OR "care pathway") AND (emergency OR EMS) AND (optimization OR simulation) AND (2020:2026[dp])',
            '("bed management" OR "hospital capacity") AND (emergency OR EMS) AND (coordination OR dispatch)',
            '("clinical pathway" OR "patient routing") AND (prehospital OR EMS) AND (decision OR optimization)',
        ],
        "nl_queries": [
            "optimisation parcours de soins urgences flux patients lits disponibles",
            "emergency patient pathway optimization bed management",
            "orientation intelligente des patients EMS HUG CHUV cliniques",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'optimisation des parcours de soins et l'orientation des patients. "
            "Extrayez : (1) les modèles d'orientation basés sur la capacité hospitalière en temps réel, "
            "(2) les algorithmes d'évitement de l'engorgement des urgences, "
            "(3) l'impact des filières directes (AVC, IDM, gériatrie) sur les délais de prise en charge. "
            "Contexte : Grand Genève, réseau hospitalier public-privé (HUG, CHUV, Clinique Générale-Beaulieu, Hôpital de la Tour, etc.)."
        ),
        "model_info": {
            "algorithm": "Graphe de décision + Programmation Linéaire",
            "variables": ["pathologie_principale", "gravite_clinique", "temps_trajet_hopitaux", "encombrement_urgences", "disponibilite_lits_specialises"],
            "output": "Recommandation d'orientation hospitalière optimale (Hôpital + Service)",
            "update_frequency": "À chaque orientation de patient",
        },
        "alert_thresholds": {
            "green": {"label": "Filière optimale disponible", "condition": "saturation_hopital_cible < 0.8"},
            "orange": {"label": "Filière engorgée — réorientation recommandée", "condition": "0.8 ≤ saturation_hopital_cible < 0.95"},
            "red": {"label": "Filière saturée — réorientation obligatoire", "condition": "saturation_hopital_cible ≥ 0.95"},
        },
        "databases": [
            "Système de suivi de l'occupation des lits hospitaliers (HUG/CHUV)",
            "Données de régulation préhospitalière SAGA",
            "Tableau de bord de tension des urgences (ex: score de NEDOCS live)"
        ],
        "outcome_definition": "Durée totale de séjour aux Urgences (Length of Stay - LOS) et taux de réadmission non programmée à 48 heures.",
        "variables_detail": {
            "pathologie_principale": {
                "definition": "Diagnostic principal suspecté en préhospitalier (ex: IDM, AVC, fracture col du fémur).",
                "plugged": False,
                "source": "Régulation / Fiche SMUR (Manquant - Dataset requis)"
            },
            "gravite_clinique": {
                "definition": "Niveau de tri de gravité clinique (1=Critique à 5=Non urgent).",
                "plugged": False,
                "source": "Tri préhospitalier (Manquant - Dataset requis)"
            },
            "temps_trajet_hopitaux": {
                "definition": "Temps de transport routier estimé en minutes vers les différents hôpitaux du réseau.",
                "plugged": True,
                "source": "OSRM API (Live)"
            },
            "encombrement_urgences": {
                "definition": "Indicateur de tension en temps réel des services d'urgences cibles (ex: nombre de patients en attente).",
                "plugged": False,
                "source": "Système d'information des urgences (Manquant - Dataset requis)"
            },
            "disponibilite_lits_specialises": {
                "definition": "Nombre de lits disponibles immédiatement dans le service de spécialité requis (USIC, UNV, Gériatrie).",
                "plugged": False,
                "source": "Gestionnaire de lits hospitaliers (Manquant - Dataset requis)"
            }
        }
    },

    "mci-victim-estimation": {
        "boolean_queries": [
            '("mass casualty" OR "mass casualty incident" OR MCI) AND (victim OR casualty) AND (estimation OR prediction OR model) AND (2020:2026[dp])',
            '(MCI OR "disaster response") AND (triage OR "victim count") AND (simulation OR algorithm)',
            '("emergency response") AND ("casualty estimation") AND (scenarios OR disaster OR accident)',
        ],
        "nl_queries": [
            "estimation nombre victimes accident catastrophe mass casualty incident",
            "mass casualty incident victim count estimation model",
            "modélisation afflux victimes attentat accident majeur Grand Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'estimation du nombre de victimes lors d'accidents de grande ampleur (MCI). "
            "Extrayez : (1) les modèles mathématiques ou algorithmes d'estimation rapide du nombre de victimes, "
            "(2) les facteurs d'échelle (type d'incident, météo, densité de population), "
            "(3) les ratios d'urgence relative (UA) vs urgence absolue (UR) observés historiquement. "
            "Contexte : Grand Genève, plans ORCA / NOVI, risques ferroviaires (CFF), aéroportuaires (GVA), autoroutiers."
        ),
        "model_info": {
            "algorithm": "Modèle d'impact physique + Monte Carlo",
            "variables": ["type_evenement", "localisation", "densite_population_heure", "conditions_meteo", "perimetre_impact"],
            "output": "Estimation du nombre total de victimes + répartition par gravité (Absolue/Relative/Décédés)",
            "update_frequency": "Instantané à la saisie de l'événement",
        },
        "alert_thresholds": {
            "green": {"label": "Incident gérable (Moyen local)", "condition": "nb_victimes_estime < 10"},
            "orange": {"label": "Incident majeur — déclenchement Plan Blanc/NOVI", "condition": "10 ≤ nb_victimes_estime < 50"},
            "red": {"label": "Catastrophe majeure — renforts nationaux/transfrontaliers", "condition": "nb_victimes_estime ≥ 50"},
        },
        "databases": [
            "Données de géolocalisation et densité de population horaire (LandScan / données mobiles)",
            "Base de données des plans d'urgence cantonaux (ORCA Genève)",
            "Données météo locales (Open-Meteo)"
        ],
        "outcome_definition": "Nombre réel de victimes secourues et admises dans les structures hospitalières du réseau transfrontalier.",
        "variables_detail": {
            "type_evenement": {
                "definition": "Nature de l'incident (accident ferroviaire, industriel SEVESO, terroriste, effondrement).",
                "plugged": False,
                "source": "Déclaration régulation (Manquant - Dataset requis)"
            },
            "localisation": {
                "definition": "Coordonnées géographiques (latitude/longitude) du point d'impact.",
                "plugged": False,
                "source": "Régulation / GPS (Manquant - Dataset requis)"
            },
            "densite_population_heure": {
                "definition": "Estimation de la population présente dans le périmètre à l'heure de l'incident (données cadastrales et de flux).",
                "plugged": False,
                "source": "Données démographiques géospatiales (Manquant - Dataset requis)"
            },
            "conditions_meteo": {
                "definition": "Vitesse et direction du vent, précipitations pour modéliser une dispersion chimique ou thermique.",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            },
            "perimetre_impact": {
                "definition": "Rayon estimé de la zone d'impact en mètres selon le type d'incident.",
                "plugged": False,
                "source": "Saisie opérationnelle (Manquant - Dataset requis)"
            }
        }
    },

    "environmental-risk-forecasting": {
        "boolean_queries": [
            '("environmental risk" OR "air pollution" OR "ozone" OR "particulate matter") AND (EMS OR "emergency calls" OR "ambulance") AND (2020:2026[dp])',
            '("extreme weather" OR "heatwave" OR "cold wave") AND (EMS OR prehospital OR emergency) AND (prediction OR forecasting)',
            '("environmental exposure") AND ("cardiorespiratoire") AND (EMS OR prehospital OR emergency)',
        ],
        "nl_queries": [
            "prévision risque environnemental pollution air particules fines appels EMS",
            "environmental risk forecasting air quality ambulance demand",
            "impact pics d'ozone et pollution sur les urgences respiratoires Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'impact des risques environnementaux (pollution, ozone, particules fines PM2.5/PM10) sur la demande EMS. "
            "Extrayez : (1) les coefficients de corrélation ou risques relatifs (RR) entre les pics de pollution et les appels pour motifs cardiorespiratoires, "
            "(2) les modèles de prévision de la demande EMS intégrant la qualité de l'air, "
            "(3) les seuils d'alerte environnementaux recommandés pour les EMS. "
            "Contexte : Grand Genève, cuvette lémanique sujette aux inversions thermiques en hiver et pics d'ozone en été."
        ),
        "model_info": {
            "algorithm": "GAM (Generalized Additive Model) + AirQuality API",
            "variables": ["PM2_5", "PM10", "Ozone", "NO2", "temperature", "historique_appels_respi_7j"],
            "output": "Indice de surcoût d'appels pour motif cardiorespiratoire à J+1 à J+3",
            "update_frequency": "Quotidienne (7h UTC)",
        },
        "alert_thresholds": {
            "green": {"label": "Risque environnemental faible", "condition": "aqi < 50"},
            "orange": {"label": "Risque modéré — hausse des appels respiratoires (+15%)", "condition": "50 ≤ aqi < 100"},
            "red": {"label": "Risque élevé — pic d'appels cardiorespiratoires (+30%)", "condition": "aqi ≥ 100"},
        },
        "databases": [
            "Réseau de surveillance de la qualité de l'air (ex: Air Genève, ATMO Auvergne-Rhône-Alpes)",
            "Données de régulation médicale 144/15 pour motifs respiratoires",
            "API Copernicus Atmospheric Monitoring Service (CAMS)"
        ],
        "outcome_definition": "Nombre quotidien d'appels EMS pour motifs cardiorespiratoires (asthme, BPCO, insuffisance cardiaque) dans le Grand Genève.",
        "variables_detail": {
            "PM2_5": {
                "definition": "Concentration de particules fines de diamètre < 2.5 µg/m³ dans l'air ambiant.",
                "plugged": True,
                "source": "Copernicus Air Quality API (Live)"
            },
            "PM10": {
                "definition": "Concentration de particules de diamètre < 10 µg/m³.",
                "plugged": True,
                "source": "Copernicus Air Quality API (Live)"
            },
            "Ozone": {
                "definition": "Concentration d'ozone (O3) en µg/m³ au niveau du sol.",
                "plugged": True,
                "source": "Copernicus Air Quality API (Live)"
            },
            "NO2": {
                "definition": "Concentration de dioxyde d'azote en µg/m³.",
                "plugged": True,
                "source": "Copernicus Air Quality API (Live)"
            },
            "temperature": {
                "definition": "Température maximale de l'air à 2 mètres.",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            },
            "historique_appels_respi_7j": {
                "definition": "Moyenne mobile sur 7 jours des appels régulés pour motifs cardiorespiratoires.",
                "plugged": False,
                "source": "Base de données SAGA 144 (Manquant - Dataset requis)"
            }
        }
    },

    "disaster-risk-assessment": {
        "boolean_queries": [
            '("disaster risk" OR "hazard assessment") AND (EMS OR emergency OR prehospital) AND (mapping OR GIS OR modeling) AND (2020:2026[dp])',
            '("natural hazard" OR "flooding" OR "landslide") AND (EMS OR emergency OR "road closure") AND (accessibility OR routing)',
            '("disaster preparedness") AND ("resource allocation") AND (EMS OR emergency OR prehospital)',
        ],
        "nl_queries": [
            "évaluation risque catastrophe naturelle accessibilité routière EMS",
            "disaster risk assessment prehospital routing road closure flooding",
            "modélisation inondation glissement terrain blocage routes Grand Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'évaluation des risques de catastrophes naturelles et leur impact sur l'accessibilité des EMS. "
            "Extrayez : (1) les méthodologies de cartographie de vulnérabilité du réseau routier pour les secours, "
            "(2) l'impact des inondations, glissements de terrain ou séismes sur les délais de réponse des ambulances, "
            "(3) les stratégies de pré-positionnement des ressources de secours. "
            "Contexte : Grand Genève, risques d'inondation de l'Arve/Rhône, glissements de terrain en Haute-Savoie."
        ),
        "model_info": {
            "algorithm": "Analyse de réseau géospatial (Network Analyst) + SIG",
            "variables": ["zones_inondables", "risques_glissement", "fermetures_routes_live", "localisation_casernes", "points_passage_alternatifs"],
            "output": "Carte d'accessibilité dynamique des secours + temps de trajet dégradé par secteur",
            "update_frequency": "Quotidienne ou à chaque alerte météo majeure",
        },
        "alert_thresholds": {
            "green": {"label": "Accessibilité normale", "condition": "routes_bloquees == 0"},
            "orange": {"label": "Accessibilité dégradée — allongement délais (+5-15 min)", "condition": "1 ≤ routes_bloquees < 5"},
            "red": {"label": "Secteurs isolés — intervention hélicoptère requise", "condition": "routes_bloquees ≥ 5"},
        },
        "databases": [
            "Système d'Information Géographique (SIG) cantonal (SITG Genève)",
            "Cartographie des dangers naturels (Suisse/France)",
            "Flux d'info trafic routier en temps réel (Viasuisse / Waze API)"
        ],
        "outcome_definition": "Allongement moyen du temps de trajet des ambulances vers les zones sinistrées par rapport au temps de référence hors catastrophe.",
        "variables_detail": {
            "zones_inondables": {
                "definition": "Polygones SIG délimitant les zones de crue centennale de l'Arve, du Rhône et de la Versoix.",
                "plugged": True,
                "source": "SITG Genève (Live)"
            },
            "risques_glissement": {
                "definition": "Carte d'aléa de glissement de terrain ou d'éboulement (zones montagneuses Haute-Savoie).",
                "plugged": True,
                "source": "Géoportail France (Live)"
            },
            "fermetures_routes_live": {
                "definition": "Liste des tronçons routiers actuellement fermés ou impraticables.",
                "plugged": False,
                "source": "API Info-Trafic / Waze (Manquant - Dataset requis)"
            },
            "localisation_casernes": {
                "definition": "Coordonnées géographiques de toutes les bases de départ d'ambulances et de pompiers.",
                "plugged": True,
                "source": "SITG / Base de données interne"
            },
            "points_passage_alternatifs": {
                "definition": "Itinéraires de contournement calculés automatiquement en cas de coupure de ponts sur l'Arve ou le Rhône.",
                "plugged": True,
                "source": "OSRM API (Live)"
            }
        }
    },

    "climate-impact-on-ems": {
        "boolean_queries": [
            '("climate change" OR "global warming") AND (EMS OR "emergency medical services" OR "ambulance") AND (demand OR volume) AND (2020:2026[dp])',
            '("long-term trend") AND ("extreme weather") AND (EMS OR emergency OR ambulance) AND (impact OR demand)',
            '("climate projection") AND ("heatwave" OR "flooding") AND (EMS OR emergency)',
        ],
        "nl_queries": [
            "impact changement climatique demande EMS ambulances long terme",
            "climate change impact emergency medical services long-term demand",
            "projections climatiques Grand Genève hausse température appels 144",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'impact à long terme du changement climatique sur l'activité des EMS. "
            "Extrayez : (1) les projections d'augmentation du volume d'appels d'urgence liées à la hausse globale des températures, "
            "(2) l'évolution de la saisonnalité des appels (ex: décalage des pics hivernaux vers des vagues de chaleur estivales), "
            "(3) les recommandations d'adaptation structurelle pour les EMS à l'horizon 2030-2050. "
            "Contexte : Grand Genève, réchauffement des Alpes, augmentation de la fréquence des canicules."
        ),
        "model_info": {
            "algorithm": "Modèle de régression multivariée + Scénarios GIEC (RCP 4.5 / 8.5)",
            "variables": ["anomalie_temperature_annuelle", "nombre_jours_canicule_an", "projection_demographique_2030", "historique_appels_annuel"],
            "output": "Projection du volume annuel d'appels d'urgence EMS à l'horizon 2030-2040",
            "update_frequency": "Annuelle",
        },
        "alert_thresholds": {
            "green": {"label": "Impact climatique soutenable", "condition": "hausse_appels_estimee < 5%"},
            "orange": {"label": "Impact modéré — besoin d'adaptation des effectifs (+10%)", "condition": "5% ≤ hausse_appels_estimee < 15%"},
            "red": {"label": "Impact critique — saturation prévisible sans restructuration majeure", "condition": "hausse_appels_estimee ≥ 15%"},
        },
        "databases": [
            "Scénarios climatiques nationaux CH2018 (Suisse) / Météo-France Drias",
            "Projections démographiques cantonales (OCSTAT Genève)",
            "Historique décennal des appels de régulation 144/15"
        ],
        "outcome_definition": "Taux de croissance annuel de la demande EMS attribuable spécifiquement aux facteurs climatiques (hors croissance démographique).",
        "variables_detail": {
            "anomalie_temperature_annuelle": {
                "definition": "Écart de la température moyenne annuelle par rapport à la normale climatologique 1991-2020.",
                "plugged": True,
                "source": "MétéoSuisse / Copernicus (Live)"
            },
            "nombre_jours_canicule_an": {
                "definition": "Nombre de jours par an où la température maximale dépasse 30°C et la minimale ne descend pas sous 20°C.",
                "plugged": True,
                "source": "Open-Meteo Historical (Live)"
            },
            "projection_demographique_2030": {
                "definition": "Estimation de la population du Grand Genève en 2030 par classe d'âge (surreprésentation des >75 ans).",
                "plugged": False,
                "source": "OCSTAT Genève (Manquant - Dataset requis)"
            },
            "historique_appels_annuel": {
                "definition": "Nombre total d'interventions d'urgence EMS par an sur les 10 dernières années.",
                "plugged": False,
                "source": "Registre d'activité EMS (Manquant - Dataset requis)"
            }
        }
    },

    "heatwave-ems-impact": {
        "boolean_queries": [
            '("heatwave" OR "extreme heat" OR "high temperature") AND (EMS OR "emergency medical services" OR "ambulance") AND (demand OR volume OR mortality) AND (2020:2026[dp])',
            '("heatwave") AND ("vulnerable population" OR elderly OR "cardiovascular") AND (EMS OR emergency)',
            '("heatwave response plan") AND (EMS OR prehospital OR emergency) AND (effectiveness OR evaluation)',
        ],
        "nl_queries": [
            "impact canicule appels ambulances urgences personnes âgées",
            "heatwave impact EMS emergency calls elderly vulnerable",
            "plan canicule Genève Haute-Savoie efficacité régulation 144",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'impact immédiat des canicules (vagues de chaleur extrême) sur les appels EMS. "
            "Extrayez : (1) le délai de latence (lag) entre le début du pic de chaleur et la hausse des appels (généralement 1 à 3 jours), "
            "(2) les pathologies les plus fréquentes (déshydratation, coup de chaleur, insuffisance rénale, troubles cognitifs), "
            "(3) l'efficacité des plans canicule préventifs sur la réduction de la surcharge des EMS. "
            "Contexte : Grand Genève, populations vulnérables (personnes âgées isolées en milieu urbain)."
        ),
        "model_info": {
            "algorithm": "Distributed Lag Non-linear Models (DLNM) + Random Forest",
            "variables": ["temperature_max_24h", "temperature_min_24h", "duree_vague_chaleur", "taux_humidite", "indice_chaleur_apparent", "appels_canicule_baseline"],
            "output": "Estimation du surcoût d'appels d'urgence à 24h et 48h lié à la chaleur",
            "update_frequency": "Quotidienne (toutes les 12h en été)",
        },
        "alert_thresholds": {
            "green": {"label": "Pas d'impact thermique", "condition": "temperature_max_24h < 30°C"},
            "orange": {"label": "Vigilance Canicule — hausse des appels (+20%)", "condition": "30°C ≤ temperature_max_24h < 34°C"},
            "red": {"label": "Alerte Canicule Extrême — hausse critique (+40% appels)", "condition": "temperature_max_24h ≥ 34°C"},
        },
        "databases": [
            "Données météo quotidiennes (MétéoSuisse / Météo-France)",
            "Base de données de régulation 144 (motifs de déshydratation, hyperthermie, malaise)",
            "Système de surveillance de la mortalité en temps réel (MOMO Suisse)"
        ],
        "outcome_definition": "Excès d'appels d'urgence régulés par jour par rapport à la moyenne historique pour un même jour de l'année hors canicule.",
        "variables_detail": {
            "temperature_max_24h": {
                "definition": "Température maximale enregistrée au cours des dernières 24 heures en degrés Celsius.",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            },
            "temperature_min_24h": {
                "definition": "Température minimale enregistrée (refroidissement nocturne capital pour la récupération de l'organisme).",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            },
            "duree_vague_chaleur": {
                "definition": "Nombre de jours consécutifs où la température maximale dépasse le seuil de 31°C.",
                "plugged": True,
                "source": "Calculé à partir de l'historique Open-Meteo"
            },
            "taux_humidite": {
                "definition": "Humidité relative moyenne en journée (aggrave l'inconfort thermique et limite la sudation).",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            },
            "indice_chaleur_apparent": {
                "definition": "Humidex ou température ressentie combinant chaleur et humidité.",
                "plugged": True,
                "source": "Calculé (Formule Humidex)"
            },
            "appels_canicule_baseline": {
                "definition": "Moyenne historique des appels d'urgence régulés pour un jour d'été normal.",
                "plugged": False,
                "source": "Historique de régulation SAGA (Manquant - Dataset requis)"
            }
        }
    },

    "emergency-call-qualification": {
        "boolean_queries": [
            '("emergency call" OR "triage call") AND (qualification OR classification OR "natural language processing" OR NLP) AND (2020:2026[dp])',
            '("speech-to-text" OR transcription) AND ("emergency call" OR dispatch) AND (accuracy OR performance)',
            '("decision support system" OR DSS) AND ("emergency dispatcher" OR calltaker) AND (qualification OR triage)',
        ],
        "nl_queries": [
            "qualification automatique appels urgences transcription NLP IA",
            "emergency call qualification speech-to-text NLP dispatcher",
            "aide à la décision régulateur 144 transcription temps réel",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la qualification automatique des appels d'urgence par traitement du langage naturel (NLP). "
            "Extrayez : (1) la précision (accuracy, F1-score) des algorithmes de classification des motifs d'appels d'urgence, "
            "(2) l'apport de la transcription de la parole en temps réel (Speech-to-Text) pour les régulateurs, "
            "(3) les risques d'erreurs ou de biais identifiés lors de l'utilisation de l'IA à la régulation. "
            "Contexte : Grand Genève, régulation bilingue (Français/Anglais majoritaires), jargon médical et expressions de détresse."
        ),
        "model_info": {
            "algorithm": "Whisper Live + CamemBERT Classifier",
            "variables": ["audio_stream_call", "mots_cles_detectes", "ton_de_la_voix", "temps_appel", "historique_appelant"],
            "output": "Catégorie de motif suspectée (ex: Douleur thoracique, AVC, Traumatisme) + Niveau de gravité estimé",
            "update_frequency": "Temps réel pendant l'appel (latence < 2s)",
        },
        "alert_thresholds": {
            "green": {"label": "Qualification normale", "condition": "confidence_score ≥ 0.80"},
            "orange": {"label": "Incertitude — poser questions complémentaires", "condition": "0.50 ≤ confidence_score < 0.80"},
            "red": {"label": "Divergence forte — régulation manuelle prioritaire", "condition": "confidence_score < 0.50"},
        },
        "databases": [
            "Enregistrements audio anonymisés de la centrale d'appels 144/15",
            "Dossiers de régulation médicale SAGA",
            "Lexique ontologique d'urgence médicale (SNOMED-CT / CIM-10)"
        ],
        "outcome_definition": "Exactitude de la classification automatique du motif d'appel par rapport au diagnostic final posé par l'équipe médicale de terrain.",
        "variables_detail": {
            "audio_stream_call": {
                "definition": "Flux audio brut de l'appel téléphonique entrant pour transcription en temps réel.",
                "plugged": False,
                "source": "Serveur de téléphonie IP centrale 144 (Manquant - Dataset requis)"
            },
            "mots_cles_detectes": {
                "definition": "Liste des termes cliniques clés identifiés dans la transcription (ex: 'poitrine', 'bras gauche', 'étouffe').",
                "plugged": False,
                "source": "Module NLP CamemBERT (Manquant - Dataset requis)"
            },
            "ton_de_la_voix": {
                "definition": "Analyse acoustique du ton de la voix de l'appelant (indice d'anxiété, de panique ou de dyspnée).",
                "plugged": False,
                "source": "Analyseur de spectre audio (Manquant - Dataset requis)"
            },
            "temps_appel": {
                "definition": "Durée écoulée depuis le décroché de l'appel en secondes.",
                "plugged": True,
                "source": "Horloge système"
            },
            "historique_appelant": {
                "definition": "Nombre d'appels passés par le même numéro de téléphone au cours des 30 derniers jours (détection d'appelants fréquents).",
                "plugged": False,
                "source": "Base SAGA (Manquant - Dataset requis)"
            }
        },
        "keywords": ["Régulation médicale", "Speech-to-Text", "NLP", "CamemBERT", "Classification des motifs", "Transcription temps réel"],
        "clinical_rationale": (
            "La centrale de régulation reçoit des flux d'appels massifs sous haute tension. L'utilisation du traitement automatique du langage naturel (NLP) et de la transcription en temps réel assiste le régulateur en extrayant instantanément les signaux faibles et les symptômes critiques, réduisant le temps de qualification de l'appel."
        ),
    },

    "call-prioritization": {
        "boolean_queries": [
            '("call prioritization" OR "dispatch prioritization") AND (EMS OR emergency OR ambulance) AND (algorithm OR "decision support") AND (2020:2026[dp])',
            '("under-triage" OR "over-triage") AND ("call prioritization" OR dispatch) AND (prehospital OR EMS)',
            '("priority dispatch system" OR MPDS) AND (accuracy OR outcomes) AND (EMS OR emergency)',
        ],
        "nl_queries": [
            "priorisation appels urgences répartition ambulances algorithme",
            "emergency call prioritization dispatch triage algorithm",
            "triage et priorisation des missions d'urgence 144 Grand Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la priorisation des appels et des missions d'urgence dans les EMS. "
            "Extrayez : (1) les algorithmes de priorisation validés (ex: MPDS, critères de tri de la centrale), "
            "(2) l'impact de la priorisation sur les délais de réponse pour les urgences vitales (P1), "
            "(3) les taux de sur-triage (over-triage) et sous-triage (under-triage) acceptables. "
            "Contexte : Grand Genève, priorisation des ambulances en période de forte tension."
        ),
        "model_info": {
            "algorithm": "Arbre de décision XGBoost + Règles expertes cliniques",
            "variables": ["age", "motif_appel", "constantes_vitales_declarees", "nombre_ambulances_dispo", "temps_attente_moyen"],
            "output": "Niveau de priorité de la mission (P1: Urgence vitale immédiate, P2: Urgence relative, P3: Transport différable)",
            "update_frequency": "Instantané à la saisie de l'appel",
        },
        "alert_thresholds": {
            "green": {"label": "Priorité adéquate", "condition": "risque_sous_triage < 5%"},
            "orange": {"label": "Risque de sous-triage modéré — réévaluation par médecin régulateur", "condition": "5% ≤ risque_sous_triage < 15%"},
            "red": {"label": "Risque de sous-triage élevé — envoi immédiat ambulance P1", "condition": "risque_sous_triage ≥ 15%"},
        },
        "databases": [
            "Base de données des fiches de régulation SAGA (144)",
            "Registre des diagnostics d'arrivée aux urgences hospitalières",
            "Statut de disponibilité des ambulances en temps réel"
        ],
        "outcome_definition": "Taux de sous-triage réel (missions classées P2/P3 ayant nécessité un geste de réanimation ou une admission directe en Soins Intensifs).",
        "variables_detail": {
            "age": {
                "definition": "Âge du patient en années.",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "motif_appel": {
                "definition": "Code de motif d'appel standardisé selon la nomenclature de régulation.",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "constantes_vitales_declarees": {
                "definition": "Paramètres physiologiques décrits par l'appelant (conscience, respiration, douleur).",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "nombre_ambulances_dispo": {
                "definition": "Nombre d'ambulances actuellement libres et prêtes au départ dans le canton.",
                "plugged": False,
                "source": "Système de répartition des ambulances (Manquant - Dataset requis)"
            },
            "temps_attente_moyen": {
                "definition": "Temps d'attente moyen en minutes pour les appels de basse priorité.",
                "plugged": True,
                "source": "Calculé à partir des données de file d'attente"
            }
        },
        "keywords": ["Priorisation", "File d'attente", "Triage des appels", "Sous-triage", "Algorithme de tri", "Urgence vitale"],
        "clinical_rationale": (
            "Lors de pics d'activité, la priorisation automatisée des appels entrants garantit que les détresses vitales absolues (infarctus, arrêt cardiaque) soient traitées en priorité absolue par les régulateurs, évitant les pertes de chances liées à l'attente téléphonique."
        ),
    },

    "mass-casualty-triage": {
        "boolean_queries": [
            '("mass casualty triage" OR "disaster triage") AND (START OR SALT OR "triage protocol") AND (accuracy OR survival) AND (2020:2026[dp])',
            '("mass casualty triage") AND ("decision support" OR algorithm OR digital) AND (prehospital OR EMS)',
            '("disaster medicine") AND ("triage accuracy") AND (simulation OR training OR incident)',
        ],
        "nl_queries": [
            "triage catastrophe mass casualty START SALT protocoles efficacité",
            "mass casualty triage START SALT protocol accuracy EMS",
            "triage des victimes d'accident majeur plan ORCA HUG",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur le triage lors d'incidents de masse (MCI). "
            "Extrayez : (1) les protocoles de triage validés (START, SALT, CareFlight) et leur sensibilité/spécificité, "
            "(2) l'impact des outils de triage numériques (e-triage, codes QR, puces RFID) sur la vitesse d'évacuation, "
            "(3) les erreurs de triage les plus fréquentes commises par les premiers intervenants. "
            "Contexte : Grand Genève, exercices intercantonaux et transfrontaliers (CH/FR)."
        ),
        "model_info": {
            "algorithm": "Arbre de décision clinique SALT (Sort, Assess, Lifesaving Interventions, Treatment/Transport)",
            "variables": ["reponse_ordres_simples", "respiration_spontanee", "frequence_respiratoire", "pouls_radial", "controle_hemorragie"],
            "output": "Code couleur de triage (Rouge: Extrême urgence, Jaune: Urgence, Vert: Blessé léger, Noir: Décédé)",
            "update_frequency": "À chaque évaluation de victime (temps réel)",
        },
        "alert_thresholds": {
            "green": {"label": "Blessé léger (Vert)", "condition": "priorite_triage == 'Vert'"},
            "orange": {"label": "Urgence (Jaune) — évacuation secondaire", "condition": "priorite_triage == 'Jaune'"},
            "red": {"label": "Extrême Urgence (Rouge) — évacuation prioritaire", "condition": "priorite_triage == 'Rouge'"},
        },
        "databases": [
            "Système de gestion numérique des victimes de catastrophe (ex: IVENA / SanQA)",
            "Registre des exercices de catastrophe cantonaux",
            "Dossiers médicaux des admissions post-catastrophe"
        ],
        "outcome_definition": "Précision du triage initial de terrain par rapport au diagnostic médical approfondi posé au Poste Médical Avancé (PMA).",
        "variables_detail": {
            "reponse_ordres_simples": {
                "definition": "Capacité de la victime à obéir à des ordres simples (ex: 'serrez-moi la main', 'marchez vers moi').",
                "plugged": False,
                "source": "Évaluation premier intervenant (Manquant - Dataset requis)"
            },
            "respiration_spontanee": {
                "definition": "Présence ou absence de mouvements respiratoires spontanés.",
                "plugged": False,
                "source": "Évaluation premier intervenant (Manquant - Dataset requis)"
            },
            "frequence_respiratoire": {
                "definition": "Fréquence respiratoire de la victime (seuil critique à 30 cycles/min).",
                "plugged": False,
                "source": "Évaluation premier intervenant (Manquant - Dataset requis)"
            },
            "pouls_radial": {
                "definition": "Présence du pouls radial ou temps de recoloration cutanée (seuil critique à 2 secondes).",
                "plugged": False,
                "source": "Évaluation premier intervenant (Manquant - Dataset requis)"
            },
            "controle_hemorragie": {
                "definition": "Présence d'une hémorragie externe massive nécessitant la pose immédiate d'un garrot.",
                "plugged": False,
                "source": "Évaluation premier intervenant (Manquant - Dataset requis)"
            }
        }
    },

    "undertriage-detection": {
        "boolean_queries": [
            '("under-triage" OR undertriage) AND (EMS OR prehospital OR emergency) AND (detection OR prediction OR model) AND (2020:2026[dp])',
            '("under-triage") AND ("trauma center" OR "geriatric triage" OR "cardiac arrest") AND (outcomes OR mortality)',
            '("machine learning" OR "deep learning") AND (undertriage OR "under-triage") AND (EMS OR emergency)',
        ],
        "nl_queries": [
            "détection sous-triage urgences ambulances taux de mortalité",
            "undertriage detection machine learning EMS trauma geriatric",
            "analyse rétrospective du sous-triage des urgences vitales Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la détection et la prévention du sous-triage (undertriage) dans les EMS. "
            "Extrayez : (1) les définitions mathématiques et cliniques du sous-triage (ex: formule de Cribari), "
            "(2) les populations les plus à risque de sous-triage (personnes âgées, traumatismes fermés, minorités), "
            "(3) les modèles d'IA capables de détecter en temps réel un risque de sous-triage à la régulation. "
            "Contexte : Grand Genève, sous-triage des traumatismes gériatriques et des syndromes coronariens atypiques."
        ),
        "model_info": {
            "algorithm": "Forêt d'arbres décisionnels (Random Forest) + Analyse de texte NLP",
            "variables": ["age", "sexe", "motif_appel_texte", "premiere_priorite_attribuee", "constantes_vitales_SMUR", "diagnostic_final_urgences"],
            "output": "Probabilité que le patient ait été sous-trié à la régulation (ex: classé P2 au lieu de P1)",
            "update_frequency": "Quotidienne (analyse rétrospective automatisée)",
        },
        "alert_thresholds": {
            "green": {"label": "Taux de sous-triage conforme", "condition": "taux_sous_triage < 5%"},
            "orange": {"label": "Alerte sous-triage modéré — besoin de révision des protocoles", "condition": "5% ≤ taux_sous_triage < 10%"},
            "red": {"label": "Alerte sous-triage critique — audit immédiat requis", "condition": "taux_sous_triage ≥ 10%"},
        },
        "databases": [
            "Base croisée SAGA (Régulation) et DPI des urgences hospitalières (HUG/CHUV)",
            "Registre des audits cliniques de régulation",
            "Base de données de facturation des urgences (codes CIM-10)"
        ],
        "outcome_definition": "Proportion de patients régulés en priorité basse (P2/P3) ayant présenté un critère de gravité extrême à l'arrivée (mortalité précoce, réanimation, bloc immédiat).",
        "variables_detail": {
            "age": {
                "definition": "Âge du patient en années.",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "sexe": {
                "definition": "Sexe biologique du patient (les femmes présentent souvent des symptômes atypiques d'IDM).",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "motif_appel_texte": {
                "definition": "Transcription textuelle des premières secondes de l'appel d'urgence.",
                "plugged": False,
                "source": "Whisper STT (Manquant - Dataset requis)"
            },
            "premiere_priorite_attribuee": {
                "definition": "Niveau de priorité initialement affecté à la mission par le régulateur.",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "constantes_vitales_SMUR": {
                "definition": "Premières constantes vitales réelles mesurées par l'équipe de terrain.",
                "plugged": False,
                "source": "Fiche Corpuls (Manquant - Dataset requis)"
            },
            "diagnostic_final_urgences": {
                "definition": "Diagnostic de sortie des urgences codé selon la CIM-10.",
                "plugged": False,
                "source": "DPI HUG/CHUV (Manquant - Dataset requis)"
            }
        }
    },

    "triage-support": {
        "boolean_queries": [
            '("triage support" OR "clinical decision support" OR CDSS) AND (EMS OR emergency OR triage) AND (2020:2026[dp])',
            '("triage algorithm" OR "triage software") AND (emergency OR EMS) AND (accuracy OR outcomes)',
            '("triage support") AND ("machine learning" OR "artificial intelligence") AND (emergency OR EMS)',
        ],
        "nl_queries": [
            "aide au triage urgences algorithme logiciel décisionnel",
            "clinical decision support system triage emergency EMS",
            "logiciel d'aide à la régulation et au triage médical 144",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur les systèmes d'aide à la décision clinique (CDSS) pour le triage aux urgences et en régulation. "
            "Extrayez : (1) la réduction du temps de triage grâce aux CDSS, "
            "(2) l'amélioration de la concordance inter-observateur (fiabilité du tri), "
            "(3) les fonctionnalités clés des logiciels de tri les plus performants. "
            "Contexte : Grand Genève, harmonisation des critères de tri entre les centrales de Haute-Savoie (15) et de Genève (144)."
        ),
        "model_info": {
            "algorithm": "Réseau de neurones profond (Multi-Layer Perceptron) + Ontologie médicale",
            "variables": ["symptome_principal", "antecedents_medicaux", "age", "frequence_cardiaque", "saturation_O2", "douleur_score"],
            "output": "Recommandation de niveau de tri de gravité (Échelle suisse du tri - EST ou échelle de Rouen)",
            "update_frequency": "Instantané au cours de la saisie des symptômes",
        },
        "alert_thresholds": {
            "green": {"label": "Tri assisté fiable", "condition": "discrepance_tri_expert < 10%"},
            "orange": {"label": "Incohérence mineure — réévaluation recommandée", "condition": "10% ≤ discrepance_tri_expert < 20%"},
            "red": {"label": "Incohérence majeure — validation obligatoire par médecin", "condition": "discrepance_tri_expert ≥ 20%"},
        },
        "databases": [
            "Base de données de l'Échelle Suisse du Tri (EST)",
            "Dossiers de régulation médicale SAGA",
            "Historique des décisions de triage validées par des experts"
        ],
        "outcome_definition": "Concordance (coefficient Kappa de Cohen) entre le niveau de tri suggéré par l'outil d'IA et le niveau de tri final validé par le médecin régulateur.",
        "variables_detail": {
            "symptome_principal": {
                "definition": "Plainte principale verbalisée par le patient ou son entourage (ex: dyspnée, céphalée intense).",
                "plugged": False,
                "source": "Saisie régulateur (Manquant - Dataset requis)"
            },
            "antecedents_medicaux": {
                "definition": "Comorbidités majeures du patient (diabète, insuffisance cardiaque, cancer).",
                "plugged": False,
                "source": "Dossier patient partagé / CARADOC (Manquant - Dataset requis)"
            },
            "age": {
                "definition": "Âge du patient en années.",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "frequence_cardiaque": {
                "definition": "Fréquence cardiaque estimée ou mesurée.",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "saturation_O2": {
                "definition": "Saturation en oxygène si mesurée par un oxymètre personnel.",
                "plugged": False,
                "source": "Déclaration patient (Manquant - Dataset requis)"
            },
            "douleur_score": {
                "definition": "Intensité de la douleur évaluée sur une échelle visuelle analogique de 0 à 10.",
                "plugged": False,
                "source": "Interrogatoire (Manquant - Dataset requis)"
            }
        },
        "keywords": ["Triage urgences", "Infirmier d'accueil", "Échelle de tri", "EST", "Orientation patient", "Concordance de tri"],
        "clinical_rationale": (
            "À l'arrivée aux urgences, le tri clinique par l'infirmier organisateur d'accueil (IOA) détermine le délai maximal d'attente sécuritaire. L'assistance par IA améliore la concordance inter-observateur, réduit le risque d'erreur de tri et anticipe les besoins d'hospitalisation d'aval."
        ),
    },

    "dispatch-decision-support": {
        "boolean_queries": [
            '("dispatch decision support" OR "emergency dispatch") AND (algorithm OR "machine learning" OR "decision support") AND (2020:2026[dp])',
            '("ambulance dispatch" OR "resource dispatch") AND (optimization OR "decision support") AND (EMS OR emergency)',
            '("dispatch decision") AND ("response time" OR outcomes) AND (EMS OR prehospital)',
        ],
        "nl_queries": [
            "aide à la décision d'envoi ambulance régulation EMS",
            "dispatch decision support system ambulance emergency dispatch",
            "optimisation du choix du moyen de secours 144 Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur les systèmes d'aide à la décision d'envoi (dispatch) pour les régulateurs EMS. "
            "Extrayez : (1) les modèles d'optimisation du choix de la ressource (ambulance de base vs SMUR vs hélicoptère), "
            "(2) l'impact de l'envoi ciblé du SMUR sur la survie des patients en détresse vitale, "
            "(3) la réduction des envois inutiles (over-dispatch) grâce aux outils d'aide à la décision. "
            "Contexte : Grand Genève, régulation bilingue, coordination SMUR HUG / SMUR Haute-Savoie."
        ),
        "model_info": {
            "algorithm": "Modèle de choix discret (Multinomial Logit) + Forêt d'arbres",
            "variables": ["gravite_suspectee", "localisation_incident", "dispo_ambulances_proches", "dispo_smur_proche", "temps_arrivee_estime"],
            "output": "Moyen de secours recommandé à envoyer immédiatement (Ambulance seule, Ambulance + SMUR, Hélicoptère)",
            "update_frequency": "Instantané à la qualification de l'appel",
        },
        "alert_thresholds": {
            "green": {"label": "Ressource optimale disponible", "condition": "temps_arrivee_smur_estime < 15 min"},
            "orange": {"label": "SMUR local indisponible — envoi SMUR secondaire", "condition": "15 min ≤ temps_arrivee_smur_estime < 25 min"},
            "red": {"label": "Pas de SMUR disponible à moins de 25 min — envoi hélicoptère / ambulance seule", "condition": "temps_arrivee_smur_estime ≥ 25 min"},
        },
        "databases": [
            "Système de répartition assistée par ordinateur (Techwan / TechCAD)",
            "Flux de géolocalisation des véhicules de secours (GPS live)",
            "API de calcul d'itinéraire routier prédictif (trafic historique)"
        ],
        "outcome_definition": "Taux d'adéquation de l'envoi (proportion de missions où le niveau de médicalisation envoyé correspondait exactement aux besoins cliniques réels du patient).",
        "variables_detail": {
            "gravite_suspectee": {
                "definition": "Niveau d'urgence clinique estimé à l'appel (P1 à P3).",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "localisation_incident": {
                "definition": "Coordonnées géographiques précises du lieu d'intervention.",
                "plugged": False,
                "source": "TechCAD (Manquant - Dataset requis)"
            },
            "dispo_ambulances_proches": {
                "definition": "Nombre d'ambulances disponibles dans un rayon de 10 km.",
                "plugged": False,
                "source": "TechCAD (Manquant - Dataset requis)"
            },
            "dispo_smur_proche": {
                "definition": "Statut de disponibilité de l'équipe médicale SMUR la plus proche.",
                "plugged": False,
                "source": "TechCAD (Manquant - Dataset requis)"
            },
            "temps_arrivee_estime": {
                "definition": "Temps de trajet estimé en minutes pour le premier moyen de secours sur les lieux.",
                "plugged": True,
                "source": "OSRM API (Live)"
            }
        },
        "keywords": ["Aide à la décision", "Dispatch", "Moyen de secours", "SMUR transfrontalier", "Adéquation d'envoi", "Over-dispatch"],
        "clinical_rationale": (
            "Le choix de la ressource à envoyer (ambulance simple, équipe médicale SMUR, hélicoptère) est crucial. Un outil d'aide à la décision permet d'optimiser l'adéquation d'envoi, évitant la sur-médicalisation inutile (over-dispatch) tout en sécurisant l'envoi rapide d'équipes médicales pour les cas critiques."
        ),
    },

    "response-time-optimization": {
        "boolean_queries": [
            '("response time" OR "travel time") AND (EMS OR emergency OR ambulance) AND (optimization OR prediction) AND (2020:2026[dp])',
            '(" ambulance relocation" OR "dynamic deployment") AND ("response time") AND (EMS OR emergency)',
            '("traffic congestion" OR "routing algorithm") AND (emergency vehicle OR EMS OR ambulance) AND ("response time")',
        ],
        "nl_queries": [
            "optimisation temps de réponse ambulances déploiement dynamique",
            "ambulance response time optimization dynamic deployment",
            "réduction délai d'arrivée ambulances trafic Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'optimisation du temps de réponse des ambulances. "
            "Extrayez : (1) les algorithmes de déploiement dynamique (relocalisation préventive des ambulances en attente), "
            "(2) l'impact de l'info trafic en temps réel sur les itinéraires d'urgence, "
            "(3) l'impact clinique réel du temps de réponse sur la survie des patients (courbe d'efficacité temps-dépendante). "
            "Contexte : Grand Genève, congestion du centre-ville, goulets d'étranglement des ponts sur le lac."
        ),
        "model_info": {
            "algorithm": "Algorithme génétique + OSRM Traffic Routing",
            "variables": ["heure_depart", "coordonnees_depart", "coordonnees_arrivee", "densite_trafic_live", "conditions_meteo"],
            "output": "Itinéraire d'urgence optimal + Temps de trajet estimé en minutes",
            "update_frequency": "Temps réel à l'envoi de la mission",
        },
        "alert_thresholds": {
            "green": {"label": "Temps de réponse optimal", "condition": "temps_reponse_estime < 8 min"},
            "orange": {"label": "Temps de réponse limite (congestion)", "condition": "8 min ≤ temps_reponse_estime < 15 min"},
            "red": {"label": "Temps de réponse critique — risque de perte de chance", "condition": "temps_reponse_estime ≥ 15 min"},
        },
        "databases": [
            "Système d'Information Géographique routier (SITG / TomTom API)",
            "Historique des temps de trajet réels des ambulances",
            "Flux de trafic en temps réel de l'office cantonal des transports"
        ],
        "outcome_definition": "Temps de réponse réel (délai entre la validation de l'appel à la centrale et l'arrivée de l'ambulance sur les lieux de l'incident).",
        "variables_detail": {
            "heure_depart": {
                "definition": "Heure exacte de départ du véhicule de sa base ou de sa position d'attente.",
                "plugged": True,
                "source": "Horloge système"
            },
            "coordonnees_depart": {
                "definition": "Latitude et longitude de la position de départ de l'ambulance.",
                "plugged": False,
                "source": "GPS ambulance (Manquant - Dataset requis)"
            },
            "coordonnees_arrivee": {
                "definition": "Latitude et longitude du lieu d'intervention.",
                "plugged": False,
                "source": "TechCAD (Manquant - Dataset requis)"
            },
            "densite_trafic_live": {
                "definition": "Indice de congestion du trafic en temps réel sur l'itinéraire (0=Fluide à 1=Saturé).",
                "plugged": False,
                "source": "Waze API (Manquant - Dataset requis)"
            },
            "conditions_meteo": {
                "definition": "Présence de pluie, neige ou brouillard ralentissant la vitesse de progression.",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            }
        },
        "keywords": ["Temps de réponse", "Routage dynamique", "Trafic temps réel", "Itinéraire d'urgence", "Priorité aux feux"],
        "clinical_rationale": (
            "Le temps de réponse des secours est un déterminant majeur de la survie dans les urgences temps-dépendantes. L'optimisation des itinéraires via des algorithmes prédictifs prenant en compte la congestion urbaine et les goulots d'étranglement transfrontaliers permet de gagner des minutes précieuses."
        ),
    },

    "ambulance-dispatch-optimization": {
        "boolean_queries": [
            '("ambulance dispatch" OR "vehicle allocation") AND (optimization OR algorithm OR "integer programming") AND (2020:2026[dp])',
            '("coverage model" OR "maximal covering location") AND (ambulance OR EMS OR prehospital) AND (2020:2026[dp])',
            '("dynamic ambulance allocation") AND ("response time" OR coverage) AND (EMS OR emergency)',
        ],
        "nl_queries": [
            "optimisation répartition ambulances algorithme couverture spatiale",
            "ambulance dispatch optimization maximal covering location model",
            "modèle de couverture et positionnement des ambulances Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'optimisation de la répartition et du positionnement des ambulances. "
            "Extrayez : (1) les modèles mathématiques de couverture maximale (MCLP, DSM), "
            "(2) l'efficacité des stratégies de relocalisation dynamique en temps réel pour maintenir la couverture, "
            "(3) les critères de choix de l'ambulance à envoyer (proximité géographique vs disponibilité future). "
            "Contexte : Grand Genève, répartition de ~40 ambulances réparties sur une dizaine de bases privées et publiques."
        ),
        "model_info": {
            "algorithm": "DSM (Dynamic Double Standard Model) + Programmation Linéaire en Nombres Entiers",
            "variables": ["ambulances_disponibles_total", "zones_sous_couvertes", "probabilite_appels_secteur", "temps_relocalisation_moyen"],
            "output": "Recommandation de relocalisation préventive pour l'ambulance X vers la base d'attente Y",
            "update_frequency": "Toutes les 5 minutes",
        },
        "alert_thresholds": {
            "green": {"label": "Couverture optimale du territoire", "condition": "couverture_population_10min ≥ 95%"},
            "orange": {"label": "Couverture dégradée — relocalisation requise", "condition": "85% ≤ couverture_population_10min < 95%"},
            "red": {"label": "Couverture critique — risque majeur de retard de secours", "condition": "couverture_population_10min < 85%"},
        },
        "databases": [
            "Système de suivi d'activité TechCAD",
            "Données démographiques géolocalisées à haute résolution (SITG)",
            "Historique de répartition spatio-temporelle des appels d'urgence"
        ],
        "outcome_definition": "Taux de couverture de la population (proportion de la population résidente pouvant être atteinte par au moins une ambulance disponible en moins de 10 minutes).",
        "variables_detail": {
            "ambulances_disponibles_total": {
                "definition": "Nombre total d'ambulances opérationnelles et non engagées sur une mission à l'instant t.",
                "plugged": False,
                "source": "TechCAD (Manquant - Dataset requis)"
            },
            "zones_sous_couvertes": {
                "definition": "Secteurs géographiques où aucune ambulance disponible ne peut arriver en moins de 10 minutes.",
                "plugged": False,
                "source": "Calculé par SIG (Manquant - Dataset requis)"
            },
            "probabilite_appels_secteur": {
                "definition": "Probabilité d'apparition d'un appel d'urgence dans le secteur au cours de la prochaine heure (modèle prédictif).",
                "plugged": False,
                "source": "Modèle de demande historique (Manquant - Dataset requis)"
            },
            "temps_relocalisation_moyen": {
                "definition": "Temps de trajet estimé pour déplacer une ambulance disponible vers une zone sous-couverte.",
                "plugged": True,
                "source": "OSRM API (Live)"
            }
        },
        "keywords": ["Gestion de flotte", "Couverture opérationnelle", "Relocalisation dynamique", "Modèle MCLP", "Positionnement préventif"],
        "clinical_rationale": (
            "Plutôt que d'attendre passivement dans les bases, les ambulances peuvent être relocalisées préventivement en fonction des risques prédictifs de demande. Cela permet de maintenir une couverture territoriale homogène et de garantir un temps d'accès inférieur à 10 minutes pour toute la population."
        ),
    },

    "staffing-level-prediction": {
        "boolean_queries": [
            '("staffing level" OR "workforce planning" OR "shift scheduling") AND (EMS OR emergency OR ambulance) AND (prediction OR forecasting) AND (2020:2026[dp])',
            '("workload forecasting") AND (EMS OR prehospital OR emergency) AND (staffing OR scheduling)',
            '("queueing theory" OR simulation) AND (EMS OR emergency dispatch) AND (staffing OR "personnel")',
        ],
        "nl_queries": [
            "prédiction effectifs ambulances planification gardes planification",
            "EMS staffing level prediction shift scheduling queueing theory",
            "planification des équipes d'ambulanciers Genève pics d'activité",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la prédiction des besoins en effectifs (staffing) et la planification des gardes dans les EMS. "
            "Extrayez : (1) les modèles de prévision de la charge de travail (workload) à court et moyen terme, "
            "(2) l'application de la théorie des files d'attente (modèles Erlang) pour dimensionner le nombre d'ambulances requises, "
            "(3) l'impact du sous-effectif sur l'épuisement professionnel (burnout) et les délais de réponse. "
            "Contexte : Grand Genève, gestion des plannings d'équipes de secours 24/7."
        ),
        "model_info": {
            "algorithm": "Modèle Erlang-C + Série Temporelle LSTM",
            "variables": ["prevision_appels_heure", "duree_moyenne_mission", "taux_absence_prevu", "effectif_actuel_dispo", "seuil_qualite_service"],
            "output": "Nombre d'équipes d'ambulances requises par heure pour garantir un temps de réponse < 10 min à 90%",
            "update_frequency": "Hebdomadaire (planification à S+1 et S+2)",
        },
        "alert_thresholds": {
            "green": {"label": "Effectifs suffisants", "condition": "adequation_effectif ≥ 1.0"},
            "orange": {"label": "Sous-effectif modéré — risque de dépassement des délais", "condition": "0.85 ≤ adequation_effectif < 1.0"},
            "red": {"label": "Sous-effectif critique — rappel de personnel d'urgence requis", "condition": "adequation_effectif < 0.85"},
        },
        "databases": [
            "Logiciel de gestion des ressources humaines et des plannings (ex: Polypoint)",
            "Historique d'activité de régulation SAGA",
            "Statistiques de durée de prise en charge des missions"
        ],
        "outcome_definition": "Taux de conformité des effectifs planifiés par rapport aux besoins réels calculés rétrospectivement pour chaque heure de la journée.",
        "variables_detail": {
            "prevision_appels_heure": {
                "definition": "Nombre d'appels d'urgence prévu pour l'heure cible (généré par le modèle de prévision de demande).",
                "plugged": False,
                "source": "Modèle de demande (Manquant - Dataset requis)"
            },
            "duree_moyenne_mission": {
                "definition": "Durée moyenne d'une intervention d'ambulance de l'appel au retour à la base (généralement 60 à 90 minutes).",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "taux_absence_prevu": {
                "definition": "Taux moyen d'absence maladie et congés estimé pour la période.",
                "plugged": False,
                "source": "Logiciel RH (Manquant - Dataset requis)"
            },
            "effectif_actuel_dispo": {
                "definition": "Nombre d'ambulanciers actuellement inscrits au planning de garde pour l'heure cible.",
                "plugged": False,
                "source": "Logiciel RH (Manquant - Dataset requis)"
            },
            "seuil_qualite_service": {
                "definition": "Objectif de qualité de service fixé par le contrat de prestations (ex: 90% des appels P1 servis en moins de 10 min).",
                "plugged": True,
                "source": "Réglementation cantonale (Fixe: 90%)"
            }
        }
    },

    "hospital-capacity-forecasting": {
        "boolean_queries": [
            '("hospital capacity" OR "bed occupancy" OR "emergency department overcrowding") AND (forecasting OR prediction OR model) AND (2020:2026[dp])',
            '("overcrowding" OR "boarding") AND ("emergency department") AND (prediction OR machine learning) AND (2022:2026[dp])',
            '("hospital capacity") AND ("patient discharge" OR "admission forecasting") AND (emergency OR EMS)',
        ],
        "nl_queries": [
            "prévision capacité hospitalière encombrement urgences lits",
            "hospital capacity forecasting emergency department overcrowding",
            "modélisation saturation urgences lits aval HUG Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la prévision de la capacité hospitalière et de l'encombrement des urgences. "
            "Extrayez : (1) les indicateurs validés de tension hospitalière (NEDOCS, score de saturation), "
            "(2) les modèles prédictifs d'afflux de patients aux urgences à 24h et 48h, "
            "(3) l'impact de l'engorgement des urgences sur les délais de transfert des ambulances (ambulance diversion). "
            "Contexte : Grand Genève, gestion des lits d'aval aux HUG et au CHUV."
        ),
        "model_info": {
            "algorithm": "Régression Ridge + Prophet (Série Temporelle)",
            "variables": ["patients_urgences_actuels", "temps_attente_moyen", "prevision_entrees_24h", "taux_occupation_lits_aval", "sorties_prevues_24h"],
            "output": "Indice de saturation des urgences (score NEDOCS) prévu pour les prochaines 12 heures",
            "update_frequency": "Toutes les heures",
        },
        "alert_thresholds": {
            "green": {"label": "Urgences fluides", "condition": "nedocs < 100"},
            "orange": {"label": "Urgences surchargées — activer mesures de fluidification", "condition": "100 ≤ nedocs < 140"},
            "red": {"label": "Urgences saturées — déclenchement Plan Blanc interne / déviation ambulances", "condition": "nedocs ≥ 140"},
        },
        "databases": [
            "Dossier Patient Informatisé des Urgences (HUG/CHUV)",
            "Système de gestion des lits hospitaliers en temps réel",
            "Données de régulation d'arrivée des ambulances (SAGA)"
        ],
        "outcome_definition": "Score NEDOCS (National Emergency Department Overcrowding Score) horaire, mesurant le niveau de tension des urgences.",
        "variables_detail": {
            "patients_urgences_actuels": {
                "definition": "Nombre de patients actuellement présents physiquement dans le service des urgences.",
                "plugged": False,
                "source": "DPI Urgences (Manquant - Dataset requis)"
            },
            "temps_attente_moyen": {
                "definition": "Temps d'attente moyen en minutes avant le premier examen médical aux urgences.",
                "plugged": False,
                "source": "DPI Urgences (Manquant - Dataset requis)"
            },
            "prevision_entrees_24h": {
                "definition": "Nombre d'admissions prévues aux urgences au cours des prochaines 24 heures (généré par le modèle).",
                "plugged": False,
                "source": "Modèle d'afflux (Manquant - Dataset requis)"
            },
            "taux_occupation_lits_aval": {
                "definition": "Pourcentage d'occupation des lits de médecine et chirurgie générale d'aval.",
                "plugged": False,
                "source": "Gestionnaire de lits (Manquant - Dataset requis)"
            },
            "sorties_prevues_24h": {
                "definition": "Nombre de sorties de patients ou de transferts planifiés pour libérer des lits.",
                "plugged": False,
                "source": "Gestionnaire de lits (Manquant - Dataset requis)"
            }
        },
        "keywords": ["Capacité hospitalière", "Gestion des lits", "Saturation urgences", "Score NEDOCS", "Lits d'aval", "Ambulance diversion"],
        "clinical_rationale": (
            "L'engorgement des urgences nuit gravement à la sécurité des patients. Prédire la saturation hospitalière à 24-48h permet d'anticiper la libération de l'aval, d'organiser des transferts fluides et d'éviter les situations de déviation des ambulances (ambulance diversion) vers d'autres établissements."
        ),
    },

    "demand-forecasting": {
        "boolean_queries": [
            '("emergency medical services" OR EMS OR ambulance) AND (demand OR call OR volume) AND (forecasting OR prediction OR "time series") AND (2020:2026[dp])',
            '("ambulance demand") AND ("spatial-temporal" OR "machine learning") AND (prediction OR forecasting)',
            '("emergency call volume") AND (weather OR calendar OR holiday) AND (forecasting OR prediction)',
        ],
        "nl_queries": [
            "prévision demande EMS ambulance volume appels séries temporelles météo",
            "prehospital EMS demand forecasting time series machine learning",
            "modélisation volume appels 144 Genève météo vacances grippe",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la prévision de la demande EMS et du volume d'appels d'urgence. "
            "Extrayez : (1) les modèles mathématiques de séries temporelles (Prophet, ARIMA, LSTM, XGBoost) et leur précision (MAPE), "
            "(2) les variables explicatives les plus significatives (météo, épidémies, calendrier, événements sportifs), "
            "(3) la résolution spatio-temporelle optimale pour les prévisions opérationnelles (ex: maille de 1km², créneau de 4h). "
            "Contexte : Grand Genève, prévision de la demande pour la régulation 144 et les ambulances."
        ),
        "model_info": {
            "algorithm": "Prophet + LightGBM + données météo Open-Meteo",
            "variables": ["historique_appels_7j", "temperature", "precipitations", "jour_semaine", "feries", "incidence_grippale"],
            "output": "Prévision demande EMS J+1 à J+7 par heure et par zone",
            "update_frequency": "Quotidienne (6h UTC)",
        },
        "alert_thresholds": {
            "green": {"label": "Demande normale attendue", "condition": "erreur_prevision < 10%"},
            "orange": {"label": "Hausse significative de la demande (+15%) — adapter effectifs", "condition": "10% ≤ erreur_prevision < 20%"},
            "red": {"label": "Surcharge critique attendue (+30% appels) — plan de crise", "condition": "erreur_prevision ≥ 20%"},
        },
        "databases": [
            "Historique de régulation SAGA (144/15)",
            "API météorologique Open-Meteo",
            "Données de surveillance épidémiologique (Sentinelles Suisse/France)"
        ],
        "outcome_definition": "Nombre d'appels d'urgence qualifiés reçus par la centrale 144 par heure.",
        "variables_detail": {
            "historique_appels_7j": {
                "definition": "Nombre d'appels reçus au cours des 7 derniers jours à la même heure.",
                "plugged": False,
                "source": "Base SAGA (Manquant - Dataset requis)"
            },
            "temperature": {
                "definition": "Température de l'air à 2 mètres.",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            },
            "precipitations": {
                "definition": "Hauteur des précipitations en mm au cours de l'heure.",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            },
            "jour_semaine": {
                "definition": "Jour de la semaine (0=Lundi, 6=Dimanche).",
                "plugged": True,
                "source": "Horloge système"
            },
            "feries": {
                "definition": "Variable binaire indiquant si le jour est férié en Suisse ou en France voisine.",
                "plugged": True,
                "source": "Calendrier interne"
            },
            "incidence_grippale": {
                "definition": "Nombre de cas de syndrome grippal pour 100 000 habitants déclarés par le réseau Sentinelles.",
                "plugged": False,
                "source": "Réseau Sentinelles (Manquant - Dataset requis)"
            }
        },
        "keywords": ["Prévision de la demande", "Séries temporelles", "Météo-sensibilité", "Surveillance épidémique", "Planification des effectifs"],
        "clinical_rationale": (
            "La demande en secours préhospitaliers fluctue selon la météo, le calendrier et les épidémies. Prédire précisément ces variations permet de dimensionner adéquatement les équipes de régulation et les ambulances de garde, évitant la surcharge du système ou le gaspillage de ressources."
        ),
    },

    "resource-allocation": {
        "boolean_queries": [
            '("resource allocation" OR "resource management") AND (EMS OR emergency OR ambulance) AND (optimization OR algorithm) AND (2020:2026[dp])',
            '("ambulance allocation") AND ("location-allocation model" OR heuristic) AND (EMS OR prehospital)',
            '("emergency resource") AND ("supply-demand mismatch") AND (optimization OR simulation) AND (EMS)',
        ],
        "nl_queries": [
            "allocation ressources EMS ambulances optimisation couverture",
            "emergency resource allocation location-allocation model optimization",
            "optimisation de l'emplacement et du nombre de véhicules de secours Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur l'allocation optimale des ressources EMS (ambulances, SMUR, hélicoptères). "
            "Extrayez : (1) les modèles de localisation-allocation (ex: p-médian, p-centre), "
            "(2) les méthodes d'ajustement de l'offre de soins préhospitaliers en fonction de la demande fluctuante, "
            "(3) la réduction des coûts opérationnels sans perte de qualité de service (temps de réponse). "
            "Contexte : Grand Genève, gestion de flottes d'ambulances réparties sur plusieurs compagnies."
        ),
        "model_info": {
            "algorithm": "Modèle de Programmation Linéaire Mixte (MIP) + Heuristique gloutonne",
            "variables": ["demande_prevue_secteur", "localisation_casernes", "cout_operationnel_vehicule", "temps_reponse_cible", "dispo_budgetaire"],
            "output": "Nombre optimal d'ambulances à affecter à chaque caserne/base d'attente pour la saison",
            "update_frequency": "Trimestrielle (planification stratégique)",
        },
        "alert_thresholds": {
            "green": {"label": "Ressources allouées optimales", "condition": "adequation_ressources_demande ≥ 0.95"},
            "orange": {"label": "Inadéquation offre-demande locale — ajustement mineur requis", "condition": "0.80 ≤ adequation_ressources_demande < 0.95"},
            "red": {"label": "Inadéquation majeure — rupture de couverture prévisible", "condition": "adequation_ressources_demande < 0.80"},
        },
        "databases": [
            "Base de données d'activité TechCAD",
            "Données de coûts d'exploitation des compagnies d'ambulances",
            "Cahier des charges et budgets cantonaux"
        ],
        "outcome_definition": "Taux d'utilisation des ambulances (proportion du temps de garde où une ambulance est engagée sur une intervention d'urgence).",
        "variables_detail": {
            "demande_prevue_secteur": {
                "definition": "Volume de demande d'interventions annuel prévu par secteur géographique.",
                "plugged": False,
                "source": "Modèle prédictif long terme (Manquant - Dataset requis)"
            },
            "localisation_casernes": {
                "definition": "Coordonnées des points de départ possibles des ambulances.",
                "plugged": True,
                "source": "SIG SITG"
            },
            "cout_operationnel_vehicule": {
                "definition": "Coût horaire d'exploitation d'une ambulance en garde active (personnel + véhicule).",
                "plugged": False,
                "source": "Données financières internes (Manquant - Dataset requis)"
            },
            "temps_reponse_cible": {
                "definition": "Délai maximal toléré pour atteindre 90% de la population du secteur.",
                "plugged": True,
                "source": "Cahier des charges cantonal (Fixe: 15 min)"
            },
            "dispo_budgetaire": {
                "definition": "Budget annuel maximal alloué par le canton pour le dispositif de secours préhospitalier.",
                "plugged": False,
                "source": "Direction de la santé publique (Manquant - Dataset requis)"
            }
        }
    },

    "epidemic-early-warning": {
        "boolean_queries": [
            '("early warning" OR syndromic OR surveillance) AND (epidemic OR influenza OR COVID-19 OR gastroenteritis) AND (EMS OR "emergency calls" OR "ambulance") AND (2020:2026[dp])',
            '("syndromic surveillance") AND (EMS OR prehospital OR emergency) AND (forecasting OR detection OR algorithm)',
            '("epidemic early warning") AND ("influenza-like illness" OR ILI) AND (EMS OR emergency)',
        ],
        "nl_queries": [
            "détection précoce épidémie grippe gastro appels 144 surveillance syndromique",
            "epidemic early warning syndromic surveillance EMS emergency calls",
            "détection vagues de grippe et gastro à Genève via les appels d'urgence",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la détection précoce des épidémies (grippe, COVID-19, gastro-entérite) par surveillance syndromique des appels EMS. "
            "Extrayez : (1) l'avance temporelle (lead time) des appels EMS par rapport aux données de médecine de ville ou d'hospitalisation (généralement 3 à 7 jours), "
            "(2) les algorithmes de détection d'anomalies (Farrington, cumulative sum - CUSUM) appliqués aux séries temporelles d'appels, "
            "(3) les mots-clés ou codes de régulation les plus sensibles pour chaque syndrome épidémique. "
            "Contexte : Grand Genève, surveillance syndromique transfrontalière."
        ),
        "model_info": {
            "algorithm": "Algorithme de Farrington modifié + CUSUM",
            "variables": ["appels_fievre_24h", "appels_dyspnee_24h", "appels_diarrhee_24h", "historique_appels_baseline", "alertes_sentinelles"],
            "output": "Probabilité de dépassement de seuil épidémique (Grippe, Gastro, COVID) à J+3",
            "update_frequency": "Quotidienne (6h UTC)",
        },
        "alert_thresholds": {
            "green": {"label": "Activité épidémique normale (Bruit de fond)", "condition": "cusum_score < 2.0"},
            "orange": {"label": "Alerte épidémique modérée — hausse anormale des appels", "condition": "2.0 ≤ cusum_score < 4.0"},
            "red": {"label": "Seuil épidémique franchi — début de vague épidémique", "condition": "cusum_score ≥ 4.0"},
        },
        "databases": [
            "Base de données de régulation SAGA (motifs cliniques d'appels)",
            "Réseau Sentinelles Suisse (Meteo-Sentinelles) / France",
            "Données de biologie médicale des laboratoires hospitaliers (HUG)"
        ],
        "outcome_definition": "Dépassement du seuil épidémique national de cas cliniques déclarés en médecine de premier recours.",
        "variables_detail": {
            "appels_fievre_24h": {
                "definition": "Nombre d'appels régulés au cours des dernières 24 heures pour motif de fièvre isolée ou syndrome grippal.",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "appels_dyspnee_24h": {
                "definition": "Nombre d'appels pour détresse respiratoire aiguë chez l'adulte ou l'enfant.",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "appels_diarrhee_24h": {
                "definition": "Nombre d'appels pour vomissements/diarrhées évoquant une gastro-entérite aiguë.",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "historique_appels_baseline": {
                "definition": "Moyenne historique attendue pour ces motifs sur les 5 dernières années à la même période (hors épidémie).",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "alertes_sentinelles": {
                "definition": "Statut d'alerte officiel publié par les réseaux de surveillance nationaux (0=Pas d'alerte, 1=Alerte).",
                "plugged": False,
                "source": "Réseau Sentinelles API (Manquant - Dataset requis)"
            }
        }
    },

    "surveillance": {
        "boolean_queries": [
            '("public health surveillance" OR "syndromic surveillance") AND (EMS OR emergency OR ambulance) AND (system OR implementation) AND (2020:2026[dp])',
            '("real-time surveillance") AND (EMS OR emergency) AND (outbreak OR "anomaly detection") AND (2022:2026[dp])',
            '("data integration") AND ("syndromic surveillance") AND (prehospital OR EMS OR emergency)',
        ],
        "nl_queries": [
            "surveillance syndromique santé publique appels urgences temps réel",
            "syndromic surveillance public health emergency department EMS",
            "intégration des données 144 pour la veille sanitaire Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la mise en place de systèmes de surveillance de santé publique basés sur les données EMS. "
            "Extrayez : (1) les architectures logicielles d'intégration en temps réel des données de régulation, "
            "(2) l'apport de ces données pour la détection d'événements inhabituels (bioterrorisme, vagues de chaleur, monoxyde de carbone), "
            "(3) les cadres juridiques et de protection des données (RGPD/LPD) pour la réutilisation de ces données. "
            "Contexte : Grand Genève, veille sanitaire transfrontalière CH/FR."
        ),
        "model_info": {
            "algorithm": "Détection d'anomalies de séries temporelles (Isolation Forest + CUSUM)",
            "variables": ["appels_totaux_24h", "distribution_motifs_appels", "clusters_spatiaux_appels", "alertes_meteo_live", "alertes_qualite_air"],
            "output": "Indice d'anomalie sanitaire global (0 à 1) + Cartographie des micro-clusters d'appels",
            "update_frequency": "Toutes les heures",
        },
        "alert_thresholds": {
            "green": {"label": "Situation sanitaire stable", "condition": "score_anomalie < 0.50"},
            "orange": {"label": "Hausse inhabituelle d'activité — investigation requise", "condition": "0.50 ≤ score_anomalie < 0.80"},
            "red": {"label": "Anomalie sanitaire majeure détectée — alerte santé publique", "condition": "score_anomalie ≥ 0.80"},
        },
        "databases": [
            "Base de données de régulation SAGA (144/15)",
            "Flux de données météo et pollution (Open-Meteo / Copernicus)",
            "Registre cantonal des maladies déclarables (Médecin cantonal Genève)"
        ],
        "outcome_definition": "Survenue d'un événement sanitaire d'importance internationale (USPII) ou d'un cluster d'infections localisé.",
        "variables_detail": {
            "appels_totaux_24h": {
                "definition": "Nombre total d'appels d'urgence reçus au cours des dernières 24 heures.",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "distribution_motifs_appels": {
                "definition": "Répartition en pourcentage des différents motifs d'appels (cardiaque, traumatisme, psychiatrie, etc.).",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            },
            "clusters_spatiaux_appels": {
                "definition": "Présence d'un regroupement spatial anormal d'appels similaires (calculé par l'indice de Moran).",
                "plugged": False,
                "source": "Calculé par SIG (Manquant - Dataset requis)"
            },
            "alertes_meteo_live": {
                "definition": "Alertes de vigilance météorologique actives (canicule, grand froid, orages).",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            },
            "alertes_qualite_air": {
                "definition": "Pics de pollution de l'air déclarés par les stations de mesure locales.",
                "plugged": True,
                "source": "Copernicus Air Quality API (Live)"
            }
        }
    },

    "surge-management": {
        "boolean_queries": [
            '("surge capacity" OR "surge management" OR "disaster capacity") AND (EMS OR emergency OR hospital) AND (2020:2026[dp])',
            '("mass influx" OR "casualty surge") AND (emergency OR hospital OR EMS) AND (coordination OR planning)',
            '("resource surge") AND ("emergency department" OR ambulance) AND (optimization OR simulation)',
        ],
        "nl_queries": [
            "gestion afflux massif victimes surge capacity urgences hôpital plan blanc",
            "surge capacity management EMS emergency hospital influx",
            "plan de gestion de l'afflux massif de blessés urgences HUG CHUV",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la gestion de l'afflux massif de patients (surge management) dans les EMS et les hôpitaux. "
            "Extrayez : (1) les stratégies de montée en charge (surge capacity) en lits, personnel et matériel, "
            "(2) l'utilisation d'hôpitaux de campagne ou de structures mobiles de secours (PMA), "
            "(3) les critères de décharge rapide de patients stables pour libérer des lits d'urgence. "
            "Contexte : Grand Genève, plans de crise transfrontaliers, coordination HUG / CH Métropole Savoie."
        ),
        "model_info": {
            "algorithm": "Modèle de file d'attente dynamique à compartiments (SEIR opérationnel)",
            "variables": ["flux_entree_patients", "temps_traitement_moyen", "lits_urgences_dispo", "personnel_reserve_dispo", "taux_transfert_interhopitaux"],
            "output": "Délai estimé avant saturation complète des urgences (en heures)",
            "update_frequency": "Toutes les 30 minutes en cas d'afflux massif",
        },
        "alert_thresholds": {
            "green": {"label": "Capacité d'afflux normale", "condition": "temps_avant_saturation > 12h"},
            "orange": {"label": "Saturation à court terme (<6h) — pré-alerte plan blanc", "condition": "2h ≤ temps_avant_saturation < 6h"},
            "red": {"label": "Saturation imminente (<2h) — déclenchement plan blanc immédiat", "condition": "temps_avant_saturation < 2h"},
        },
        "databases": [
            "Système d'information hospitalier HUG (flux de patients)",
            "Base de données de régulation TechCAD (ambulances engagées)",
            "Registre du personnel médical d'astreinte et de réserve"
        ],
        "outcome_definition": "Délai réel avant lequel le service des urgences est contraint de refuser de nouvelles admissions d'ambulances.",
        "variables_detail": {
            "flux_entree_patients": {
                "definition": "Nombre de patients admis par heure aux urgences (ambulances + patients couchés).",
                "plugged": False,
                "source": "DPI Urgences (Manquant - Dataset requis)"
            },
            "temps_traitement_moyen": {
                "definition": "Durée moyenne de prise en charge d'un patient aux urgences avant décision d'hospitalisation ou de sortie.",
                "plugged": False,
                "source": "DPI Urgences (Manquant - Dataset requis)"
            },
            "lits_urgences_dispo": {
                "definition": "Nombre de lits d'examen actuellement libres dans le service des urgences.",
                "plugged": False,
                "source": "DPI Urgences (Manquant - Dataset requis)"
            },
            "personnel_reserve_dispo": {
                "definition": "Nombre de médecins et d'infirmiers mobilisables en moins d'une heure (réserve opérationnelle).",
                "plugged": False,
                "source": "Logiciel RH (Manquant - Dataset requis)"
            },
            "taux_transfert_interhopitaux": {
                "definition": "Nombre de patients transférés par heure vers d'autres établissements du réseau pour soulager le centre principal.",
                "plugged": False,
                "source": "SAGA (Manquant - Dataset requis)"
            }
        }
    },

    "pandemic-preparedness": {
        "boolean_queries": [
            '("pandemic preparedness" OR "pandemic planning") AND (EMS OR emergency OR prehospital) AND (2020:2026[dp])',
            '("pandemic preparedness") AND ("stockpile" OR PPE OR ventilator) AND (EMS OR emergency OR hospital)',
            '("pandemic preparedness") AND (surveillance OR modeling) AND (EMS OR prehospital OR emergency)',
        ],
        "nl_queries": [
            "préparation pandémie EMS ambulances planification stocks EPI",
            "pandemic preparedness emergency medical services planning PPE",
            "plan de préparation pandémique régulation 144 Genève",
        ],
        "evidence_extraction_prompt": (
            "Vous analysez des articles sur la préparation des EMS aux pandémies (type COVID-19, grippe aviaire). "
            "Extrayez : (1) les modèles de prévision de la consommation d'équipements de protection individuelle (EPI) et de matériel critique, "
            "(2) les protocoles de régulation médicale spécifiques aux pandémies (triage téléphonique strict pour éviter l'engorgement), "
            "(3) les stratégies de vaccination et de protection du personnel de première ligne. "
            "Contexte : Grand Genève, coordination fédérale (OFSP) et transfrontalière."
        ),
        "model_info": {
            "algorithm": "Modèle épidémiologique compartmentalisé (SEIR) couplé à un modèle de chaîne logistique",
            "variables": ["taux_reproduction_R0", "consommation_masques_jour", "stock_EPI_disponible", "taux_infection_personnel", "prevision_hospitalisations_7j"],
            "output": "Autonomie estimée du stock d'EPI en jours",
            "update_frequency": "Hebdomadaire en période de crise",
        },
        "alert_thresholds": {
            "green": {"label": "Stocks et effectifs sécurisés", "condition": "autonomie_stocks_jours ≥ 30"},
            "orange": {"label": "Tension sur les stocks — réapprovisionnement d'urgence", "condition": "10 ≤ autonomie_stocks_jours < 30"},
            "red": {"label": "Rupture de stock imminente — rationnement et réutilisation", "condition": "autonomie_stocks_jours < 10"},
        },
        "databases": [
            "Inventaire logistique centralisé de la pharmacie hospitalière (HUG)",
            "Système de suivi épidémiologique cantonal",
            "Registre des absences du personnel de secours"
        ],
        "outcome_definition": "Nombre de jours restants avant épuisement complet des stocks d'équipements de protection individuelle (masques FFP2, surblouses).",
        "variables_detail": {
            "taux_reproduction_R0": {
                "definition": "Nombre moyen de cas secondaires infectés par un seul cas (indice de transmissibilité de l'épidémie).",
                "plugged": False,
                "source": "OFSP / Office de santé publique (Manquant - Dataset requis)"
            },
            "consommation_masques_jour": {
                "definition": "Nombre moyen de masques FFP2 consommés par jour par les équipages d'ambulances.",
                "plugged": False,
                "source": "Gestion des stocks interne (Manquant - Dataset requis)"
            },
            "stock_EPI_disponible": {
                "definition": "Quantité physique de masques, gants et blouses en stock à la pharmacie centrale.",
                "plugged": False,
                "source": "Gestion des stocks interne (Manquant - Dataset requis)"
            },
            "taux_infection_personnel": {
                "definition": "Pourcentage d'ambulanciers et de régulateurs actuellement en arrêt maladie pour cause d'infection.",
                "plugged": False,
                "source": "Logiciel RH (Manquant - Dataset requis)"
            },
            "prevision_hospitalisations_7j": {
                "definition": "Nombre de nouvelles admissions hospitalières prévues pour la semaine à venir.",
                "plugged": False,
                "source": "Modèle prédictif cantonal (Manquant - Dataset requis)"
            }
        }
    },

    "cross-border-coordination": {
        "boolean_queries": [
            '("cross-border" OR transboundary OR international) AND (coordination OR cooperation) AND (EMS OR emergency OR ambulance) AND (2020:2026[dp])',
            '("cross-border EMS") AND (agreements OR protocols OR legal) AND (France OR Switzerland OR Europe)',
            '("cross-border coordination") AND (interoperability OR communication) AND (emergency OR EMS)',
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
        "databases": [
            "Système de répartition TechCAD transfrontalier",
            "Flux de données douanières de trafic aux frontières",
            "Registre des accords de coopération sanitaire signés"
        ],
        "outcome_definition": "Délai moyen de déclenchement et d'arrivée d'un moyen de secours du pays voisin sur le lieu de l'intervention transfrontalière.",
        "variables_detail": {
            "position_unites_CH": {
                "definition": "Position géographique en temps réel des ambulances suisses (Genève).",
                "plugged": False,
                "source": "TechCAD 144 (Manquant - Dataset requis)"
            },
            "position_unites_FR": {
                "definition": "Position géographique en temps réel des ambulances françaises (SAMU 74).",
                "plugged": False,
                "source": "TechCAD SAMU (Manquant - Dataset requis)"
            },
            "protocoles_actifs": {
                "definition": "Variable binaire indiquant si les accords d'assistance mutuelle transfrontalière sont actuellement actifs.",
                "plugged": True,
                "source": "Registre juridique (Fixe: True)"
            },
            "temps_passage_frontiere": {
                "definition": "Temps de ralentissement estimé en minutes aux différents postes de douane (congestion ou contrôles).",
                "plugged": True,
                "source": "OSRM / Google Maps API (Live)"
            },
            "disponibilite_ressources": {
                "definition": "Nombre de véhicules d'urgence libres de part et d'autre de la frontière.",
                "plugged": False,
                "source": "TechCAD (Manquant - Dataset requis)"
            }
        }
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
        "databases": [
            "Système d'Information Géographique SITG (Genève)",
            "Flux de données d'activité de régulation TechCAD",
            "Flux d'alertes météo (MétéoSuisse) et qualité de l'air (Copernicus)"
        ],
        "outcome_definition": "Indice de conscience situationnelle opérationnelle (mesurant le taux d'information correcte et disponible pour le commandement lors d'un événement majeur).",
        "variables_detail": {
            "position_unites_GPS": {
                "definition": "Flux de coordonnées GPS de l'ensemble de la flotte de véhicules de secours.",
                "plugged": False,
                "source": "GPS live (Manquant - Dataset requis)"
            },
            "statut_disponibilite": {
                "definition": "Statut opérationnel de chaque ambulance (libre, en route, sur place, à l'hôpital, indisponible).",
                "plugged": False,
                "source": "TechCAD (Manquant - Dataset requis)"
            },
            "alertes_meteo_live": {
                "definition": "Flux d'alertes de vigilance météo de niveau orange ou rouge sur le secteur.",
                "plugged": True,
                "source": "Open-Meteo API (Live)"
            },
            "alertes_epidemiques": {
                "definition": "Signalement d'un dépassement de seuil d'alerte épidémique par le médecin cantonal.",
                "plugged": False,
                "source": "Service du médecin cantonal (Manquant - Dataset requis)"
            },
            "trafic_temps_reel": {
                "definition": "Flux d'événements routiers majeurs (accidents, routes coupées, bouchons de plus de 15 minutes).",
                "plugged": False,
                "source": "Waze API (Manquant - Dataset requis)"
            }
        }
    },
}
