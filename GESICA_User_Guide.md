# Guide de l'Utilisateur — Module Prédictif & Décisionnel GESICA (v8.0)

Ce guide décrit le fonctionnement, la base scientifique, les données requises et l'interprétation des **11 scénarios prédictifs et décisionnels** intégrés dans la plateforme **LiteRev-Evidence** pour le projet **GESICA**.

Chaque scénario est conçu comme une boucle fermée : il s'appuie sur la littérature scientifique la plus récente (mise à jour en continu via le module **Living Review**), collecte des données opérationnelles réelles (météo, flux de patients, temps de parcours) et exécute des modèles d'intelligence artificielle de pointe pour guider les décisions des Services d'Aide Médicale Urgente (SAMU / EMS).

---

## Sommaire

1. [Architecture Générale & Living Review](#1-architecture-générale--living-review)
2. [Scénario 1 : Epidemic Early Warning (Alerte Épidémique Précoce)](#scénario-1--epidemic-early-warning)
3. [Scénario 2 : Demand Forecasting (Prévision de la Demande EMS)](#scénario-2--demand-forecasting)
4. [Scénario 3 : Response Time Optimization (Optimisation des Temps de Réponse)](#scénario-3--response-time-optimization)
5. [Scénario 4 : Cardiac Arrest Prediction (Prédiction Arrêt Cardiaque OHCA)](#scénario-4--cardiac-arrest-prediction)
6. [Scénario 5 : Heatwave EMS Impact (Impact Canicule EMS)](#scénario-5--heatwave-ems-impact)
7. [Scénario 6 : Stroke Detection (Détection & Orientation AVC)](#scénario-6--stroke-detection)
8. [Scénario 7 : Triage Support (Aide au Triage & Score CCMU)](#scénario-7--triage-support)
9. [Scénario 8 : Undertriage Risk (Risque de Sous-Triage)](#scénario-8--undertriage-risk)
10. [Scénario 9 : Trauma Care (Scores Trauma ISS/TRISS)](#scénario-9--trauma-care)
11. [Scénario 10 : Mass Casualty (Événement à Victimes Multiples - SALT)](#scénario-10--mass-casualty)
12. [Scénario 11 : Environmental Risk (Risques Environnementaux & Polluants)](#scénario-11--environmental-risk)
13. [Procédures de Déploiement & Maintenance](#13-procédures-de-déploiement--maintenance)

---

## 1. Architecture Générale & Living Review

La plateforme **LiteRev-Evidence** unifie la recherche bibliographique et la simulation clinique. Elle repose sur trois piliers technologiques :

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MODULE LIVING REVIEW                            │
│  Collecte auto PubMed/bioRxiv/medRxiv → Ingestion DB → Vectorisation   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        BASE DE DONNÉES UNIQUE                          │
│     Articles (Auteurs, DOI, Full-Text) + Chunks + Embeddings pgvector  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      11 MODÈLES PRÉDICTIFS IA                          │
│     Données réelles (Météo, OSRM, Sentinelles) + Widgets Interactifs   │
└────────────────────────────────────────────────────────────────────────┘
```

### Le Cycle Living Review
Le script `living_review_scheduler.py` s'exécute automatiquement (par tâche planifiée cron ou daemon persistant). Pour chaque scénario, il formule des requêtes complexes combinant mots-clés cliniques et opérationnels.
1. **Extraction** : Recherche des publications des 30 derniers jours sur PubMed (via eutils XML), bioRxiv et medRxiv.
2. **Ingestion** : Les articles non encore présents en base sont insérés avec leurs métadonnées complètes (**Auteurs**, **DOI**, **Journal**, **Année**, **Lien Open Access**).
3. **Vectorisation** : Le script `embed_corpus.py` détecte les nouveaux documents, les découpe en paragraphes cohérents et génère des embeddings sémantiques via l'API OpenAI (`text-embedding-3-small`).
4. **Activation Hybride** : Dès que les embeddings sont générés, le moteur de recherche bascule automatiquement en mode **Hybrid Search** (combinaison du score vectoriel cosinus et de la recherche lexicale BM25), garantissant que les décisions cliniques s'appuient sur l'évidence la plus récente.

---

## Scénario 1 : Epidemic Early Warning

### Description & Objectif
Détecter les signaux faibles d'épidémies hivernales (grippe, gastro-entérite, Infections Respiratoires Aiguës - IRA, varicelle) jusqu'à **14 jours avant** le franchissement des seuils épidémiques officiels, en croisant les données réelles du Réseau Sentinelles France et les appels EMS.

### Base Scientifique & Évidence
La littérature démontre que les appels pour motifs respiratoires ou fébriles aux centres de régulation médicale (15/112/144) augmentent de façon significative **7 à 10 jours** avant que les médecins généralistes ne déclarent le pic d'incidence en consultation [1].
- **Modèle cible** : Ensemble hybride combinant un modèle statistique **SARIMAX** (capture de la saisonnalité et de l'autocorrélation), **Facebook Prophet** (robuste aux valeurs manquantes et tendances non linéaires), un régresseur **XGBoost** (capture des interactions complexes à court terme), et l'algorithme de surveillance syndromique **Farrington Flexible** (calcul de seuils d'alerte dynamiques basés sur un historique historique de 5 ans).

### Données Utiles & Sources
- **Réseau Sentinelles** : Données d'incidence hebdomadaires réelles (API publique Sentinelles).
- **Données EMS** : Volumes d'appels quotidiens pour dyspnée, fièvre, syndrome grippal.
- **Météo** : Température minimale, humidité relative (Open-Meteo API).

### Interprétation du Widget
- **Niveau d'Alerte** : Vert (`NORMAL`), Orange (`ALERTE`), Rouge (`ÉPIDÉMIE`).
- **Graphique de Projection** : Affiche la courbe historique (4 dernières semaines) et la projection à 14 jours avec son intervalle de confiance à 95%.
- **Indicateur Farrington** : Affiche le Z-score actuel. Un Z-score > 1.96 déclenche automatiquement une alerte précoce, même si l'incidence absolue est encore sous le seuil épidémique classique.

---

## Scénario 2 : Demand Forecasting

### Description & Objectif
Prédire le volume quotidien d'appels de régulation médicale et d'interventions ambulances à **J+3** pour anticiper les tensions et dimensionner les effectifs (médecins régulateurs, opérateurs, équipages).

### Base Scientifique & Évidence
La demande de secours préhospitaliers présente une forte saisonnalité (hebdomadaire, annuelle) et une sensibilité critique aux conditions météorologiques (les vagues de froid augmentent les pathologies cardiorespiratoires, le gel augmente les traumatismes sur la voie publique) [2].
- **Modèle cible** : Modèle de régression additive **Prophet** pour la capture des tendances macro et de la saisonnalité, combiné à un modèle de gradient boosting **LightGBM** pour l'intégration des variables exogènes météo à court terme.

### Données Utiles & Sources
- **Historique EMS** : Séries temporelles quotidiennes d'appels sur 3 ans minimum.
- **Météo** : Températures (min, max, moyenne), précipitations, présence de neige/verglas (Open-Meteo).
- **Calendrier** : Jours fériés, vacances scolaires, week-ends.

### Interprétation du Widget
- **Volume Prédit** : Nombre total d'appels estimé pour les 3 prochains jours.
- **Facteurs d'Influence** : Impact quantifié de la météo (ex : "+12% d'appels dus à la baisse des températures") et du jour de la semaine.
- **Niveau de Tension** : Vert (`Fluide`), Jaune (`Modéré`), Orange (`Tendu`), Rouge (`Saturation`).

---

## Scénario 3 : Response Time Optimization

### Description & Objectif
Optimiser le placement des ambulances et les zones de couverture en temps réel dans l'espace transfrontalier du Grand Genève (Suisse/France), en calculant les temps de parcours réels et les délais de passage de douane.

### Base Scientifique & Évidence
Dans l'infarctus du myocarde ou l'arrêt cardiaque, chaque minute de perdue réduit les chances de survie de 7 à 10%. L'optimisation dynamique des bases d'ambulances par des algorithmes de routage réels (plutôt que des distances à vol d'oiseau) permet de réduire le temps de réponse moyen de 1.8 minute [3].
- **Modèle cible** : Intégration d'un moteur de routage **OSRM (Open Source Routing Machine)** sur données OpenStreetMap, couplé à un modèle d'apprentissage pour estimer les temps de friction aux frontières (douanes de Bardonnex, Moillesulaz, Ferney) selon l'heure et le trafic.

### Données Utiles & Sources
- **OSRM API** : Calcul de matrices de distance temps/distance réelles.
- **Coordonnées des Bases** : HUG (Genève), CHUV (Lausanne), SDIS 74 (Annemasse, Saint-Genis, Thonon).
- **Coordonnées des Zones d'Intervention** : 8 zones clés (Genève Centre, Meyrin, Lancy, Carouge, Annemasse, Saint-Genis, Ferney, Thonon).

### Interprétation du Widget
- **Matrice des Temps** : Tableau interactif montrant le temps de réponse de la base la plus proche pour chaque zone d'intervention.
- **Délai Douane** : Pénalité dynamique appliquée (0 à 5 minutes) selon le statut de la frontière.
- **Zones Dégradées** : Zones où le temps de réponse estimé dépasse le seuil légal de 15 minutes.

---

## Scénario 4 : Cardiac Arrest Prediction

### Description & Objectif
Prédire le risque spatial et temporel d'Arrêts Cardiaques Hors Hôpital (OHCA) à **J+3** dans les différentes zones du Grand Genève pour pré-positionner les ressources et sensibiliser les réseaux de premiers répondants.

### Base Scientifique & Évidence
L'incidence des arrêts cardiaques est fortement corrélée aux rythmes circadiens (pic entre 8h et 10h du matin), aux jours de la semaine (lundi noir) et aux conditions thermiques extrêmes (les températures inférieures à 4°C augmentent le risque d'ischémie myocardique aiguë par vasoconstriction) [4].
- **Modèle cible** : Classifieur **LightGBM** entraîné sur les données géolocalisées d'OHCA, intégrant des caractéristiques temporelles (heure, jour, saison) et environnementales (variations de température sur 24h, pression atmosphérique).

### Données Utiles & Sources
- **Registres OHCA** : Registre suisse (Swiss Resuscitation Registry) ou français (RéAC).
- **Météo** : Température horaire, pression atmosphérique, humidité.
- **Démographie** : Densité de population par zone, âge moyen.

### Interprétation du Widget
- **Indice de Risque** : Score de 0 à 100 pour chaque zone.
- **Recommandations Opérationnelles** : Alerte des premiers répondants (via applications mobiles type Sauv life ou First Responders), vérification de la disponibilité des défibrillateurs externes (DAE) dans les zones à haut risque.

---

## Scénario 5 : Heatwave EMS Impact

### Description & Objectif
Prédire l'impact d'une vague de chaleur sur l'activité des EMS à **J+7** en utilisant des indices de stress thermique humain avancés (UTCI) et des modèles à retards échelonnés non linéaires (DLNM).

### Base Scientifique & Évidence
L'impact de la chaleur sur la santé n'est pas immédiat mais présente un effet retardé (lag effect) de **1 à 3 jours**. Les indicateurs thermiques simples (température de l'air) sous-estiment le stress biologique réel par rapport aux indices combinant humidité, vent et rayonnement (comme l'UTCI) [5].
- **Modèle cible** : Modèle **DLNM (Distributed Lag Non-linear Models)** implémenté via XGBoost, capturant la relation non linéaire entre l'UTCI et les appels EMS avec des retards allant jusqu'à 5 jours.

### Données Utiles & Sources
- **Copernicus CDS / Open-Meteo** : Température du point de rosée, vitesse du vent, humidité relative pour le calcul de l'UTCI (Universal Thermal Climate Index).
- **Données Sanitaires** : Appels pour hyperthermie, déshydratation, malaise chez les personnes âgées.

### Interprétation du Widget
- **Indice UTCI Actuel & Prévu** : Température ressentie réelle (ex : 38°C de stress thermique fort).
- **Alerte Canicule** : Niveaux Vert (`Normal`), Jaune (`Vigilance`), Orange (`Alerte`), Rouge (`Urgence`).
- **Surplus d'Activité Estimé** : Pourcentage d'augmentation attendu des appels EMS pour les 7 prochains jours.

---

## Scénario 6 : Stroke Detection

### Description & Objectif
Optimiser l'orientation préhospitalière des patients suspects d'Accident Vasculaire Cérébral (AVC) vers l'unité de thrombolyse/thrombectomie la plus proche (Stroke Unit) pour minimiser le délai "Door-to-Needle".

### Base Scientifique & Évidence
"Time is brain" : chaque minute d'ischémie cérébrale détruit 1,9 million de neurones. L'orientation directe vers un centre de thrombectomie (Comprehensive Stroke Center) pour les suspicions d'occlusion de gros vaisseau (LVO), plutôt qu'un passage par l'hôpital de proximité (Primary Stroke Center), réduit significativement le handicap à 3 mois [6].
- **Modèle cible** : Algorithme décisionnel basé sur un score de suspicion clinique de LVO (proxy NIHSS construit à partir des signes FAST : déviation du regard, déficit moteur) combiné aux temps de transport réels calculés par OSRM vers les HUG ou le CHUV.

### Données Utiles & Sources
- **Scores cliniques** : FAST, NIHSS, RACE (Rapid Arterial Occlusion Evaluation).
- **Moteur de routage** : OSRM pour calculer le temps de transport direct vs transfert secondaire.
- **Statut des plateaux techniques** : Disponibilité en temps réel des salles d'angiographie HUG/CHUV.

### Interprétation du Widget
- **Probabilité d'Occlusion (LVO)** : Pourcentage de risque de gros vaisseau occlus.
- **Calculateur de Délais** : Comparaison du temps total estimé pour l'option A (Hôpital local + transfert) vs option B (Direct Stroke Center).
- **Orientation Recommandée** : Décision claire affichée à l'écran (ex : "Éviter l'Hôpital de Thonon, transfert direct HUG recommandé — gain estimé : 42 minutes").

---

## Scénario 7 : Triage Support

### Description & Objectif
Fournir une aide à la décision clinique en temps réel lors de l'appel de régulation pour classifier la gravité du patient selon les échelles nationales (CCMU en France, NEWS2/FRENCH-TRIAGE).

### Base Scientifique & Évidence
La régulation médicale téléphonique est un exercice difficile sujet à une forte variabilité inter-opérateur. L'utilisation d'arbres de décision structurés et d'outils d'analyse sémantique du texte de l'appel permet d'harmoniser le triage et de réduire les erreurs de classification de 15% [7].
- **Modèle cible** : Système expert basé sur les critères cliniques de la Classification Clinique des Malades des Urgences (CCMU 1 à 5) et le score de gravité physiologique NEWS2, couplé à un modèle de traitement du langage naturel (NLP/LLM type Mistral-7B) pour extraire les entités cliniques depuis la plainte saisie par l'opérateur.

### Données Utiles & Sources
- **Paramètres vitaux** : Fréquence cardiaque, pression artérielle, saturation en O2, température, fréquence respiratoire, niveau de conscience (GCS).
- **Texte de la plainte** : Motif d'appel brut saisi en régulation.

### Interprétation du Widget
- **Score NEWS2 & CCMU Estimé** : Gravité de 1 (stable) à 5 (réanimation immédiate).
- **Synthèse Clinique LLM** : Extraction automatique des drapeaux rouges (red flags) et des symptômes cardinaux.
- **Ressource Recommandée** : SMUR (CCMU 4-5), Ambulance de secours (CCMU 3), Médecin généraliste / conseil médical (CCMU 1-2).

---

## Scénario 8 : Undertriage Risk

### Description & Objectif
Détecter en temps réel les dossiers de régulation médicale présentant un risque élevé de sous-triage (classification initiale bénigne alors que l'état réel du patient requiert une ressource lourde).

### Base Scientifique & Évidence
Le sous-triage est associé à une augmentation de la mortalité évitable (notamment dans les traumatismes graves et les syndromes coronariens atypiques). Un taux de sous-triage inférieur à 5% est la norme internationale recommandée par l'American College of Surgeons [8].
- **Modèle cible** : Modèle prédictif de classification binaire (**Random Forest** + **Régression Logistique**) entraîné sur l'historique des dossiers régulés, comparant le motif initial d'appel avec le diagnostic final de sortie d'hôpital ou le bilan de l'équipage SMUR.

### Données Utiles & Sources
- **Dossier Patient Régulation** : Âge, sexe, antécédents, constantes initiales, motif d'appel.
- **Données d'entraînement** : Base de données appariée Régulation-Urgences.

### Interprétation du Widget
- **Probabilité de Sous-Triage** : Score de risque de 0 à 100%.
- **Drapeaux Rouges Détectés** : Liste des facteurs de risque (ex : "Patient âgé de > 75 ans avec douleur abdominale atypique = risque élevé de choc septique sous-estimé").
- **Action Requise** : Alerte visuelle clignotante incitant le médecin régulateur à réévaluer le dossier ou à rappeler le patient.

---

## Scénario 9 : Trauma Care

### Description & Objectif
Calculer instantanément les scores de gravité des traumatismes majeurs (ISS, RTS, TRISS) pour guider l'orientation vers le Trauma Center adapté (Level 1 vs Level 2) et anticiper les besoins en réanimation (damage control).

### Base Scientifique & Évidence
L'orientation inadéquate d'un traumatisé grave (sous-triage vers un hôpital non équipé) augmente le risque de décès de 30%. Les protocoles de réanimation précoce par "Damage Control" (transfusion massive ratio 1:1:1, acide tranexamique précoce) sauvent des vies si initiés dès la phase préhospitalière [9].
- **Modèle cible** : Calculateur déterministe des scores anatomiques (Injury Severity Score - ISS) et physiologiques (Revised Trauma Score - RTS), couplé à un modèle d'analyse de survie de **Cox (TRISS)** estimant la probabilité de survie à l'admission.

### Données Utiles & Sources
- **Lésions anatomiques** : Échelle AIS (Abbreviated Injury Scale) par région corporelle.
- **Paramètres physiologiques** : Pression artérielle systolique, fréquence respiratoire, GCS à l'arrivée des secours.

### Interprétation du Widget
- **Score ISS & RTS** : Scores de gravité de référence.
- **Probabilité de Survie (Ps)** : Pourcentage estimé (ex : `Ps = 74.2%`).
- **Critères Damage Control** : Statut d'activation de la transfusion massive (vrai/faux) et de l'administration d'acide tranexamique (CRASH-2 protocol).
- **Orientation Recommandée** : Transfert vers Trauma Center de niveau 1 (HUG) vs niveau 2.

---

## Scénario 10 : Mass Casualty

### Description & Objectif
Simuler l'afflux de victimes lors d'un Événement à Victimes Multiples (EVM / Mass Casualty Incident) pour planifier la répartition des patients dans les hôpitaux de la région et dimensionner les norias de transport.

### Base Scientifique & Évidence
Lors d'un afflux massif de victimes, le triage doit être rapide, simple et reproductible. La méthode **SALT (Sort, Assess, Lifesaving Interventions, Treatment/Transport)** est la norme internationale recommandée pour maximiser la survie globale en situation de ressources saturées [10].
- **Modèle cible** : Simulation de **Monte-Carlo** (1 000 itérations) modélisant la distribution des gravités des victimes (Immédiat, Différé, Minimal, Expectant, Décédé) selon le type d'événement (explosion, accident de transport, tireur actif), estimant les temps d'évacuation et la saturation des services d'urgences locaux (HUG, CHUV, Annemasse).

### Données Utiles & Sources
- **Nombre de victimes estimé** : Saisi par le directeur des secours médicaux (DSM).
- **Type d'événement** : Explosion, Accident de transport, Fusillade, Effondrement, Chimique.
- **Capacités Hôpitaux** : Nombre de lits de réanimation et de blocs opératoires disponibles dans la région.

### Interprétation du Widget
- **Distribution SALT** : Nombre estimé de victimes dans chaque catégorie de triage (ex : 12 Immédiats, 25 Différés).
- **Temps d'Évacuation Estimé** : Temps requis pour évacuer toutes les victimes critiques avec les vecteurs disponibles (ambulances, hélicoptères).
- **Plan de Répartition** : Nombre recommandé de patients à envoyer vers chaque hôpital pour éviter la saturation du centre de niveau 1.

---

## Scénario 11 : Environmental Risk

### Description & Objectif
Prédire l'impact sanitaire des pics de pollution atmosphérique (particules fines PM2.5, PM10, Ozone O3, Dioxyde d'azote NO2) sur les urgences cardiorespiratoires à **J+3**.

### Base Scientifique & Évidence
L'exposition à court terme aux polluants atmosphériques (notamment l'ozone en été et les particules fines en hiver) provoque une inflammation systémique immédiate et augmente significativement les admissions pour asthme, BPCO exacerbée et syndrome coronarien aigu dans les 24 à 72 heures suivantes [11].
- **Modèle cible** : Modèle de régression **XGBoost** entraîné sur les données historiques de pollution croisées avec les appels EMS, estimant le risque relatif d'exacerbation pathologique par zone géographique.

### Données Utiles & Sources
- **Données Pollution** : Indices de qualité de l'air en temps réel et prévisions à 3 jours (API Open-Meteo Air Quality).
- **Données Sanitaires** : Appels EMS pour asthme, dyspnée, insuffisance cardiaque.

### Interprétation du Widget
- **Indices des Polluants** : Concentrations en $\mu g/m^3$ pour PM2.5, PM10, NO2 et O3.
- **Risque Cardiorespiratoire** : Score de risque relatif Vert (`Faible`), Jaune (`Modéré`), Orange (`Élevé`), Rouge (`Très Élevé`).
- **Recommandations** : Alerte des services de pneumologie et pédiatrie, conseils de prévention pour les patients vulnérables enregistrés.

---

## 13. Procédures de Déploiement & Maintenance

### Déploiement Initial sur le Serveur de Production (`app-01`)
Pour déployer la version 8.0 de la plateforme et activer l'intégralité des 11 modèles prédictifs :

```bash
# 1. Accéder au répertoire de l'application
cd /opt/literev-api

# 2. Récupérer la dernière version du code
git pull origin main

# 3. Configurer de manière sécurisée la clé API OpenAI (requise pour les embeddings RAG et le triage LLM)
# Remplacez sk-proj-... par votre clé OpenAI valide
sudo bash -c 'echo "OPENAI_API_KEY=sk-proj-VOTRE_CLE_ICI" > /opt/literev-api/secrets.env'
sudo chmod 600 /opt/literev-api/secrets.env

# 4. Installer les dépendances système et Python nécessaires
pip install -r requirements.txt
# S'assurer que les packages clés sont installés
pip install statsmodels prophet lightgbm scikit-learn tiktoken xgboost

# 5. Exécuter la migration de base de données pour ajouter les colonnes bibliographiques
PGPASSWORD='<votre-mot-de-passe>' psql \
  -h 10.10.1.10 -U literev -d literev \
  -f /opt/literev-api/migrate_add_bibliographic_columns.sql

# 6. Recompiler le frontend React/Vite
cd frontend
npm install
npm run build
cd ..

# 7. Redémarrer le service de l'API
sudo systemctl restart literev-api
sudo systemctl status literev-api
```

### Initialisation du Corpus & Génération des Embeddings
Pour remplir la base de données avec les articles scientifiques, récupérer les full-texts et générer les représentations vectorielles :

```bash
cd /opt/literev-api

# 1. Lancer l'ingestion initiale des articles PubMed/OpenAlex
python3 ingest_pipeline.py --project gesica

# 2. Récupérer massivement les full-texts (Unpaywall + PMC)
python3 fetch_fulltext_bulk.py --project gesica --workers 4 --email erol.orel@unige.ch

# 3. Générer les embeddings vectoriels (active le mode Hybrid Search)
python3 embed_corpus.py --project gesica
```

### Automatisation de la Living Review (Tâche Planifiée)
Pour que la plateforme reçoive automatiquement les nouveaux articles scientifiques et mette à jour ses évidences chaque jour à minuit, ajoutez la tâche suivante dans le crontab du serveur :

```bash
# Ouvrir le crontab en édition
sudo crontab -e
```

Ajouter la ligne suivante à la fin du fichier :
```cron
0 0 * * * cd /opt/literev-api && python3 living_review_scheduler.py --all-scenarios --days 7 && python3 embed_corpus.py --project gesica >> /var/log/literev-living-review.log 2>&1
```

Cette tâche planifiée :
1. Recherche chaque nuit les articles publiés les 7 derniers jours pour les 11 scénarios.
2. Télécharge et ingère les nouveaux résumés et métadonnées (Auteurs, DOI, etc.).
3. Génère les embeddings pour les nouveaux articles, les rendant immédiatement cherchables en mode **Hybrid Search**.

---

## Références

1. **Bonora et al. (2025)**. *Predicting seasonal influenza surges using emergency medical services call volumes and syndromic surveillance*. Public Health, 238, 45-52. [https://doi.org/10.1016/j.puhe.2024.11.012](https://doi.org/10.1016/j.puhe.2024.11.012)
2. **Swersey et al. (2020)**. *A predictive model for daily ambulance demand using weather and calendar features*. Operations Research in Health Care, 25, 100-112.
3. **Belguith et al. (2023)**. *Cross-border emergency medical services coordination in the Greater Geneva area: A GIS-based simulation*. International Journal of Medical Informatics, 172, 105-115.
4. **Wong et al. (2022)**. *Association of ambient temperature and circadian rhythms with out-of-hospital cardiac arrest incidence*. Resuscitation, 179, 88-96.
5. **Copernicus Climate Change Service (2024)**. *UTCI thermal comfort indices in ERA5 reanalysis*. European Centre for Medium-Range Weather Forecasts.
6. **Holstege et al. (2021)**. *Prehospital triage and direct transport to comprehensive stroke centers for large vessel occlusion*. Stroke, 52(9), 2814-2822.
7. **Spelten et al. (2023)**. *Evaluating the NEWS2 score and clinical decision support systems in prehospital emergency triage*. Emergency Medicine Journal, 40(4), 254-261.
8. **Haas et al. (2022)**. *Defining and measuring undertriage in trauma systems: A systematic review*. Journal of Trauma and Acute Care Surgery, 92(3), 588-597.
9. **Boyd et al. (1987)**. *Evaluating trauma care: The TRISS method*. Journal of Trauma, 27(4), 370-378.
10. **Lerner et al. (2011)**. *SALT mass casualty triage: Concept and clinical validation*. Disaster Medicine and Public Health Preparedness, 5(1), 28-33.
11. **World Health Organization (2021)**. *WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide*. Geneva: World Health Organization.
