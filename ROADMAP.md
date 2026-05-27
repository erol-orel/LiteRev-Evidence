# Feuille de Route Stratégique et Technique : LiteRev-Evidence
**Auteur :** Manus AI  
**Date :** 27 mai 2026  
**Version :** 3.0.0 — Intégration complète des sources de données validées (GESICA + GeoAI4EI)

---

## 1. Positionnement Stratégique et Vision Unifiée

LiteRev-Evidence est conçu comme le **moteur d'intelligence par les preuves** commun à trois initiatives complémentaires, selon l'**Option 3 (Evidence-to-Scenario)** retenue [^1] :

| Initiative | Programme | Rôle de LiteRev | Partenaires Clés |
| :--- | :--- | :--- | :--- |
| **GESICA** | Interreg France-Suisse | Extraction de preuves EMS, aide à la décision de crise, Evidence Panels liés aux scénarios SAGA | HUG, CHUV, HEIG-VD, TECHWAN, UNIGE [^5] [^6] |
| **GeoAI4EI** | Horizon HLTH-2025-01 | Extraction structurée pour l'intelligence épidémique, graphe de connaissances, QA conversationnel pour HERA/ATHINA/MOOD | UNIGE, partenaires Horizon [^1] |
| **EVA LiteRev** | Financement innovation | Maturité UX, workflows PRISMA, screening actif, documentation et formation | UNIGE DSMC [^2] [^3] |

La vision est celle d'un **système d'aide à la décision basé sur les preuves** qui couple, en temps réel, les sorties des modèles opérationnels (prévisions de charge EMS, cartes de risque épidémique) avec des panneaux de preuves scientifiques contextualisés. Cette approche différencie GESICA de ses concurrents directs (Hexagon, Swisscom Avanti) par une couche d'intelligence scientifique unique [^5] [^6].

---

## 2. Besoins Opérationnels GESICA : Traduction depuis la Littérature

L'analyse de la revue systématique sur l'IA dans les SMU en situation de crise [^4] permet d'identifier cinq axes thématiques prioritaires. LiteRev-Evidence doit extraire, classifier et synthétiser les preuves pour chacun :

| Axe | Cas d'Usage Opérationnel GESICA | Variables à Extraire | Signaux GESICA |
| :--- | :--- | :--- | :--- |
| **Axe 1 — Prévision des Risques** | Anticipation des crises environnementales (vagues de chaleur, inondations, pandémies) affectant la région transfrontalière Genève-Vaud-Neuchâtel-France [^4] [^6] | Types de catastrophes, modèles prédictifs, variables climatiques/épidémiques | `disaster_type`, `risk_level`, `geographic_scope` |
| **Axe 2 — Prévision de la Demande EMS** | Modélisation du volume d'appels (144 Suisse / 15 France) et des taux d'arrivée aux urgences [^4] | Horizons de prévision (h, j), métriques d'erreur (MAE, MAPE, RMSE), algorithmes (XGBoost, LSTM, Prophet) | `forecast_horizon`, `performance_metrics`, `ml_model` |
| **Axe 3 — Optimisation des Ressources** | Dimensionnement des flottes d'ambulances, affectation SMUR/SAMU, gestion des capacités HUG/CHUV [^5] [^6] | Temps de réponse, taux d'occupation, stratégies d'allocation dynamique | `resource_type`, `response_time_target`, `optimization_method` |
| **Axe 4 — Triage et Régulation** | Priorisation des appels critiques (P0/P1 vs P2/P3) et détection précoce de la gravité clinique [^4] | Données audio, notes de régulation, scores de gravité (NEWS, SOFA), précision du triage | `triage_score`, `data_modality`, `clinical_outcome` |
| **Axe 5 — Incertitude et Fiabilité** | Quantification de l'incertitude des prédictions pour sécuriser les décisions des directeurs de crise [^4] | Intervalles de confiance, étalonnage des modèles, méthodes bayésiennes, ensembles | `uncertainty_method`, `calibration_score`, `explainability_level` |

---

## 3. Besoins Opérationnels GeoAI4EI : Traduction depuis les WPs Horizon

La structure des work packages de GeoAI4EI [^1] définit des besoins précis auxquels LiteRev-Evidence doit répondre :

| Work Package | Tâche Horizon | Traduction LiteRev | Livrables Attendus |
| :--- | :--- | :--- | :--- |
| **WP3 — Extraction** | T3.2 : Extraction depuis publications scientifiques et médias | Pipeline d'ingestion multi-source (PubMed, PMC, OpenAlex, grey literature) + extraction structurée | Corpus FAIR versionné, bibliothèque Python d'extraction [^1] |
| **WP3 — Biais & UQ** | T3.3 : Évaluation et mitigation des biais dans l'extraction IA | Module d'évaluation de la qualité des preuves (biais, représentativité, incertitude épistémique) | Rapport de biais par article, score de qualité méthodologique [^1] [^4] |
| **WP3 — Fusion** | T3.4 : Fusion d'événements depuis sources hétérogènes | Déduplication et alignement des entités (pathogènes, interventions, MCMs) entre sources | Graphe de connaissances épidémique unifié [^1] |
| **WP3 — QA** | T3.5 : Assistants conversationnels pour aide à la décision | Interface de chat contextuelle sur le corpus actif, réponses ancrées dans les preuves | Assistant QA intégré aux dashboards HERA/ATHINA/MOOD [^1] |
| **WP4 — Scénarios** | Outils de scénarios pour agences de santé publique | Couplage des prévisions épidémiques (GeoAI4EI) avec les prévisions de charge EMS (GESICA) | Dashboards scénarios avec Evidence Panels [^1] [^5] |

---

## 4. Matrice d'Intégration des Sources de Données

Pour alimenter ce double graphe de connaissances et fournir une aide à la décision robuste, LiteRev-Evidence intègre une matrice de sources de données classées par priorité opérationnelle.

### Priorité P0 — Littérature Scientifique (Moteur de Recherche de Preuves)
*Ces sources constituent le corpus scientifique. La sélection se fait sur titre, résumé et métadonnées (screening). Pour les articles retenus, le texte intégral est extrait pour alimenter l'Evidence Panel et l'assistant QA.*
- **A1 : PubMed** (NCBI API) — Référence biomédicale, EMS, épidémiologie et urgences [^4].
- **A2 : PubMed Central (PMC)** (NCBI API) — Accès au texte intégral en accès libre.
- **A3 : OpenAlex** (REST API) — Métadonnées ouvertes de 250M+ publications mondiales.
- **A4 : CrossRef** (REST API) — Métadonnées DOI et liens d'éditeurs.

### Priorité P1 — Données Opérationnelles GESICA MVP (Temps Réel et Contexte)
*Données requises pour le tableau de bord d'aide à la décision et la prévision de charge immédiate dans la région transfrontalière.*
- **D1 : SAMU / Centre 15 (France)** — Volume d'appels, temps de réponse, nature des interventions [^5] [^6].
- **D2 : 144 Vaud / ORCA (Suisse)** — Données de régulation médicale Suisse romande [^5] [^6].
- **D3 : HUG / CHUV Urgences** — Flux patients, taux d'occupation des lits, temps d'attente [^5] [^6].
- **B1 : Météo-France API** — Alertes vigilance, températures extrêmes, précipitations [^4].
- **B2 : MeteoSwiss API** — Données météo nationales et alertes de vigilance pour la Suisse [^4].
- **G1 : Swisstopo & IGN** — Cartographie transfrontalière, limites cantonales, réseau routier [^6].
- **G3 : OpenStreetMap (OSM)** — Géolocalisation des points d'intérêt (hôpitaux, casernes, héliports).

### Priorité P2 — Données GESICA Enrichies (Normalisation et Modélisation)
*Données utilisées pour affiner les modèles prédictifs et normaliser les taux de demande par bassin de population.*
- **C1 : Réseau Sentinelles (France)** — Taux d'incidence hebdomadaires des maladies infectieuses (grippe, gastro).
- **C5 : Santé publique France (OpenData)** — Passages aux urgences OS-Médecins et hospitalisations régionales.
- **C6 : OFSP / Swissmedic (Suisse)** — Déclarations obligatoires et surveillance épidémique Suisse.
- **E1 : INSEE (France)** — Données démographiques, densité et pyramide des âges par commune.
- **E2 : OFS (Suisse)** — Démographie cantonale et structure de la population transfrontalière.
- **F1 : HERE Traffic API** — Trafic temps réel et congestion routière pour le calcul dynamique des temps de réponse.
- **F2 : OpenStreetMap + OSRM** — Routage d'urgence et calcul d'isochrones ambulance.

### Priorité P3 — Données GeoAI4EI (Intelligence Épidémique Globale)
*Données requises pour la détection précoce des épidémies et l'évaluation des contre-mesures médicales (MCMs).*
- **C2 : ECDC API** — Surveillance épidémique européenne et alertes harmonisées [^1].
- **C3 : WHO Global Health Observatory** — Indicateurs de santé mondiaux et épidémies à déclaration obligatoire.
- **C4 : MOOD / ATHINA / ProMED** — Signaux informels de détection précoce d'épidémies émergentes [^1].
- **B3 : Copernicus Climate Data Store** — Réanalyses climatiques ERA5 et indicateurs de vagues de chaleur [^1].
- **I1 : ProMED / HealthMap** — Veille sanitaire collaborative et alertes épidémiques non officielles.
- **I2 : GDELT Project** — Événements et signaux épidémiques extraits des médias mondiaux.

### Priorité P4 — Sources d'Extension et Logistique
*Données de logistique sanitaire et sources de littérature complémentaires.*
- **A5 : Europe PMC** — Extraction de texte intégral XML structuré de haute qualité.
- **A6 : Embase** — Littérature pharmacologique et d'urgence (accès payant).
- **A7 : medRxiv / bioRxiv** — Signaux très précoces via les prépublications épidémiques.
- **H1 : ANSM (France)** — Suivi des ruptures de stock de médicaments et contre-mesures médicales.
- **H2 : Swissmedic** — Autorisations et alertes de sécurité sur les dispositifs médicaux en Suisse.
- **H4 : Pharmacies de Garde (OSM)** — Géolocalisation des ressources pharmaceutiques de garde.

---

## 5. Architecture Technique Cible

LiteRev-Evidence évolue d'un moteur de recherche documentaire vers un **système d'aide à la décision à double graphe de connaissances**.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          COUCHES UTILISATEURS (UI)                           │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌──────────────┐  │
│  │      Espace GESICA      │  │    Espace GeoAI4EI      │  │  Espace EVA  │  │
│  │  EMS · Crises · SAGA    │  │  Épidémies · HERA/MOOD  │  │   PRISMA     │  │
│  │  Evidence Panels        │  │  Risk Maps · Scenarios  │  │  Screening   │  │
│  └───────────┬─────────────┘  └───────────┬─────────────┘  └──────┬───────┘  │
└──────────────┼────────────────────────────┼─────────────────────┼────────────┘
               │                            │                     │
┌──────────────▼────────────────────────────▼─────────────────────▼────────────┐
│                           MOTEUR LITEREV CORE (API)                          │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │   Recherche Hybride Sémantique + BM25 (BGE-M3 · pgvector)             │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │   Extracteur de Signaux Structurés (GESICA Signals · EI Signals)      │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │   Graphe de Connaissances EMS  ·  Graphe de Connaissances Épidémique  │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │   Assistant Conversationnel Contextuel (QA sur corpus filtré)         │  │
│  ├────────────────────────────────────────────────────────────────────────┤  │
│  │   Pipeline NLP de Normalisation (backfill métadonnées manquantes)     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
               │                            │                     │
┌──────────────▼────────────────────────────▼─────────────────────▼────────────┐
│                         COUCHE DONNÉES (PostgreSQL + pgvector)               │
│   literature_document · literature_chunk · embeddings BGE-M3                 │
│   project_context : gesica | geoai4ei | eva                                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Principes Architecturaux Clés

**Séparation des contextes par `project_context`** : Les documents sont taggués `gesica`, `geoai4ei` ou `eva`. L'interface adapte ses filtres, ses ontologies et ses panneaux d'analyse selon le contexte actif. Un document peut appartenir à plusieurs contextes (ex: un article sur la prévision de demande EMS pendant une épidémie est pertinent pour GESICA et GeoAI4EI simultanément) [^1] [^7].

**Double graphe de connaissances** : Le graphe EMS/crise (GESICA) structure les entités autour des ressources, des modèles de prévision et des protocoles de triage. Le graphe épidémique (GeoAI4EI) structure les pathogènes, vecteurs, MCMs et outcomes. Les deux graphes sont couplés via les scénarios de crise sanitaire transfrontalière [^1] [^4] [^5].

**Couplage scénario** : Les scénarios GeoAI4EI (ex: émergence d'un pathogène respiratoire en hiver) alimentent les priors de charge dans les modèles GESICA ; les prévisions GESICA fournissent en retour des indicateurs de stress du système et de capacité de réponse [^1] [^5] [^6].

---

## 6. Plan de Développement Détaillé

Le développement est structuré en **six phases consécutives**, organisées pour livrer un MVP GESICA opérationnel en priorité, puis enrichir progressivement l'outil vers GeoAI4EI et les fonctionnalités transversales EVA.

### Phase 1 — Consolidation Technique (Terminée ✓)

Cette phase, désormais achevée, a permis d'établir les fondations solides sur lesquelles repose tout le développement futur. La dette technique accumulée lors des premières itérations avec Perplexity a été entièrement résorbée [^7].

Les corrections apportées comprennent la séparation stricte des types TypeScript (UI `camelCase` vs API `snake_case`), l'établissement d'une frontière de conversion hermétique dans `api.ts`, la suppression du double endpoint `/evidence-summary` et du double `import re` dans `main.py`, ainsi que la refonte complète du script `deploy.sh` avec auto-relance post-`git pull` pour éliminer les états de build corrompus [^7].

### Phase 2 — Backend GESICA : Signaux Structurés et Endpoints Dédiés

**Objectif** : Transformer le backend en moteur d'extraction de preuves spécialisé pour les de la littérature d'urgence, et exposer des endpoints permettant à l'interface d'afficher des Evidence Panels riches.

| Endpoint | Description | Données Retournées |
| :--- | :--- | :--- |
| `GET /evidence-summary/{id}` (amélioré) | Extraction avancée des signaux GESICA depuis l'abstract et les chunks d'un document | `ml_models[]`, `forecast_horizons[]`, `performance_metrics{}`, `uncertainty_methods[]`, `geographic_scope`, `scenario_type` |
| `GET /gesica/stats` (nouveau) | Statistiques globales du corpus GESICA | Répartition des modèles IA, métriques moyennes rapportées, couverture géographique, distribution temporelle |
| `GET /gesica/scenarios` (nouveau) | Scénarios de crise prédéfinis avec preuves associées | Liste des scénarios (afflux de victimes, crise épidémique, pénurie de ressources) avec les articles les plus pertinents et les recommandations extraites |
| `GET /geoai4ei/stats` (nouveau) | Statistiques globales du corpus GeoAI4EI | Répartition des pathogènes, vecteurs, MCMs, zones géographiques |
| `GET /corpus/stats` (nouveau) | Vue globale multi-projet du corpus | Totaux par `project_context`, couverture des métadonnées, évolution temporelle du corpus |

### Phase 3 — Interface GESICA : Tableau de Bord d'Aide à la Décision

**Objectif** : Concevoir une interface utilisateur dédiée aux opérateurs EMS et aux directeurs de crise, centrée sur la lisibilité et la rapidité d'accès aux preuves pertinentes.

Les composants à développer sont les suivants. Un **sélecteur de projet** en haut de l'écran (boutons GESICA / GeoAI4EI / EVA) filtre instantanément l'ensemble de l'interface, y compris les filtres disponibles, les ontologies affichées et les colonnes de résultats [^1]. Un **Evidence Panel enrichi** dans le volet latéral affiche, pour chaque document sélectionné, les signaux extraits sous forme de badges colorés par catégorie (Algorithme, Horizon, Métrique, Incertitude, Zone Géo) ainsi qu'un micro-tableau des métriques de performance rapportées [^4] [^6]. Un **indicateur de Force des Preuves** (badge Faible / Modérée / Forte) est calculé dynamiquement selon la présence de métriques validées, d'analyses d'incertitude et de données de validation externe dans l'article [^4]. Un **filtre Année avancé** (curseur de plage 2015–2026) est intégré conformément à la fenêtre temporelle de la revue systématique [^4]. Un **panneau de statistiques du corpus** (vue `/gesica/stats`) offre une vue synthétique de l'état des preuves disponibles sur chaque axe thématique.

### Phase 4 — Pipeline de Normalisation NLP (Stage 2)

**Objectif** : Combler le manque de métadonnées structurées dans la base de données. Actuellement, seulement 51% des documents ont une pathologie définie et 32% une zone géographique renseignée [^7]. Cette lacune rend les filtres peu fiables et dégrade la qualité des Evidence Panels.

Un script de backfill NLP analysera les résumés (abstracts) de tous les articles existants via l'API OpenAI intégrée pour extraire automatiquement les champs manquants. Pour GESICA, les champs ciblés sont `scenario_type` (ex: `ems-demand-forecasting`, `resource-allocation`, `triage`), `ml_model` (ex: `XGBoost`, `LSTM`, `Random Forest`) et `forecast_horizon`. Pour GeoAI4EI, les champs ciblés sont `disease_or_condition` (ex: COVID-19, Influenza, Dengue), `geographic_scope` (ex: France, Suisse, transfrontalier, Europe) et `intervention_type` (ex: vaccination, social distancing, MCM) [^1] [^4] [^7].

### Phase 5 — Assistant Conversationnel et Couplage Scénario

**Objectif** : Implémenter l'assistant d'aide à la décision contextuel (T3.5 de GeoAI4EI) et établir le couplage scénario entre les deux domaines [^1].

L'assistant conversationnel permettra de poser des questions complexes sur le corpus filtré actif, avec des réponses ancrées dans les preuves scientifiques (ex: *"Quels modèles sont les plus performants pour prévoir une hausse de 20% des appels d'urgence à un horizon de 24 heures en contexte épidémique ?"*). Le couplage scénario permettra de lier un scénario GeoAI4EI (émergence d'un pathogène respiratoire) à des prévisions de charge EMS GESICA, avec des recommandations extraites de la littérature sur les stratégies d'allocation de ressources adaptées [^1] [^5] [^6].

### Phase 6 — Fonctionnalités Transversales EVA et Dissémination

**Objectif** : Livrer les fonctionnalités EVA (PRISMA, screening actif, export structuré) et préparer les matériaux de formation et de dissémination [^2] [^3].

Les livrables incluent des workflows PRISMA-conformes pour la gestion des revues systématiques, un module de double-screening avec gestion des conflits entre relecteurs, un export structuré incluant les signaux GESICA extraits, ainsi que des modules pédagogiques pour les écoles d'été et hackathons GeoAI4EI [^1] [^2].

---

## 7. Indicateurs de Succès par Phase

| Phase | Indicateur de Succès |
| :--- | :--- |
| Phase 2 | Les 5 endpoints GESICA retournent des données structurées valides sur le corpus existant (28 documents GESICA) |
| Phase 3 | Un opérateur EMS peut trouver les 3 articles les plus pertinents sur la prévision de demande en moins de 30 secondes |
| Phase 4 | Couverture des métadonnées `scenario_type` et `ml_model` ≥ 90% sur le corpus GESICA |
| Phase 5 | L'assistant répond correctement à 80% des questions de test sur le corpus GESICA |
| Phase 6 | Export PRISMA-conforme fonctionnel sur une revue de test de 50 articles |

---

## 8. Références

[^1]: *Designing a LiteRev-based project that aligns Horizon GeoAI4EI, GESICA, and EVA objectives (1).md* — Analyse d'alignement stratégique, description des WPs GeoAI4EI (T3.2, T3.3, T3.4, T3.5, WP4).
[^2]: *EVA Application (LiteRev-1.pdf)* — Objectifs PRISMA, screening actif, réduction du temps de revue systématique.
[^3]: *LiteRev One-Pager* — Benchmarks et cas pilotes (COVID-19, HIV, chirurgie reconstructrice).
[^4]: *Artificial Intelligence in Emergency Medical Services for Health Emergencies and Disasters: A Systematic Review (Kokou Laris Edjinedja et al., 2026)* — Base scientifique GESICA : 5 axes thématiques, méthodes IA, incertitude.
[^5]: *A.9 Annexe Business-case GESICA* — Objectifs opérationnels, partenaires (HUG, CHUV, HEIG-VD, TECHWAN), indicateurs de performance.
[^6]: *FCS Application Form GESICA (Ref: 20510)* — Cadre Interreg, priorité RSO1.1, indicateurs de livrables, interopérabilité SAGA.
[^7]: *Audit technique LiteRev-Evidence (mai 2026)* — État du code source, couverture des métadonnées DB, corrections appliquées.
