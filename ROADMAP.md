# Feuille de Route Stratégique et Technique : LiteRev-Evidence
**Auteur :** Manus AI  
**Date :** 27 mai 2026  
**Version :** 2.0.0 — Révision enrichie GESICA + GeoAI4EI

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

## 4. Architecture Technique Cible

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

## 5. Plan de Développement Détaillé

Le développement est structuré en **six phases consécutives**, organisées pour livrer un MVP GESICA opérationnel en priorité, puis enrichir progressivement l'outil vers GeoAI4EI et les fonctionnalités transversales EVA.

### Phase 1 — Consolidation Technique (Terminée ✓)

Cette phase, désormais achevée, a permis d'établir les fondations solides sur lesquelles repose tout le développement futur. La dette technique accumulée lors des premières itérations avec Perplexity a été entièrement résorbée [^7].

Les corrections apportées comprennent la séparation stricte des types TypeScript (UI `camelCase` vs API `snake_case`), l'établissement d'une frontière de conversion hermétique dans `api.ts`, la suppression du double endpoint `/evidence-summary` et du double `import re` dans `main.py`, ainsi que la refonte complète du script `deploy.sh` avec auto-relance post-`git pull` pour éliminer les états de build corrompus [^7].

### Phase 2 — Backend GESICA : Signaux Structurés et Endpoints Dédiés

**Objectif** : Transformer le backend en moteur d'extraction de preuves spécialisé pour les cinq axes GESICA, et exposer des endpoints permettant à l'interface d'afficher des Evidence Panels riches.

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

## 6. Lacunes Identifiées dans la Version 1.0 de la Feuille de Route

La révision vers la version 2.0 a permis d'identifier et de corriger plusieurs lacunes importantes :

| Lacune v1.0 | Correction v2.0 |
| :--- | :--- |
| GeoAI4EI traité comme une simple extension de Phase 5 | GeoAI4EI intégré dès la Phase 2 avec ses propres endpoints et signaux d'extraction |
| Absence de détail sur les WPs Horizon (T3.2, T3.3, T3.4, T3.5) | Tableau de traduction WP → LiteRev ajouté en Section 3 |
| Couplage scénario GESICA ↔ GeoAI4EI non spécifié | Principe architectural explicité en Section 4 et Phase 5 |
| Pas de mention des partenaires institutionnels (HUG, CHUV, HEIG-VD, TECHWAN) | Intégrés dans le tableau de positionnement stratégique |
| Signaux GESICA listés sans mapping vers les champs DB | Tableau des signaux avec noms de champs DB ajouté en Section 2 |
| Absence de WP-Z (Évaluation, Éthique, Dissémination) | Phase 6 dédiée aux fonctionnalités EVA et dissémination |
| Calendrier en 3 semaines non réaliste pour un outil de production | Calendrier retiré — remplacé par une séquence de phases sans engagement de dates |

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
