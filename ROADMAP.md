# Feuille de Route Stratégique et Technique : LiteRev-Evidence
**Auteur :** Manus AI  
**Date :** 27 mai 2026  
**Version :** 1.0.0  

---

## 1. Introduction et Positionnement Stratégique

L'application **LiteRev-Evidence** est conçue comme le moteur d'extraction de preuves scientifiques et d'intelligence décisionnelle commun à trois initiatives complémentaires [^1] :
1. **GESICA** (Interreg France-Suisse) : Gestion intelligente des crises sanitaires, optimisation des ressources d'urgence préhospitalières (SMU, ambulances) et aide à la décision transfrontalière [^2] [^3].
2. **Horizon GeoAI4EI** (Europe) : Boîte à outils IA pour l'intelligence épidémique, combinant des données multi-sources et l'analyse de publications pour soutenir les politiques de santé publique [^1].
3. **EVA LiteRev** (Financement d'innovation) : Consolidation de la plateforme de revue systématique de littérature assistée par IA (PRISMA, screening actif) [^4] [^5].

Conformément aux orientations stratégiques convenues, le projet adopte l'**Option 3 (Evidence-to-Scenario)** [^1] avec une priorité absolue accordée aux développements liés à **GESICA** [^2] [^3]. Les fonctionnalités développées pour **EVA** serviront de briques transversales de robustesse, tandis que le cadre de **GeoAI4EI** sera déployé dans un second temps pour élargir la portée de l'outil aux épidémies globales [^1].

---

## 2. Traduction des Besoins GESICA depuis la Littérature Scientifique

L'analyse de la revue systématique sur l'IA dans les services médicaux d'urgence (SMU) en situation de crise et de catastrophe [^6] permet d'identifier cinq cas d'usage opérationnels prioritaires pour GESICA. LiteRev-Evidence doit extraire, classifier et synthétiser les preuves scientifiques pour chacun de ces axes :

| Axe Thématique | Cas d'Usage Opérationnel (GESICA) | Variables et Indicateurs à Extraire |
| :--- | :--- | :--- |
| **Axe 1 : Prévision des Risques** | Anticipation des crises environnementales, vagues de chaleur, inondations ou pandémies affectant la région transfrontalière [^6]. | Types de catastrophes, modèles prédictifs, variables climatiques/épidémiques [^6]. |
| **Axe 2 : Prévision de la Demande** | Modélisation du volume d'appels au centre de régulation (144 en Suisse, 15 en France) et des taux d'arrivée des patients [^6]. | Horizons de prévision (heures, jours), métriques d'erreur (MAE, MAPE, RMSE), algorithmes utilisés (XGBoost, LSTM) [^6]. |
| **Axe 3 : Optimisation des Ressources** | Dimensionnement des flottes d'ambulances, affectation des équipages (SMUR/SAMU) et gestion des capacités de lits d'urgence (HUG, CHUV) [^2] [^6]. | Temps de réponse, taux d'occupation des lits, stratégies d'allocation dynamique [^2] [^6]. |
| **Axe 4 : Triage et Régulation** | Aide à la décision pour la priorisation des appels critiques (P0/P1 vs P2/P3) et détection précoce de la gravité clinique [^6]. | Données audio, notes textuelles de régulation, scores de gravité, précision du triage [^6]. |
| **Axe 5 : Incertitude et Fiabilité** | Quantification de l'incertitude des prédictions d'activité pour sécuriser la prise de décision des directeurs de crise [^6]. | Intervalles de confiance, étalonnage des modèles (calibration), méthodes bayésiennes [^6]. |

---

## 3. Architecture Technique Cible (Evidence-to-Scenario)

Pour répondre à ces besoins, LiteRev-Evidence évolue d'un simple moteur de recherche documentaire vers un **système d'aide à la décision basé sur les preuves**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COUCHES UTILISATEURS (UI)                       │
│  ┌───────────────────────┐  ┌───────────────────────┐  ┌────────────┐  │
│  │     Espace GESICA     │  │   Espace GeoAI4EI     │  │ Espace EVA │  │
│  │ (EMS, Crises, SAGA)   │  │ (Épidémies, HERA/MOOD)│  │  (PRISMA)  │  │
│  └───────────┬───────────┘  └───────────┬───────────┘  └──────┬─────┘  │
└──────────────┼──────────────────────────┼─────────────────────┼────────┘
               │                          │                     │
┌──────────────▼──────────────────────────▼─────────────────────▼────────┐
│                        MOTEUR LITEREV CORE (API)                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │               API de Recherche Hybride (Semantic/BM25)           │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │             Générateur d'Evidence Panels (GESICA Signals)        │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │             Assistant Conversationnel (Context-Aware QA)         │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │             Module de Normalisation & Enrichissement NLP         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### Principes de l'Architecture
- **Frontière de Données étanche** : Les documents sont taggués par `project_context` (`gesica`, `geoai4ei`, `eva`) [^1] [^7]. L'interface adapte ses filtres et ses panneaux d'analyse selon le contexte sélectionné [^1].
- **Panneau de Preuves Dynamique (Evidence Panel)** : Lors de la sélection d'un document ou d'un groupe de résultats, l'application extrait automatiquement les signaux GESICA (horizon de prévision, modèles d'apprentissage, métriques de performance, gestion de l'incertitude) [^6].
- **Assistant d'Aide à la Décision** : Un assistant conversationnel permet d'interroger le corpus actif pour répondre à des questions opérationnelles (ex: *"Quels sont les meilleurs modèles pour prévoir une hausse de 20% des appels d'urgence à un horizon de 24 heures ?"*) [^1] [^5].

---

## 4. Plan de Développement Détaillé

Le développement est structuré en **cinq phases consécutives** pour garantir une livraison rapide d'un produit minimum viable (MVP) GESICA, suivi de l'enrichissement GeoAI4EI et des fonctionnalités transversales.

### Phase 1 : Consolidation et Spécification GESICA (En cours)
- **Objectif** : Valider l'audit, définir la structure de données et nettoyer la dette technique du backend [^7].
- **Livrables** : 
  - Nettoyage complet de `main.py` (suppression des routes dupliquées, fusion des extracteurs de signaux) [^7].
  - Rédaction de la présente feuille de route stratégique.
  - Alignement des typages frontend (`App.tsx`) et backend [^7].

### Phase 2 : Enrichissement Backend et Modélisation GESICA
- **Objectif** : Structurer l'extraction des signaux GESICA pour alimenter l'Evidence Panel [^2] [^6].
- **Livrables** :
  - **Amélioration de `/evidence-summary/{id}`** : Extraction avancée et structurée des signaux (modèles d'apprentissage, métriques d'incertitude, horizons temporels, zones transfrontalières concernées) [^6].
  - **Création de `/gesica/stats`** : Endpoint fournissant des statistiques globales sur le corpus GESICA (ex: répartition des modèles d'IA les plus étudiés, précision moyenne rapportée) [^6].
  - **Création de `/gesica/scenarios`** : Endpoint permettant de lier des recommandations scientifiques à des scénarios de crise prédéfinis (ex: afflux de victimes, crise épidémique, pénurie d'ambulances) [^2] [^6].

### Phase 3 : Interface Utilisateur Dédiée GESICA
- **Objectif** : Concevoir un tableau de bord d'aide à la décision visuel et intuitif [^1].
- **Livrables** :
  - **Sélecteur de Projet** : Boutons de filtrage rapide en haut de l'écran pour basculer entre GESICA, GeoAI4EI et EVA [^1] [^7].
  - **Evidence Panel (Volet Latéral)** : Affichage structuré sous forme de badges et de micro-tableaux des signaux extraits (Algorithmes, Horizons, Métriques, Incertitude, Zone Géo) [^6].
  - **Filtre Année Avancé** : Intégration d'un curseur de plage (slider range) pour filtrer les publications récentes (2015-2025) conformément à la revue systématique [^6].
  - **Indicateur de Force des Preuves** : Badge visuel calculé dynamiquement (Faible, Modérée, Forte) selon la présence de métriques validées et d'analyses d'incertitude dans l'article [^6].

### Phase 4 : Pipeline de Normalisation Automatique (Stage 2)
- **Objectif** : Combler le manque de métadonnées dans la base de données (actuellement, seulement 51% des documents ont une pathologie définie et 32% une zone géographique) [^7].
- **Livrables** :
  - **Script de Backfill NLP** : Utilisation d'un modèle de traitement du langage naturel (via l'API OpenAI intégrée) pour analyser les résumés (abstracts) des articles existants et extraire automatiquement :
    - `disease_or_condition` (ex: COVID-19, Influenza, Trauma) [^7].
    - `geographic_scope` (ex: France, Suisse, transfrontalier) [^7].
    - `scenario_type` (ex: ems-demand-forecasting, resource-allocation) [^7].
  - Mise à jour automatique de la base de données PostgreSQL pour garantir des filtres fonctionnels à 100% [^7].

### Phase 5 : Assistant Conversationnel et Fonctionnalités Transversales (EVA & GeoAI4EI)
- **Objectif** : Finaliser l'outil d'aide à la décision et intégrer les besoins de GeoAI4EI [^1].
- **Livrables** :
  - **Assistant de Recherche** : Interface de chat intégrée permettant de poser des questions complexes sur le corpus filtré [^1] [^5].
  - **Pagination & Export** : Amélioration de l'export CSV pour inclure les signaux GESICA extraits [^7].
  - **Espace GeoAI4EI** : Activation des filtres et ontologies spécifiques à l'intelligence épidémique (pathogènes, vecteurs, facteurs environnementaux) [^1].

---

## 5. Calendrier de Réalisation Estimé

```
┌───────────────────────────────────────┬───────────┬───────────┬───────────┐
│ Étape de Développement                │ Semaine 1 │ Semaine 2 │ Semaine 3 │
├───────────────────────────────────────┼───────────┼───────────┼───────────┤
│ Phase 1 : Consolidation (En cours)    │ █████████ │           │           │
│ Phase 2 : Backend & Modèles GESICA    │ ░░███████ │           │           │
│ Phase 3 : UI & Evidence Panel GESICA  │           │ █████████ │           │
│ Phase 4 : Normalisation NLP (Stage 2) │           │ ░░███████ │           │
│ Phase 5 : Assistant & GeoAI4EI        │           │           │ █████████ │
└───────────────────────────────────────┴───────────┴───────────┴───────────┘
```

---

## 6. Références

[^1]: *Designing a LiteRev-based project that aligns Horizon GeoAI4EI, GESICA, and EVA objectives (1).md* - Analyse d'alignement stratégique des trois projets.
[^2]: *GESICA Business Case & Application Form* - Objectifs d'optimisation des ressources d'urgence et de coordination transfrontalière France-Suisse.
[^3]: *INFRASTRUCTURE.md* - Spécifications techniques des serveurs de production LiteRev.
[^4]: *EVA Application* - Objectifs d'accélération des revues systématiques de littérature et de conformité PRISMA.
[^5]: *ThemainideawouldbetofindaprojectusingLit(3).md* - Historique des échanges techniques et fonctionnels sur le développement de l'interface.
[^6]: *Artificial Intelligence in Emergency Medical Services for Health Emergencies and Disasters: A Systematic Review (2026)* - Revue systématique servant de base scientifique pour GESICA.
[^7]: *main.py & App.tsx (Dépôt Git LiteRev-Evidence)* - Code source actuel de l'application de production.
