# Spécifications Techniques des Sources de Données : LiteRev-Evidence
**Auteur :** Manus AI  
**Date :** 27 mai 2026  
**Version :** 1.0.0  

---

## 1. Introduction et Objectif

Ce document définit les spécifications techniques de chaque source de données validée pour l'intégration dans **LiteRev-Evidence** [^1]. Il sert de guide de référence pour le développement des connecteurs d'ingestion et des pipelines ETL (Extract, Transform, Load) du backend [^3].

L'objectif est d'alimenter de manière continue et structurée le **double graphe de connaissances** (EMS/crise pour GESICA et épidémique pour GeoAI4EI) en combinant la littérature scientifique et les données opérationnelles et environnementales de terrain [^1] [^4].

---

## 2. Spécifications de la Littérature Scientifique (Priorité P0)

Le corpus scientifique constitue le socle de connaissances. Les articles sont ingérés via leur titre, résumé et métadonnées (Stage 1), puis le texte intégral est extrait pour les articles pertinents afin d'alimenter les Evidence Panels et l'assistant conversationnel [^1] [^4].

| Source | API Endpoint Principal | Format de Données | Clé / Authentification | Fréquence de Mise à Jour |
| :--- | :--- | :--- | :--- | :--- |
| **A1 : PubMed** | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi` | XML / JSON | Optionnelle (recommandée pour quota) | Quotidienne |
| **A2 : PMC** | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc` | XML (Full Text) | Optionnelle (recommandée pour quota) | Quotidienne |
| **A3 : OpenAlex** | `https://api.openAlex.org/works` | JSON-LD | Optionnelle (Polite Pool via email) | Hebdomadaire |
| **A4 : CrossRef** | `https://api.crossref.org/works` | JSON | Optionnelle (Polite Pool via email) | Mensuelle |

### Variables Clés à Extraire (Moteur d'Evidence)
- **Métadonnées de base** : `title`, `abstract`, `authors`, `journal`, `publication_year`, `doi`, `url`.
- **Champs spécifiques** : `project_context` (`gesica`, `geoai4ei`, `eva`), `source_type` (`article`, `systematic_review`, `meta_analysis`, `clinical_trial`) [^1] [^7].

---

## 3. Spécifications des Données Opérationnelles GESICA (Priorité P1 & P2)

Ces données fournissent le contexte opérationnel réel de la région transfrontalière France-Suisse (Genève, Vaud, Neuchâtel, Ain, Haute-Savoie) [^5] [^6].

| Source | Type d'Accès | Format attendu | Fréquence d'Ingestion | Rôle Opérationnel |
| :--- | :--- | :--- | :--- | :--- |
| **D1 : SAMU / Centre 15** | Flux partenaire sécurisé | JSON / CSV anonymisé | Temps réel (ou agrégé 1h) | Prévision de la demande (Axe 2) [^4] |
| **D2 : 144 Vaud / ORCA** | Flux partenaire sécurisé | JSON / CSV anonymisé | Temps réel (ou agrégé 1h) | Prévision de la demande (Axe 2) [^4] |
| **D3 : HUG / CHUV Urgences** | API d'établissement | JSON structuré | Toutes les 15 minutes | Optimisation des ressources (Axe 3) [^5] |
| **D6 : TECHWAN / SAGA** | Intégration DB directe | PostgreSQL / REST | Temps réel | Gestion des scénarios de crise [^5] [^6] |

### Variables Opérationnelles Clés
- **Régulation médicale** : `timestamp`, `call_volume`, `incident_type` (médical, trauma, arrêt cardiaque), `severity_level` (P0, P1, P2, P3), `response_time_seconds` [^4].
- **Capacités hospitalières** : `available_beds_emergency`, `available_beds_icu`, `waiting_time_minutes`, `divert_status` (saturation).

---

## 4. Spécifications des Données Météo et Environnement (Priorité P1 & P3)

Les facteurs environnementaux sont des déclencheurs majeurs de vagues de chaleur, d'épidémies saisonnières ou d'accidents, impactant directement l'activité des secours [^4].

| Source | API Endpoint Principal | Format | Clé / Authentification | Variables Clés |
| :--- | :--- | :--- | :--- | :--- |
| **B1 : Météo-France** | `https://api.meteofrance.com/v1/` | JSON | Clé API Développeur | Alertes vigilance, températures max/min, précipitations |
| **B2 : MeteoSwiss** | `https://api.meteoswiss.ch/v1/` | JSON | Clé API Développeur | Températures, humidité, alertes canicule / grand froid |
| **B3 : Copernicus CDS** | `https://cds.climate.copernicus.eu/api/v2` | NetCDF / GRIB | Clé API CDS | Réanalyses ERA5, anomalies de température de surface |

---

## 5. Spécifications de la Surveillance Épidémiologique (Priorité P2 & P3)

Ces sources alimentent le graphe de connaissances épidémiques pour GeoAI4EI et servent d'indicateurs avancés pour la charge EMS de GESICA [^1] [^4].

| Source | API / Source URL | Format | Clé | Variables Épidémiques Clés |
| :--- | :--- | :--- | :--- | :--- |
| **C1 : Réseau Sentinelles** | `https://www.sentiweb.fr/api/v1/` | CSV / JSON | Gratuite | Taux d'incidence (grippe, gastro, varicelle) pour 100k hab. |
| **C2 : ECDC** | `https://opendata.ecdc.europa.eu/` | JSON / XML | Gratuite | Cas rapportés, décès, taux de positivité par pays/région [^1] |
| **C5 : Santé pub. France** | `https://www.data.gouv.fr/api/v1/` | CSV | Gratuite | Passages urgences OS-Médecins par classe d'âge et pathologie |
| **C6 : OFSP (Suisse)** | `https://www.covid19.admin.ch/api/` | JSON | Gratuite | Déclarations obligatoires de maladies transmissibles en Suisse |

---

## 6. Spécifications de la Population et du Territoire (Priorité P2)

Ces données structurelles permettent de normaliser les taux de demande (ex: appels pour 10 000 habitants) et de modéliser les temps de transport réels [^4].

| Source | API / Source URL | Format | Clé | Rôle dans les Modèles |
| :--- | :--- | :--- | :--- | :--- |
| **E1 : INSEE (France)** | `https://api.insee.fr/metadonnees/V1/` | JSON | Clé API INSEE | Pyramide des âges et population par commune (normalisation) |
| **E2 : OFS (Suisse)** | `https://www.pxweb.bfs.admin.ch/api/` | JSON-stat | Gratuite | Population résidente permanente par canton et commune |
| **F1 : HERE Traffic** | `https://traffic.ls.hereapi.com/traffic/6.2/` | JSON / XML | Clé API HERE | Vitesse du trafic temps réel et congestion routière |
| **F2 : OSM + OSRM** | `http://router.project-osrm.org/route/v1/` | JSON | Gratuite | Calcul de l'itinéraire le plus rapide pour les ambulances |

---

## 7. Pipeline d'Ingestion et Stockage Unifié

Le schéma de base de données PostgreSQL de LiteRev-Evidence est étendu pour accueillir ces données hétérogènes tout en préservant les performances de la recherche sémantique [^7].

```
                  ┌──────────────────────────────────────────┐
                  │          Pipeline d'Ingestion            │
                  └────────────────────┬─────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
   ┌─────────────────────────┐                  ┌─────────────────────────┐
   │ Littérature Scientifique│                  │  Données Opérationnelles│
   │   (PubMed, OpenAlex...) │                  │   (SAMU, Météo, ECDC...)│
   └────────────┬────────────┘                  └────────────┬────────────┘
                │                                            │
                ▼                                            ▼
   ┌─────────────────────────┐                  ┌─────────────────────────┐
   │  Base Documentaire Core │                  │  Séries Temporelles &   │
   │  (literature_document)  │                  │   Indicateurs de Crise  │
   └────────────┬────────────┘                  └────────────┬────────────┘
                │                                            │
                ▼                                            ▼
   ┌─────────────────────────┐                  ┌─────────────────────────┐
   │    Recherche Hybride    │                  │  Evidence-to-Scenario   │
   │   Sémantique + BM25     │                  │     Coupling Engine     │
   └────────────┬────────────┘                  └────────────┬────────────┘
                │                                            │
                └──────────────────────┬─────────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │   Aide à la Décision (UI)   │
                        │   Evidence Panels & QA      │
                        └─────────────────────────────┘
```

### Principes d'Intégration
- **Stockage Documentaire** : Les résumés et textes intégraux de la littérature scientifique sont stockés dans la table `literature_document` et segmentés dans `literature_chunk` avec leurs embeddings BGE-M3 associés [^7].
- **Séries Temporelles** : Les indicateurs opérationnels (volume d'appels, températures, taux d'incidence épidémique) sont stockés dans des tables dédiées indexées par date et zone géographique (`geographic_scope`) pour permettre des corrélations rapides [^4] [^7].
- **Mise en Relation (Couplage)** : Le moteur d'Evidence-to-Scenario interroge simultanément les indicateurs opérationnels en temps réel (ex: température de 38°C mesurée) et la base documentaire pour extraire les recommandations scientifiques associées (ex: protocole de canicule pour les personnes âgées) [^1] [^4] [^5].

---

## 8. Références

[^1]: *Designing a LiteRev-based project that aligns Horizon GeoAI4EI, GESICA, and EVA objectives (1).md* — Alignement stratégique, double graphe de connaissances, tâches WP3.
[^2]: *EVA Application (LiteRev-1.pdf)* — Spécifications de la plateforme de revue systématique.
[^3]: *INFRASTRUCTURE.md* — Configuration des serveurs de production LiteRev-Evidence.
[^4]: *Artificial Intelligence in Emergency Medical Services for Health Emergencies and Disasters: A Systematic Review (Kokou Laris Edjinedja et al., 2026)* — Justification scientifique des variables météo, épidémiques, de triage et de ressources.
[^5]: *A.9 Annexe Business-case GESICA* — Objectifs d'optimisation, données d'urgence et partenaires (HUG, CHUV, TECHWAN).
[^6]: *FCS Application Form GESICA (Ref: 20510)* — Cadre d'interopérabilité avec la suite SAGA et les services de secours transfrontaliers.
[^7]: *Audit technique LiteRev-Evidence (mai 2026)* — Schéma de base de données actuel et structure de l'API FastAPI.
