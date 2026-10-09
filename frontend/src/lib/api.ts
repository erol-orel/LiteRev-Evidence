import type { SearchResult } from "../types/search";
import { currentLang, tStandalone } from "../i18n/LanguageProvider";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

// Clé d'écriture (X-API-Key) jointe aux mutations (endpoints protégés côté serveur
// par require_api_key). Source UNIQUEMENT depuis le stockage navigateur de l'admin :
// sessionStorage -> localStorage. On NE lit PLUS import.meta.env.VITE_API_KEY : Vite
// « inline » les variables VITE_* dans le bundle JS *public*, ce qui exposerait le
// secret à tout visiteur du site. L'admin saisit la clé une seule fois (setApiKey) ;
// elle reste sur son appareil. Sans clé, l'accès est en lecture seule (mutations 401).
const API_KEY_STORAGE = "api_key";

export function getApiKey(): string {
  try {
    return (
      sessionStorage.getItem(API_KEY_STORAGE) ||
      localStorage.getItem(API_KEY_STORAGE) ||
      ""
    );
  } catch {
    return "";
  }
}

export function hasApiKey(): boolean {
  return getApiKey().length > 0;
}

export function clearApiKey(): void {
  try {
    sessionStorage.removeItem(API_KEY_STORAGE);
    localStorage.removeItem(API_KEY_STORAGE);
  } catch {
    /* stockage indisponible : rien à faire */
  }
}

export function setApiKey(key: string, persist = true): void {
  const trimmed = (key ?? "").trim();
  if (!trimmed) {
    clearApiKey();
    return;
  }
  try {
    (persist ? localStorage : sessionStorage).setItem(API_KEY_STORAGE, trimmed);
  } catch {
    /* stockage indisponible (navigation privée) : on ignore silencieusement */
  }
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const token = getApiKey();
  return token ? { "X-API-Key": token, ...extra } : { ...extra };
}

// ─── Résilience réseau : retries + messages d'erreur lisibles ────────────────
// Le frontend affiche le message d'erreur tel quel (cf. ErrorBox). Avant, chaque
// hoquet transitoire (429 sous charge, 502/503 pendant un déploiement) remontait
// un « HTTP 429 » brut. On (1) réessaie automatiquement les statuts transitoires
// avec back-off, et (2) traduit les statuts en messages compréhensibles.

const _RETRYABLE_5XX = new Set([502, 503, 504]);

function _isGet(init?: RequestInit): boolean {
  const m = (init?.method ?? "GET").toUpperCase();
  return m === "GET" || m === "HEAD";
}

function _sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function _backoffMs(resp: Response, attempt: number): number {
  // Respecte Retry-After mais le plafonne : inutile de figer l'UI 60 s - mieux
  // vaut quelques tentatives courtes puis un message clair.
  const ra = Number(resp.headers.get("Retry-After"));
  if (Number.isFinite(ra) && ra > 0) return Math.min(ra * 1000, 4000);
  return Math.min(500 * 2 ** attempt, 4000) + Math.floor(Math.random() * 250);
}

/**
 * fetch() avec retries sur statuts transitoires. Un 429 est toujours réessayé
 * (la requête a été rejetée AVANT traitement) ; les 502/503/504 ne sont réessayés
 * que pour les requêtes idempotentes (GET/HEAD) afin de ne pas rejouer un POST.
 * Les erreurs réseau / abort se propagent à l'identique (pas de changement de
 * comportement par rapport à fetch()).
 */
export async function safeFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
  opts: { retries?: number; retryOn5xx?: boolean } = {},
): Promise<Response> {
  const retries = opts.retries ?? 2;
  const allow5xx = opts.retryOn5xx ?? _isGet(init);
  for (let attempt = 0; ; attempt++) {
    const resp = await globalThis.fetch(input, init);
    const retryable =
      resp.status === 429 || (allow5xx && _RETRYABLE_5XX.has(resp.status));
    if (!retryable || attempt >= retries) return resp;
    await _sleep(_backoffMs(resp, attempt));
  }
}

/** Traduit un statut HTTP en message utilisateur (langue courante). */
export function httpMessage(status: number): string {
  if (status === 429) return tStandalone("errors.tooManyRequests");
  if (status === 401 || status === 403) return tStandalone("errors.unauthorized");
  if (status === 404) return tStandalone("errors.notFound");
  if (status === 502 || status === 503 || status === 504)
    return tStandalone("errors.serviceUnavailable");
  if (status >= 500) return tStandalone("errors.serverError");
  return `${tStandalone("errors.genericPrefix")} ${status}.`;
}

export interface FilterOption {
  value: string | number;
  label: string;
}

export interface FilterOptions {
  source?: FilterOption[];
  sourceType?: FilterOption[];
  diseaseOrCondition?: FilterOption[];
  scenarioType?: FilterOption[];
  geographicScope?: FilterOption[];
  evidenceCategory?: FilterOption[];
  year?: FilterOption[];
}

export interface ApiFilterOptions {
  source?: FilterOption[];
  source_type?: FilterOption[];
  disease_or_condition?: FilterOption[];
  scenario_type?: FilterOption[];
  geographic_scope?: FilterOption[];
  evidence_category?: FilterOption[];
  year?: FilterOption[];
}

export interface DocumentChunk {
  id?: number;
  chunkIndex: number;
  content: string;
  chunkType?: string | null;
  sectionLabel?: string | null;
  charStart?: number | null;
  charEnd?: number | null;
  tokenCount?: number | null;
  chunkWeight?: number | null;
  metadataJson?: Record<string, unknown> | null;
}

export interface ApiDocumentChunk {
  id?: number;
  chunk_index: number;
  content: string;
  chunk_type?: string | null;
  section_label?: string | null;
  char_start?: number | null;
  char_end?: number | null;
  token_count?: number | null;
  chunk_weight?: number | null;
  metadata_json?: Record<string, unknown> | null;
}

export interface DocumentDetail {
  id: number;
  source?: string | null;
  title?: string | null;
  abstract?: string | null;
  year?: number | null;
  url?: string | null;
  externalId?: string | null;
  projectContext?: string | null;
  sourceType?: string | null;
  diseaseOrCondition?: string | null;
  scenarioType?: string | null;
  geographicScope?: string | null;
  evidenceCategory?: string | null;
  // Métadonnées « santé publique » + types normalisés (vocabulaire contrôlé côté backend).
  authors?: string | null;
  journal?: string | null;
  doi?: string | null;
  country?: string | null;
  studyDesign?: string | null;   // devis normalisé (_STUDY_DESIGN_CASE)
  articleType?: string | null;   // type d'article normalisé (_PUB_TYPE_CASE)
}

export interface ApiDocumentDetail {
  id: number;
  source?: string | null;
  title?: string | null;
  abstract?: string | null;
  year?: number | null;
  url?: string | null;
  external_id?: string | null;
  project_context?: string | null;
  source_type?: string | null;
  disease_or_condition?: string | null;
  scenario_type?: string | null;
  geographic_scope?: string | null;
  evidence_category?: string | null;
  authors?: string | null;
  journal?: string | null;
  doi?: string | null;
  country?: string | null;
  study_design?: string | null;
  article_type?: string | null;
}

export interface DocumentDetailResponse {
  document: DocumentDetail;
  chunks: DocumentChunk[];
}

export interface ApiDocumentDetailResponse {
  document: ApiDocumentDetail;
  chunks: ApiDocumentChunk[];
}

// --- MAPPER FUNCTIONS ---

function mapFilterOptionsFromApi(apiOpts: ApiFilterOptions): FilterOptions {
  return {
    source: apiOpts.source,
    sourceType: apiOpts.source_type,
    diseaseOrCondition: apiOpts.disease_or_condition,
    scenarioType: apiOpts.scenario_type,
    geographicScope: apiOpts.geographic_scope,
    evidenceCategory: apiOpts.evidence_category,
    year: apiOpts.year,
  };
}

function mapDocumentDetailFromApi(apiDoc: ApiDocumentDetail): DocumentDetail {
  return {
    id: apiDoc.id,
    source: apiDoc.source,
    title: apiDoc.title,
    abstract: apiDoc.abstract,
    year: apiDoc.year,
    url: apiDoc.url,
    externalId: apiDoc.external_id,
    projectContext: apiDoc.project_context,
    sourceType: apiDoc.source_type,
    diseaseOrCondition: apiDoc.disease_or_condition,
    scenarioType: apiDoc.scenario_type,
    geographicScope: apiDoc.geographic_scope,
    evidenceCategory: apiDoc.evidence_category,
    authors: apiDoc.authors,
    journal: apiDoc.journal,
    doi: apiDoc.doi,
    country: apiDoc.country,
    studyDesign: apiDoc.study_design,
    articleType: apiDoc.article_type,
  };
}

function mapDocumentChunkFromApi(apiChunk: ApiDocumentChunk): DocumentChunk {
  return {
    id: apiChunk.id,
    chunkIndex: apiChunk.chunk_index,
    content: apiChunk.content,
    chunkType: apiChunk.chunk_type,
    sectionLabel: apiChunk.section_label,
    charStart: apiChunk.char_start,
    charEnd: apiChunk.char_end,
    tokenCount: apiChunk.token_count,
    chunkWeight: apiChunk.chunk_weight,
    metadataJson: apiChunk.metadata_json,
  };
}

// --- API FUNCTIONS ---

export async function getFilterOptions(): Promise<FilterOptions> {
  const response = await safeFetch(`${API_BASE_URL}/filters-options`);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(
      text || `Filter options failed with status ${response.status}`,
    );
  }

  const apiData: ApiFilterOptions = await response.json();
  return mapFilterOptionsFromApi(apiData);
}

export async function fetchDocumentDetail(
  documentId: number,
): Promise<DocumentDetailResponse> {
  const response = await safeFetch(`${API_BASE_URL}/documents/${documentId}`);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(
      text || `Document detail failed with status ${response.status}`,
    );
  }

  const apiData: ApiDocumentDetailResponse = await response.json();
  return {
    document: mapDocumentDetailFromApi(apiData.document),
    chunks: (apiData.chunks || []).map(mapDocumentChunkFromApi),
  };
}

// --- SIGNALS / STATS TYPES ---

export interface GesicaSignals {
  demandSignals: string[];
  resourceTypes: string[];
  interventionTypes: string[];
  operationalSettings: string[];
  scenarioTags: string[];
  forecastHorizon: string | null;
  crossBorder: boolean;
  crossBorderSignals: string[];
  crisisSignals: string[];
  evidenceStrength: "weak" | "moderate" | "strong";
  uncertaintyHandling: string[];
  reportedMetrics: string[];
  isEmsOrCrisisRelevant: boolean;
}

export interface EvidenceSummaryResponse {
  document: DocumentDetail;
  summary: {
    projectContext: string | null;
    scenarioType: string | null;
    evidenceCategory: string | null;
    geographicScope: string | null;
    diseaseOrCondition: string | null;
  };
  gesicaSignals: GesicaSignals;
  chunkCount: number;
}

export interface ApiEvidenceSummaryResponse {
  document: ApiDocumentDetail;
  summary: {
    project_context: string | null;
    scenario_type: string | null;
    evidence_category: string | null;
    geographic_scope: string | null;
    disease_or_condition: string | null;
  };
  gesica_signals: {
    demand_signals: string[];
    resource_types: string[];
    intervention_types: string[];
    operational_settings: string[];
    scenario_tags: string[];
    forecast_horizon: string | null;
    cross_border: boolean;
    cross_border_signals: string[];
    crisis_signals: string[];
    evidence_strength: "weak" | "moderate" | "strong";
    uncertainty_handling: string[];
    reported_metrics: string[];
    is_ems_or_crisis_relevant: boolean;
  };
  chunk_count: number;
}

export interface CorpusStats {
  totalDocuments: number;
  totalChunks: number;
  byProject: Record<string, number>;
  bySource: Record<string, number>;
  byYear: Record<string, number>;
}

export interface GesicaStats {
  totalDocuments: number;
  evidenceStrengthDistribution: Record<string, number>;
  uncertaintyMethods: Record<string, number>;
  forecastHorizons: Record<string, number>;
}

export interface GesicaScenario {
  id: string;
  title: string;
  labelShort?: string | null;
  description: string;
  query?: string;   // requête d'origine (user scenarios) - pour un libellé localisé
  cluster: string;
  articleCount: number;
  livingEvidenceNote: string;
  recommendedActions: string[];
  model?: { has_model: boolean; family?: string; metric?: string; metric_value?: number | null };
  hidden?: boolean;
  included_count?: number;
  excluded_count?: number;
  kappa_score?: number | null;
  relevantArticles: Array<{
    id: number;
    title: string;
    abstract: string | null;
    year: number | null;
    source: string;
    url: string | null;
    authors: string | null;
    doi: string | null;
    journal: string | null;
    keywords: string | null;
    language: string | null;
    study_design: string | null;
    sample_size: number | null;
    country: string | null;
    citation_count: number | null;
    open_access: boolean | null;
    has_fulltext?: boolean;
  }>;
}

// ─── Fulltext Stats ───────────────────────────────────────────────────────────
export interface FulltextStats {
  corpus: {
    total_documents: number;
    docs_with_fulltext: number;
    docs_abstract_only: number;
    fulltext_coverage_pct: number;
    duplicates?: number;
    unique_documents?: number;
  };
  chunks?: {
    total: number;
    fulltext: number;
    abstract: number;
    other: number;
  };
  embeddings: {
    total_chunks: number;
    chunks_with_embedding: number;
    chunks_pending?: number;
    embedding_coverage_pct: number;
  };
  hybrid_search: {
    active: boolean;
    openai_key_present: boolean;
    embeddings_available: boolean;
    mode: string;
    note: string;
  };
  by_source: Array<{
    source: string;
    total: number;
    with_fulltext: number;
    abstract_only: number;
    fulltext_pct: number;
  }>;
  sample_fulltext_docs: Array<{
    id: number;
    title: string;
    source: string;
    year: number | null;
    url: string | null;
    authors: string | null;
    doi: string | null;
  }>;
}
export async function fetchFulltextStats(): Promise<FulltextStats> {
  const response = await safeFetch(`${API_BASE_URL}/corpus/fulltext-stats`);
  if (!response.ok) throw new Error(httpMessage(response.status));
  return response.json();
}

// ─── Maintenance corpus (admin) : purge doublons + normalisation chunks ──────
export interface CorpusMaintenanceReport {
  dry_run: boolean;
  duplicates: {
    documents: number;
    chunks_cascade: number;
    article_scenarios: number;
    deleted_documents?: number;
  };
  legacy_chunks: {
    breakdown: Array<{ chunk_type: string; count: number; embedded: number }>;
    junk_to_delete: number;
    redundant_to_delete: number;
    unique_to_reclassify: number;
    deleted_junk?: number;
    deleted_redundant?: number;
    reclassified?: number;
  };
  backups: string[];
}
export async function corpusMaintenance(dryRun: boolean): Promise<CorpusMaintenanceReport> {
  const response = await safeFetch(`${API_BASE_URL}/admin/corpus-maintenance`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ dry_run: dryRun }),
  });
  if (!response.ok) throw new Error(httpMessage(response.status));
  return response.json();
}

// Force l'indexation (embedding) des chunks « en attente » à la demande (admin).
export interface EmbedPendingResult {
  embedded: number;
  remaining: number | null;
  cooldown?: boolean;
  error?: string;
}
export async function embedPending(limit = 200): Promise<EmbedPendingResult> {
  const r = await safeFetch(`${API_BASE_URL}/admin/embed-pending?limit=${limit}`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export interface AskRequest {
  question: string;
  projectContext?: string;
  filters?: Record<string, any>;
}

export interface AskResponse {
  answer: string;
  sources: {
    documentId: number;
    title: string;
    year: number | null;
    url: string | null;
    source: string;
    projectContext: string;
    evidenceStrength: string;
  }[];
}

export async function fetchEvidenceSummary(
  documentId: number,
): Promise<EvidenceSummaryResponse> {
  const response = await safeFetch(`${API_BASE_URL}/evidence-summary/${documentId}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Evidence summary failed with status ${response.status}`);
  }
  const apiData: ApiEvidenceSummaryResponse = await response.json();
  return {
    document: mapDocumentDetailFromApi(apiData.document),
    summary: {
      projectContext: apiData.summary.project_context,
      scenarioType: apiData.summary.scenario_type,
      evidenceCategory: apiData.summary.evidence_category,
      geographicScope: apiData.summary.geographic_scope,
      diseaseOrCondition: apiData.summary.disease_or_condition,
    },
    gesicaSignals: {
      demandSignals: apiData.gesica_signals.demand_signals,
      resourceTypes: apiData.gesica_signals.resource_types,
      interventionTypes: apiData.gesica_signals.intervention_types,
      operationalSettings: apiData.gesica_signals.operational_settings,
      scenarioTags: apiData.gesica_signals.scenario_tags,
      forecastHorizon: apiData.gesica_signals.forecast_horizon,
      crossBorder: apiData.gesica_signals.cross_border,
      crossBorderSignals: apiData.gesica_signals.cross_border_signals,
      crisisSignals: apiData.gesica_signals.crisis_signals,
      evidenceStrength: apiData.gesica_signals.evidence_strength,
      uncertaintyHandling: apiData.gesica_signals.uncertainty_handling,
      reportedMetrics: apiData.gesica_signals.reported_metrics,
      isEmsOrCrisisRelevant: apiData.gesica_signals.is_ems_or_crisis_relevant,
    },
    chunkCount: apiData.chunk_count,
  };
}

export async function fetchCorpusStats(): Promise<CorpusStats> {
  const response = await safeFetch(`${API_BASE_URL}/corpus/stats`);
  if (!response.ok) throw new Error(`Corpus stats failed with status ${response.status}`);
  const data = await response.json();
  return {
    totalDocuments: data.total_documents,
    totalChunks: data.total_chunks,
    byProject: data.by_project,
    bySource: data.by_source,
    byYear: data.by_year,
  };
}

export interface HeatmapScenarioEntry {
  name: string;
  sources: Record<string, { total: number; fulltext: number }>;
}

export interface CorpusStatsByYear {
  byYear: Record<string, number>;
  scenarioByYear: Record<string, Record<string, number>>;
  // Clé = scenario_id ; valeur = nom + par source {total, texte intégral}.
  heatmapScenarioSource: Record<string, HeatmapScenarioEntry>;
}

export async function fetchCorpusStatsByYear(): Promise<CorpusStatsByYear> {
  const response = await safeFetch(`${API_BASE_URL}/corpus/stats/by-year`);
  if (!response.ok) throw new Error(`Corpus stats by-year failed with status ${response.status}`);
  const data = await response.json();
  return {
    byYear: data.by_year,
    scenarioByYear: data.scenario_by_year,
    heatmapScenarioSource: data.heatmap_scenario_source,
  };
}

export async function fetchGesicaStats(): Promise<GesicaStats> {
  const response = await safeFetch(`${API_BASE_URL}/gesica/stats`);
  if (!response.ok) throw new Error(`LiteRev stats failed with status ${response.status}`);
  const data = await response.json();
  return {
    totalDocuments: data.total_documents,
    evidenceStrengthDistribution: data.evidence_strength_distribution,
    uncertaintyMethods: data.uncertainty_methods,
    forecastHorizons: data.forecast_horizons,
  };
}

export async function fetchGesicaScenarios(): Promise<GesicaScenario[]> {
  // The built-in catalogue is stored in French; the server renders it in the UI language.
  const response = await safeFetch(`${API_BASE_URL}/gesica/scenarios?lang=${currentLang()}`);
  if (!response.ok) throw new Error(`LiteRev scenarios failed with status ${response.status}`);
  const data: Array<{
    id: string;
    title: string;
    label_short?: string | null;
    description: string;
    cluster: string;
    article_count: number;
    living_evidence_note: string;
    recommended_actions: string[];
    relevant_articles: Array<{
      id: number;
      title: string;
      abstract: string | null;
      year: number | null;
      source: string;
      url: string | null;
      authors: string | null;
      doi: string | null;
      journal: string | null;
      keywords: string | null;
      language: string | null;
      study_design: string | null;
      sample_size: number | null;
      country: string | null;
      citation_count: number | null;
      open_access: boolean | null;
      has_fulltext?: boolean;
    }>;
  }> = await response.json();
  return data.map((s) => ({
    id: s.id,
    title: s.title,
    labelShort: s.label_short ?? null,
    description: s.description,
    cluster: s.cluster,
    articleCount: s.article_count,
    livingEvidenceNote: s.living_evidence_note,
    recommendedActions: s.recommended_actions,
    relevantArticles: s.relevant_articles,
  }));
}

// Actions recommandées (génération LLM paresseuse + cache côté serveur).
export async function getRecommendedActions(
  scenarioId: string,
): Promise<{ status: string; actions: string[]; generated_at?: string }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/recommended-actions?lang=${currentLang()}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function askAssistant(req: AskRequest): Promise<AskResponse> {
  const response = await safeFetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      question: req.question,
      project_context: req.projectContext || null,
      filters: req.filters || null,
      lang: currentLang(),
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Ask failed with status ${response.status}`);
  }

  const data = await response.json();
  return {
    answer: data.answer,
    sources: (data.sources || []).map((s: any) => ({
      documentId: s.document_id,
      title: s.title,
      year: s.year,
      url: s.url,
      source: s.source,
      projectContext: s.project_context,
      evidenceStrength: s.evidence_strength,
    })),
  };
}

export function getReadableExcerpt(
  result: SearchResult,
  detail: DocumentDetailResponse | null,
): string {
  if (result.highlight?.trim()) return result.highlight;
  if (result.content?.trim()) return result.content;
  if (detail?.document?.abstract?.trim()) return detail.document.abstract;
  if (detail?.chunks?.length) return detail.chunks[0]?.content ?? "";
  return "";
}

// ─── P5 TERRAIN DATA TYPES ───────────────────────────────────────────────────

export interface TerrainMeteo {
  source: string;
  coordinates: { latitude: number; longitude: number };
  station: string;
  temperature: number;
  apparent_temperature: number;
  humidity: number;
  wind_speed: number;
  precipitation: number;
  alert_level: "none" | "warning" | "danger";
  alert_description: string;
  impact_on_ems: string;
  architecture_note: string;
}

export interface TerrainGeo {
  source: string;
  origin: { latitude: number; longitude: number; label: string };
  destination: { latitude: number; longitude: number; label: string };
  distance_km: number;
  base_duration_min: number;
  traffic_congestion_factor: number;
  cross_border_delay_min: number;
  total_estimated_response_time_min: number;
  routing_status: string;
  coordination_action: string;
  architecture_note: string;
}

export interface TerrainEpidemicDisease {
  name: string;
  incidence_per_100k_france: number;
  incidence_per_100k_switzerland: number;
  epidemic_threshold: number;
  status: "under_threshold" | "warning" | "epidemic";
  trend: "increasing" | "stable" | "decreasing";
  last_update: string;
}

export interface TerrainEpidemic {
  source: string;
  region: string;
  diseases: TerrainEpidemicDisease[];
  global_ems_impact_risk: "low" | "moderate" | "high";
  recommended_action: string;
  architecture_note: string;
}

// ─── P5 TERRAIN API FUNCTIONS ────────────────────────────────────────────────

export async function fetchTerrainMeteo(lat = 46.2044, lon = 6.1432): Promise<TerrainMeteo> {
  const response = await safeFetch(`${API_BASE_URL}/terrain/meteo?lat=${lat}&lon=${lon}`);
  if (!response.ok) throw new Error(`Terrain meteo failed with status ${response.status}`);
  return response.json();
}

export async function fetchTerrainGeo(
  origLat = 46.2044, origLon = 6.1432,
  destLat = 46.1925, destLon = 6.2388
): Promise<TerrainGeo> {
  const url = `${API_BASE_URL}/terrain/geo?orig_lat=${origLat}&orig_lon=${origLon}&dest_lat=${destLat}&dest_lon=${destLon}`;
  const response = await safeFetch(url);
  if (!response.ok) throw new Error(`Terrain geo failed with status ${response.status}`);
  return response.json();
}

export async function fetchTerrainEpidemic(region = "transborder"): Promise<TerrainEpidemic> {
  const response = await safeFetch(`${API_BASE_URL}/terrain/epidemic?region=${region}`);
  if (!response.ok) throw new Error(`Terrain epidemic failed with status ${response.status}`);
  return response.json();
}

// ─── P5 TERRAIN EXTENDED TYPES ────────────────────────────────────────────────

export interface TerrainDemographics {
  postal_code: string;
  commune: string;
  country: string;
  population: number;
  density_per_km2: number;
  age_over_65_pct: number;
  ems_risk_multiplier: number;
  source: string;
  architecture_note: string;
}

export interface TerrainPharmacy {
  name: string;
  street: string;
  city: string;
  is_dispensary: boolean;
  opening_hours: string;
  coordinates: { latitude: number | null; longitude: number | null };
}

export interface TerrainMedicationAlert {
  medication: string;
  status: "normal" | "tension" | "rupture";
  country_affected: string;
  recommendation: string;
  source: string;
}

export interface TerrainPharmacies {
  source: string;
  pharmacies_nearby: TerrainPharmacy[];
  critical_medication_alerts: TerrainMedicationAlert[];
  architecture_note: string;
}

export interface TerrainSignal {
  id: string;
  source: string;
  title: string;
  content: string;
  date: string;
  reliability_score: number;
  severity: "low" | "moderate" | "high";
  geo_scope: string;
  impact_on_ems?: string;
  impact_on_hospital?: string;
  impact_on_gesica?: string;  // legacy
  impact_on_geoai4ei?: string;  // legacy
}

export interface TerrainInformalSignals {
  source: string;
  active_signals: TerrainSignal[];
  architecture_note: string;
}

// ─── P5 TERRAIN EXTENDED API FUNCTIONS ────────────────────────────────────────

export async function fetchTerrainDemographics(postalCode = "74100"): Promise<TerrainDemographics> {
  const response = await safeFetch(`${API_BASE_URL}/terrain/demographics?postal_code=${postalCode}`);
  if (!response.ok) throw new Error(`Terrain demographics failed with status ${response.status}`);
  return response.json();
}

export async function fetchTerrainPharmacies(lat = 46.2044, lon = 6.1432): Promise<TerrainPharmacies> {
  const response = await safeFetch(`${API_BASE_URL}/terrain/pharmacies?lat=${lat}&lon=${lon}`);
  if (!response.ok) throw new Error(`Terrain pharmacies failed with status ${response.status}`);
  return response.json();
}

export async function fetchTerrainInformalSignals(): Promise<TerrainInformalSignals> {
  const response = await safeFetch(`${API_BASE_URL}/terrain/informal-signals`);
  if (!response.ok) throw new Error(`Terrain informal signals failed with status ${response.status}`);
  return response.json();
}

// ─── P5 TERRAIN CLIMATE (COPERNICUS CDS) ──────────────────────────────────────

export interface TerrainClimate {
  source: string;
  region: string;
  coordinates: { latitude: number; longitude: number };
  climatology: {
    historical_mean_temp_may_c: number;
    current_anomaly_c: number;
    heatwave_hazard_index: "low" | "moderate" | "high" | "critical";
    soil_moisture_deficit_percent: number;
    extreme_precipitation_risk: "low" | "moderate" | "high";
  };
  projections_2030: {
    expected_heatwave_days_increase_per_year: number;
    expected_heavy_precipitation_increase_percent: number;
    ems_vulnerability_factor: string;
  };
  api_status: string;
  message?: string;
}

export async function fetchTerrainClimate(lat = 46.2044, lon = 6.1432): Promise<TerrainClimate> {
  const response = await safeFetch(`${API_BASE_URL}/terrain/climate?lat=${lat}&lon=${lon}`);
  if (!response.ok) throw new Error(`Terrain climate failed with status ${response.status}`);
  return response.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// LiteRev Scenario Detail : Phase 2 Enrichissement
// ─────────────────────────────────────────────────────────────────────────────

export interface AlertThreshold {
  label: string;
  condition: string;
}

export interface ModelInfo {
  algorithm: string;
  variables: string[];
  output: string;
  update_frequency: string;
}

export interface VariableDetail {
  definition: string;
  plugged: boolean;
  source: string;
}

/** La nature d'une question : une revue de littérature, ou un scénario qui se
 *  termine par un modèle. Absente d'une ancienne réponse, elle vaut "predictive",
 *  c'est-a-dire tout, comme avant. */
export type ScenarioKind = 'review' | 'predictive';

/** La seule capacité qui retire quelque chose à un scénario : la moitié modèle.
 *  Tout le reste, y compris la littérature grise des rapports de situation, est une
 *  source de littérature que les deux natures lisent. */
export type ScenarioCapability = 'model_spec';

export interface ScenarioDetail {
  id: string;
  title: string;
  kind?: ScenarioKind;
  capabilities?: ScenarioCapability[];
  description: string;
  cluster: string;
  query?: string;   // requête d'origine (user scenarios) - pour un libellé localisé
  // Multi-facet search (user scenarios): the WHOLE expression "(A) AND (B)" and the
  // ordered facets with the operator actually applied to each one (none on the main
  // query). `query` alone is only the main facet, which hid the AND/OR.
  combined_query?: string;
  facets?: ScenarioFacet[];
  combinator?: "union" | "intersection" | null;
  recommended_actions: string[];
  boolean_queries: string[];
  nl_queries: string[];
  evidence_extraction_prompt: string;
  model_info: ModelInfo;
  alert_thresholds: {
    green: AlertThreshold;
    orange: AlertThreshold;
    red: AlertThreshold;
  };
  databases?: string[];
  outcome_definition?: string;
  variables_detail?: Record<string, VariableDetail>;
  keywords?: string[];
  clinical_rationale?: string;
  corpus_stats: {
    total: number;
    with_fulltext: number;
    years_covered: number;
    journals_count: number;
    year_min: number | null;
    year_max: number | null;
  };
  /** Le jeu de compteurs commun, d'un seul instantané (cf. CorpusCounts). */
  counts?: CorpusCounts;
}

export interface CorpusArticle {
  id: number;
  title: string;
  abstract: string | null;
  year: number | null;
  source: string;
  url: string | null;
  authors: string | null;
  doi: string | null;
  journal: string | null;
  keywords: string | null;
  language: string | null;
  study_design: string | null;
  sample_size: number | null;
  country: string | null;
  citation_count: number | null;
  open_access: boolean | null;
  has_fulltext: boolean;
  is_new?: boolean | null;
  similarity_score?: number | null;
  rerank_score?: number | null;
  screening_status?: string | null;
  reviewer_1_status?: string | null;
  pmid?: string | null;
  publication_type?: string | null;
  quality_score?: number | null;
}

export interface ScenarioCorpus {
  scenario_id: string;
  total: number;
  above_threshold?: number;
  /** What the extractions actually read: the shared relevance gate, counted. Distinct
   *  from `above_threshold`, which is only a split by score: a narrowing can put
   *  above-threshold articles out of scope, and a reviewer can rescue articles below it.
   *  Never derive one from the other. */
  relevant?: number;
  below_threshold?: number;
  unscored?: number;
  from_local?: number | null;
  newly_fetched?: number | null;
  docs_with_fulltext?: number;
  docs_abstract_only?: number;
  source_breakdown?: Record<string, number>;
  rerank_running?: boolean;
  threshold?: number;
  /** Taille de la VUE quand elle est filtrée (année, source, texte intégral) ; null
   *  sinon. `total` reste la taille du corpus : une vue filtrée ne le rétrécit pas. */
  filtered_total?: number | null;
  /** Le jeu de compteurs commun, d'un seul instantané (cf. CorpusCounts). */
  counts?: CorpusCounts;
  offset: number;
  limit: number;
  articles: CorpusArticle[];
  year_distribution: Array<{ year: number; count: number }>;
  source_distribution: Array<{ source: string; count: number }>;
  is_user_scenario?: boolean;
  scenario_title?: string;
}

export interface ClusterTopic {
  topic_id: number;
  top_words: string[];
  weight: number;
}

export interface ClusterPoint {
  id: number;
  title: string;
  year: number | null;
  x: number;
  y: number;
}

/** One facet of a saved multi-query search as returned by the detail endpoint. */
export interface ScenarioFacet {
  kind: "boolean" | "natural";
  text: string;
  op: "and" | "or" | null;   // operator applied vs the running result; null on the main facet
}

export interface ClusterResult {
  cluster_id: number;
  cluster_name: string;
  is_noise: boolean;
  n_docs: number;
  center_x: number;
  center_y: number;
  top_words: string[];
  summary: string;
  // LLM summaries per language ("fr"/"en"); `summary` is the one for the requested lang.
  summaries?: Record<string, string>;
  points_total?: number;   // all points of the cluster (points[] may be a sample)
  representative_doc: {
    id: number;
    title: string;
    year: number | null;
    journal: string | null;
  };
  points?: ClusterPoint[];
}

export interface ScenarioClustering {
  scenario_id: string;
  n_docs: number;
  // Eligible documents; > n_docs when the clustering was capped to the most
  // relevant CLUSTER_MAX_DOCS articles (the caption says so).
  n_docs_total?: number;
  n_clusters?: number;
  clusters: ClusterResult[];
  topics: ClusterTopic[];
  message?: string;
  message_code?: string;
  status?: "running" | "done" | "error" | "not_started" | string;
  lang?: string;          // language of the served summaries
  from_cache?: boolean;
  // UMAP points are capped per response (sampled evenly per cluster); the cache
  // keeps them all. points_shown < points_total ⇒ the plot is a sample.
  points_total?: number;
  points_shown?: number;
}

export interface ScenarioRagResponse {
  answer: string;
  sources: Array<{
    document_id: number;
    title: string;
    year: number | null;
    url: string | null;
    source: string;
    authors: string | null;
    journal: string | null;
    doi: string | null;
    score: number;
  }>;
  scenario_id: string;
  model?: string;
}

export type ScenarioRagSource = ScenarioRagResponse['sources'][number];

export interface ScenarioPrisma {
  scenario_id: string;
  scenario_title: string;
  identification: {
    total_records: number;
    by_source: Record<string, number>;
    duplicates_removed: number;
    embedded: number;
    // Chiffres de la RECHERCHE (populate/rebuild) quand ils existent : enregistrements
    // ramenés par source (recoupements compris), doublons, uniques, retirés pour
    // d'autres raisons. figures_from="corpus" = scénario antérieur à cette
    // comptabilité : les nombres viennent du corpus déjà dédupliqué.
    unique_records?: number;
    // Retraits avant screening, ventilés : sans résumé (règle qualité), hors requête
    // (source par mots-clés ne correspondant pas au booléen en local), résiduel.
    removed_no_abstract?: number;
    removed_not_matching?: number;
    removed_other_reasons?: number;
    // Corpus changes since the search that produced the figures (their own lines in the panel).
    added_after_search?: number;
    removed_after_search?: number;
    records_screened_at_search?: number;
    removed_before_screening?: number;
    records_screened?: number;
    duplicate_records_across_sources?: number;
    duplicate_rows_in_database?: number;
    figures_from?: "search_run" | "corpus";
    computed_at?: string | null;
    federation_incomplete?: boolean;
    /** What each launched source actually did: ok, empty, cached, skipped, error,
     *  cut_by_budget. A source that failed is not a source that was searched, and the
     *  table used to show neither its row nor its status while still counting it. */
    source_outcomes?: Record<string, string>;
    sources_launched?: number;
    sources_searched?: number;
    sources_failed?: string[];
    sources_skipped?: string[];
    sources_cut_off?: string[];
    /** PRISMA 2020 splits identification: databases searched, and other methods. The
     *  local library belongs to the second, and counting it in the first inflated both
     *  the identified total and the duplicates. */
    records_identified_databases?: number;
    records_identified_library?: number;
    per_source_cap?: number | null;
    /** The sources that received KEYWORDS instead of the boolean query, and the keywords
     *  they received. Past 1200 characters of portable boolean, five of the twelve fall
     *  back (OpenAlex's URL limit) and nothing said so, while PRISMA-S requires the
     *  strategy actually submitted to each database. */
    keyword_fallback_sources?: string[];
    keyword_fallback_query?: string | null;
    /** "populate" = sources were searched; "rebuild" = the boolean query was replayed
     *  over the local library and nothing was searched. The panel called both a search. */
    method?: string;
    /** What the last real search had established, kept when a rebuild replaces the
     *  figures, so the record of what was searched is not lost. */
    last_search?: {
      computed_at?: string | null;
      records_identified?: number;
      records_identified_databases?: number;
      records_by_source?: Record<string, number>;
      source_outcomes?: Record<string, string>;
      sources_searched?: number;
      sources_launched?: number;
      sources_failed?: string[];
      federation_incomplete?: boolean;
      per_source_cap?: number | null;
    } | null;
    // legacy
    total_records_identified?: number;
  };
  semantic_screening: {
    threshold: number;
    above_threshold: number;
    below_threshold: number;
    method: string;
  };
  full_text: {
    with_fulltext: number;
    without_fulltext: number;
    pct: number;
    note: string;
  };
  manual_curation: {
    included: number;
    excluded: number;
    pending: number;
    screening_complete: boolean;
    /** What "complete" means, in figures: a single decision used to flip
     *  screening_complete to true on a corpus of 6 564 articles. */
    screening_started?: boolean;
    screened?: number;
    to_screen?: number;
    manually_rescued: number;
    manually_vetoed: number;
    /** Why the excluded were excluded. One scope narrowing can account for most of the
     *  total, and a total alone cannot be written into a methods section. */
    excluded_by_reason?: Array<{ reason: string; articles: number }>;
  };
  evidence: {
    /** The relevant subset through the shared gate, so equal to counts.relevant,
     *  rerank threshold included. It used to be arithmetic over a similarity-only
     *  counter, which diverged the moment a rerank threshold was set. */
    total: number;
    ai_auto_selected: number;
    manually_rescued: number;
    with_fulltext: number;
    screening_complete: boolean;
    screening_started?: boolean;
    screened?: number;
    to_screen?: number;
  };
  // legacy fields kept for backward compat
  screening?: {
    records_screened: number;
    records_excluded_title_abstract: number;
    records_included_screening: number;
    records_awaiting_screening: number;
  };
  eligibility?: {
    fulltext_assessed: number;
    fulltext_retrieved: number;
    fulltext_not_retrieved: number;
    fulltext_excluded: number;
  };
  included?: {
    total_included: number;
    awaiting_assessment: number;
    screening_complete: boolean;
    note: string;
  };
}

export interface ScreeningProgress {
  scenario_id: string;
  total_in_db: number;
  duplicates: number;
  unique_articles: number;
  screened: number;
  included: number;
  excluded: number;
  awaiting: number;
  progress_pct: number;
  screening_complete: boolean;
}

export interface PicoData {
  P: string | null;
  I: string | null;
  C: string | null;
  O: string | null;
  study_design: string | null;
  pico_confidence: number | null;
  pico_notes: string | null;
}

export async function fetchScenarioDetail(scenarioId: string): Promise<ScenarioDetail> {
  const base = scenarioBase(scenarioId);
  const response = await safeFetch(`${base}/${scenarioId}/detail?lang=${currentLang()}`);
  if (!response.ok) throw new Error(httpMessage(response.status));
  return response.json();
}

export async function fetchScenarioCorpus(
  scenarioId: string,
  options?: {
    limit?: number;
    offset?: number;
    yearFrom?: number;
    yearTo?: number;
    fulltextOnly?: boolean;
    source?: string;
    threshold?: number;
    /** Chercher DANS le corpus (titre, résumé, auteurs, revue, mots-clés, DOI, PMID).
     *  La liste est paginée côté serveur : filtrer la page affichée ne chercherait
     *  que dans les cent premiers articles d'un corpus qui en compte des milliers. */
    q?: string;
    /** Restreindre au sous-ensemble pertinent (porte commune : seuil ou inclusion par
     *  un relecteur, jamais un exclu). */
    relevantOnly?: boolean;
    abstractChars?: number;   // truncate abstracts server-side (excerpt-only views)
  }
): Promise<ScenarioCorpus> {
  const params = new URLSearchParams();
  if (options?.limit) params.set('limit', String(options.limit));
  if (options?.offset) params.set('offset', String(options.offset));
  if (options?.yearFrom) params.set('year_from', String(options.yearFrom));
  if (options?.yearTo) params.set('year_to', String(options.yearTo));
  if (options?.fulltextOnly) params.set('fulltext_only', 'true');
  if (options?.source) params.set('source', options.source);
  if (options?.threshold != null) params.set('threshold', String(options.threshold));
  if (options?.q && options.q.trim()) params.set('q', options.q.trim());
  if (options?.relevantOnly) params.set('relevant_only', 'true');
  // Truncate abstracts server-side when only an excerpt is displayed (search results
  // page): 10,000 full abstracts weighed tens of MB for a 600-character snippet.
  if (options?.abstractChars != null) params.set('abstract_chars', String(options.abstractChars));
  const base = scenarioBase(scenarioId);
  const url = `${base}/${scenarioId}/corpus?${params}`;
  const response = await safeFetch(url);
  if (!response.ok) throw new Error(httpMessage(response.status));
  return response.json();
}

export async function fetchScenarioClustering(
  scenarioId: string,
  nClusters?: number
): Promise<ScenarioClustering> {
  const params = nClusters ? `?n_clusters=${nClusters}` : '';
  const langParam = `${params ? '&' : '?'}lang=${currentLang()}`;
  const base = scenarioBase(scenarioId);
  const response = await safeFetch(`${base}/${scenarioId}/clustering${params}${langParam}`);
  if (!response.ok) throw new Error(httpMessage(response.status));
  return response.json();
}

/** Poll a background clustering job; the result comes back in the current UI language. */
export async function fetchScenarioClusteringStatus(scenarioId: string): Promise<ScenarioClustering & { error?: string }> {
  const base = scenarioBase(scenarioId);
  const response = await safeFetch(`${base}/${scenarioId}/clustering/status?lang=${currentLang()}`);
  if (!response.ok) throw new Error(httpMessage(response.status));
  return response.json();
}

export async function fetchScenarioPrisma(scenarioId: string): Promise<ScenarioPrisma> {
  const base = scenarioBase(scenarioId);
  const response = await safeFetch(`${base}/${scenarioId}/prisma`);
  if (!response.ok) throw new Error(httpMessage(response.status));
  return response.json();
}

export interface UploadDatasetResponse {
  message: string;
  filename: string;
  size_bytes: number;
  detected_rows: number;
  detected_columns: string[];
  status: string;
  instructions: string;
}

export async function uploadScenarioDataset(
  scenarioId: string,
  file: File
): Promise<UploadDatasetResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  const base = scenarioBase(scenarioId);
  const response = await safeFetch(`${base}/${scenarioId}/upload-dataset`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  });
  
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Upload failed with status ${response.status}`);
  }

  return response.json();
}

export interface ModelDataUploadResponse {
  status: string;
  dataset_id?: number;
  n_rows?: number;
  n_cols?: number;
  validation?: ModelDataset['validation'];
  training_started?: boolean;
}

// Branche les données sur le pipeline modèle (valide vs data_template, stocke,
// et déclenche l'entraînement automatiquement si les données suffisent).
export async function uploadModelData(scenarioId: string, file: File): Promise<ModelDataUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/data`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Upload failed with status ${response.status}`);
  }
  return response.json();
}

export async function screenArticle(
  scenarioId: string,
  articleId: number,
  status: 'included' | 'excluded' | 'pending',
  reason?: string,
  notes?: string
): Promise<{ id: number; status: string; updated: boolean }> {
  const params = new URLSearchParams({ status });
  if (reason) params.set('reason', reason);
  if (notes) params.set('notes', notes);
  const base = scenarioBase(scenarioId);
  const response = await safeFetch(
    `${base}/${scenarioId}/articles/${articleId}/screen?${params}`,
    { method: 'POST', headers: authHeaders() }
  );
  if (!response.ok) throw new Error(httpMessage(response.status));
  return response.json();
}

export async function fetchScreeningProgress(scenarioId: string): Promise<ScreeningProgress> {
  const base = scenarioBase(scenarioId);
  const response = await safeFetch(`${base}/${scenarioId}/screening-progress`);
  if (!response.ok) throw new Error(httpMessage(response.status));
  return response.json();
}

export async function fetchArticlePico(
  scenarioId: string,
  articleId: number
): Promise<{ article_id: number; pico: PicoData | null; extracted_at: string | null }> {
  const base = scenarioBase(scenarioId);
  const response = await safeFetch(
    `${base}/${scenarioId}/articles/${articleId}/pico`
  );
  if (!response.ok) throw new Error(httpMessage(response.status));
  return response.json();
}

export async function extractPicoBatch(
  scenarioId?: string,
  limit = 100000
): Promise<{ extracted: number; skipped: number; errors: number; message: string }> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (scenarioId) params.set('scenario_id', scenarioId);
  const response = await safeFetch(`${API_BASE_URL}/pico/extract?${params}`, { method: 'POST', headers: authHeaders() });
  if (!response.ok) throw new Error(httpMessage(response.status));
  return response.json();
}

// ─── PICO Bulk ────────────────────────────────────────────────────────────────
export interface PicoBulkArticle {
  id: number;
  title: string;
  year: number | null;
  source: string;
  authors: string | null;
  doi: string | null;
  journal: string | null;
  study_design: string | null;
  pico_confidence: number | null;
  P: string | null;
  I: string | null;
  C: string | null;
  O: string | null;
  pico_notes: string | null;
  has_pico: boolean;
  pico_extracted_at: string | null;
  screening_status: string | null;
}

export interface PicoBulkResponse {
  scenario_id: string;
  total: number;
  with_pico: number;
  offset: number;
  limit: number;
  /** Rows in this page, and whether more remain (the endpoint pages, it never truncates silently). */
  returned?: number;
  truncated?: boolean;
  next_offset?: number | null;
  page_max?: number;
  articles: PicoBulkArticle[];
}

/** Every article of the scenario with its PICO, by following the pages to the end.
 *  The PICO tab claims to show "all articles": it must therefore hold all of them. */
export async function fetchAllScenarioPico(
  scenarioId: string,
  maxPages = 40,
): Promise<PicoBulkResponse> {
  const first = await fetchScenarioPicoBulk(scenarioId, 5000, 0);
  const all = [...first.articles];
  let next = first.truncated ? first.next_offset ?? null : null;
  for (let page = 1; next !== null && page < maxPages; page++) {
    const chunk = await fetchScenarioPicoBulk(scenarioId, 5000, next);
    all.push(...chunk.articles);
    next = chunk.truncated ? chunk.next_offset ?? null : null;
  }
  return { ...first, articles: all, returned: all.length, offset: 0,
           truncated: next !== null, next_offset: next };
}

export async function fetchScenarioPicoBulk(
  scenarioId: string,
  limit = 100000,
  offset = 0
): Promise<PicoBulkResponse> {
  const base = scenarioBase(scenarioId);
  const r = await safeFetch(`${base}/${scenarioId}/pico-bulk?limit=${limit}&offset=${offset}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Structured extraction (template shaped, one row per observation) ─────────
export interface ExtractionStatus {
  scenario_id: string;
  n_relevant: number;
  n_extracted: number;
  n_from_fulltext: number;
  n_from_abstract: number;
  n_pending: number;
  n_observations: number;
  n_given_up: number;
  running: boolean;
  job: { total: number; done: number; failed: number } | null;
  version: number;
}

export interface ExtractionDigest {
  complete?: boolean;
  error?: string;
  n_relevant: number;
  n_extracted: number;
  n_fulltext: number;
  n_abstract: number;
  n_unread: number;
  coverage: Record<string, number>;
  coverage_fulltext: Record<string, number>;
  by_sheet: { sheet: string; n_rows: number; n_articles: number; n_quote_found: number }[];
  top_groups: { sheet: string; value: string; n: number }[];
  crude_counts: { sheet: string; covariate: string; n_studies: number; n_cases: number; pop_risk: number }[];
  crude_counts_note?: string;
}

export interface ExtractionArticle {
  id: number;
  title: string;
  year: number | null;
  doi: string | null;
  journal: string | null;
  has_extraction: boolean;
  source: "fulltext" | "abstract" | null;
  text_truncated: boolean;
  coverage: Record<string, boolean> | null;
  n_observations: number;
  n_quote_found: number;
  attempts: number;
  n_reviewed: number;
  n_rejected: number;
  n_conflict: number;
}

export interface ExtractionArticlesResponse {
  scenario_id: string;
  total: number;
  extracted: number;
  offset: number;
  limit: number;
  returned: number;
  truncated: boolean;
  next_offset: number | null;
  articles: ExtractionArticle[];
}

export interface ExtractionObservation {
  sheet: string;
  transmission_mode: string | null;
  disease: string | null;
  group: string | null;
  covariate: string;
  value: number | null;
  descr: string | null;
  notes: string | null;
  n_cases: number | null;
  pop_risk: number | null;
  original_name: string | null;
  page_section: string | null;
  source_kind: string | null;
  quote: string | null;
  quote_verified: boolean;
  /** Where the labels sit in the codebook (added when read; the stored words are untouched). */
  l1?: string | null;
  l2?: string | null;
  label_path?: string | null;
  matched?: boolean;
  /** The review state over all reviewers, and the fields after their corrections. */
  obs_key?: string;
  review_status?: ReviewStatus;
  reviews?: { reviewer: string; status: string; edits: Record<string, unknown> | null; note: string | null }[];
  effective?: Partial<ExtractionObservation>;
}

export type ReviewStatus = "unreviewed" | "accepted" | "edited" | "rejected" | "conflict";

export interface ArticleExtraction {
  id: number;
  title: string;
  extracted_at: string | null;
  extraction: {
    source: string;
    truncated: boolean;
    /** The reproducibility record: which model and prompt made this, and when. */
    model?: string;
    prompt_sha?: string;
    extracted_at?: string;
    ref: { description?: string | null; article_type?: string | null; location?: string | null };
    observations: ExtractionObservation[];
  } | null;
  /** Decisions whose observation the extraction no longer has (a re-extraction read the paper differently). */
  stale_reviews?: { obs_key: string; reviewer: string; status: string }[];
}

export async function fetchExtractionStatus(scenarioId: string): Promise<ExtractionStatus> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/extraction/status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchExtractionDigest(scenarioId: string): Promise<ExtractionDigest> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/extraction/coverage`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

/** EVERY relevant article, page by page: the screen says "all", so it loads all. */
export async function fetchAllExtractionArticles(scenarioId: string, maxPages = 20): Promise<ExtractionArticlesResponse> {
  const base = `${scenarioBase(scenarioId)}/${scenarioId}/extraction/articles`;
  let offset: number | null = 0;
  let first: ExtractionArticlesResponse | null = null;
  const all: ExtractionArticle[] = [];
  for (let page = 0; offset !== null && page < maxPages; page++) {
    const r = await safeFetch(`${base}?limit=5000&offset=${offset}`);
    if (!r.ok) throw new Error(httpMessage(r.status));
    const chunk: ExtractionArticlesResponse = await r.json();
    first = first ?? chunk;
    all.push(...chunk.articles);
    offset = chunk.truncated ? chunk.next_offset : null;
  }
  if (!first) throw new Error(httpMessage(500));
  return { ...first, articles: all, returned: all.length, offset: 0, truncated: offset !== null, next_offset: offset };
}

export async function fetchArticleExtraction(scenarioId: string, articleId: number): Promise<ArticleExtraction> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/articles/${articleId}/extraction`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

/** Starts the background extraction (needs the admin key). `status` is started, running, no_llm... */
export async function startExtraction(scenarioId: string): Promise<{ status: string }> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/extraction/run`, {
    method: "POST", headers: authHeaders(),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export function extractionExportUrl(scenarioId: string, format: "xlsx" | "csv"): string {
  return `${scenarioBase(scenarioId)}/${scenarioId}/extraction/export?format=${format}`;
}

export interface ReviewSummary {
  scenario_id: string;
  n_observations: number;
  counts: Record<ReviewStatus, number>;
  n_reviewed: number;
  share_reviewed: number;
  reviewers: { reviewer: string; n_decisions: number }[];
  agreement: { reviewers: [string, string]; n_common: number; observed: number | null; kappa: number | null }[];
  n_stale_decisions: number;
}

export async function fetchReviewSummary(scenarioId: string): Promise<ReviewSummary> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/extraction/review/summary`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

/** Accept, edit or reject one observation as `reviewer`; status "clear" removes that reviewer's decision. */
export async function reviewObservation(
  scenarioId: string, articleId: number,
  body: { obs_key: string; reviewer: string; status: "accepted" | "edited" | "rejected" | "clear";
          edits?: Record<string, unknown>; note?: string },
): Promise<{ observation: ExtractionObservation }> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/articles/${articleId}/extraction/review`, {
    method: "POST", headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json()).detail ?? ""; } catch { /* the body is not JSON */ }
    throw new Error(detail || httpMessage(r.status));
  }
  return r.json();
}

export async function reviewBulk(
  scenarioId: string, articleId: number, reviewer: string, status: "accepted" | "rejected",
): Promise<{ n_recorded: number; n_skipped: number }> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/articles/${articleId}/extraction/review/bulk`, {
    method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ reviewer, status, verified_only: true }),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export function extractionReportUrl(scenarioId: string, format: "pdf" | "docx" | "md", lang: string): string {
  return `${scenarioBase(scenarioId)}/${scenarioId}/extraction/report?format=${format}&lang=${lang.toLowerCase().startsWith("fr") ? "fr" : "en"}`;
}

export function extractionDatasetUrl(scenarioId: string): string {
  return `${scenarioBase(scenarioId)}/${scenarioId}/extraction/dataset`;
}

// ─── The paper's text around a quote ───
export interface TextWindow {
  article_id: number;
  title: string;
  source: "fulltext" | "abstract";
  text_truncated: boolean;
  n_chars: number;
  found: boolean;
  partial: boolean;
  start: number;
  end: number;
  window_start: number;
  window_end: number;
  before: string;
  match: string;
  after: string;
}

export async function fetchTextWindow(scenarioId: string, articleId: number, quote: string, context = 700): Promise<TextWindow> {
  const q = `quote=${encodeURIComponent(quote.slice(0, 600))}&context=${context}`;
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/articles/${articleId}/text?${q}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Where the evidence comes from (country and NUTS region of each study) ───
export interface GeoRegion { code: string; name: string; n_papers: number }
export interface GeoCountry {
  iso2: string;
  name: string;
  nuts0: string | null;
  n_papers: number;
  n_rows: number;
  regions: GeoRegion[];
}
export interface GeographyResponse {
  scenario_id: string;
  n_papers: number;
  n_resolved: number;
  n_unresolved: number;
  n_several_countries: number;
  nuts_source: "builtin" | "loaded";
  countries: GeoCountry[];
  unresolved: { location: string; n: number }[];
}

export async function fetchGeography(scenarioId: string): Promise<GeographyResponse> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/extraction/geography`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchNutsStatus(): Promise<{ source: "builtin" | "loaded"; n_regions: number; note: string }> {
  const r = await safeFetch(`${API_BASE_URL}/geo/nuts/status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function importNutsCsv(csv: string): Promise<{ loaded: number; by_level: Record<string, number>; countries: number }> {
  const r = await safeFetch(`${API_BASE_URL}/geo/nuts/import`, {
    method: "POST", headers: authHeaders({ "Content-Type": "text/csv" }), body: csv,
  });
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json()).detail ?? ""; } catch { /* the body is not JSON */ }
    throw new Error(detail || httpMessage(r.status));
  }
  return r.json();
}

// ─── Pooled estimates (random-effects meta-analysis of what the studies report) ───
export interface PooledStudy {
  article_id: number;
  title: string | null;
  year: number | null;
  first_author: string;
  x: number;
  n: number;
  review_status: string;
  p?: number;
  or?: number;
  ci_low?: number;
  ci_high?: number;
  weight_pct?: number;
  x1?: number; n1?: number; x2?: number; n2?: number;
}

export interface Heterogeneity {
  Q: number; df: number; p: number; I2: number; tau2: number;
  band: "low" | "moderate" | "substantial" | "considerable";
}

export interface PooledGroup {
  sheet: string;
  group: string;
  label: string;
  disease: string | null;
  label_path: string | null;
  mapped: boolean;
  k: number;
  n_total: number;
  events_total: number;
  pooled: { p: number; ci_low: number; ci_high: number; pi_low: number | null; pi_high: number | null } | null;
  heterogeneity: Heterogeneity | null;
  reason: string | null;
  studies: PooledStudy[];
}

export interface PooledComparison {
  sheet: string;
  group: string;
  disease: string | null;
  a: string;
  b: string;
  k: number;
  pooled: { or: number; ci_low: number; ci_high: number; pi_low: number | null; pi_high: number | null };
  heterogeneity: Heterogeneity;
  studies: PooledStudy[];
}

export interface PooledResponse {
  scenario_id: string;
  filters: { reviewed_only: boolean; verified_only: boolean; split_disease: boolean; min_studies: number };
  n_rows_used: number;
  n_duplicate_rows_dropped: number;
  excluded: Record<string, number>;
  pooled: PooledGroup[];
  comparisons: PooledComparison[];
  /** What EXISTS, beside what is served. The lists are capped (120 groups, 40
   *  comparisons) and the panel used to count its "too few studies" line on the capped
   *  list: it said 120 where there were 306, and the CSV carried only the 120. */
  groups_total?: number;
  groups_returned?: number;
  groups_truncated?: boolean;
  groups_too_few_studies?: number;
  comparisons_total?: number;
  comparisons_returned?: number;
  comparisons_truncated?: boolean;
  caps?: { groups: number; comparisons: number; pairs_per_group: number };
}

export async function fetchPooled(
  scenarioId: string,
  o: { reviewedOnly: boolean; verifiedOnly: boolean; splitDisease: boolean },
): Promise<PooledResponse> {
  const q = `reviewed_only=${o.reviewedOnly}&verified_only=${o.verifiedOnly}&split_disease=${o.splitDisease}`;
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/extraction/pooled?${q}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Label codebook (hierarchy per extraction sheet, applied when the labels are read) ───
export interface CodebookNode {
  sheet: string;
  l1: string;
  l2: string | null;
  l3: string | null;
  synonyms: string[];
  label_en: string | null;
  label_fr: string | null;
}

export interface CodebookResponse {
  scenario_id: string;
  source: "default" | "custom";
  n_nodes: number;
  nodes: CodebookNode[];
}

export interface UnmappedLabel {
  sheet: string;
  group: string | null;
  covariate: string | null;
  n_rows: number;
  n_articles: number;
  l1: string | null;
}

export interface UnmappedResponse {
  scenario_id: string;
  rows_mapped: number;
  rows_unmapped: number;
  n_distinct_unmapped: number;
  unmapped: UnmappedLabel[];
}

export async function fetchCodebook(scenarioId: string): Promise<CodebookResponse> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/codebook`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchUnmappedLabels(scenarioId: string, top = 50): Promise<UnmappedResponse> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/codebook/unmapped?top=${top}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function addCodebookSynonym(
  scenarioId: string, body: { sheet: string; l1: string; l2: string | null; label: string },
): Promise<void> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/codebook/synonym`, {
    method: "POST", headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
}

export async function importCodebookCsv(scenarioId: string, csv: string): Promise<{ n_nodes: number }> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/codebook/import`, {
    method: "POST", headers: authHeaders({ "Content-Type": "text/csv" }), body: csv,
  });
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json()).detail ?? ""; } catch { /* the body is not JSON */ }
    throw new Error(detail || httpMessage(r.status));
  }
  return r.json();
}

export async function resetCodebook(scenarioId: string): Promise<void> {
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/codebook`, {
    method: "DELETE", headers: authHeaders(),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
}

export function codebookExportUrl(scenarioId: string): string {
  return `${scenarioBase(scenarioId)}/${scenarioId}/codebook/export`;
}

// ─── Evidence Brief ───────────────────────────────────────────────────────────
export interface EvidenceBriefData {
  scenario_id: string;
  generated_at: string;
  corpus_stats: {
    total: number;
    duplicates: number;
    with_pico: number;
    with_fulltext: number;
    relevant?: number;
    relevant_with_pico?: number;
    relevant_with_fulltext?: number;
    threshold?: number;
    included: number;
    excluded: number;
    pending: number;
    year_min: number | null;
    year_max: number | null;
    avg_citations: number | null;
    max_citations: number | null;
    /** Sur combien d'articles la moyenne porte : peu de sources en renvoient un. */
    citations_known?: number;
    pico_coverage_pct: number;
  };
  double_blind_stats: {
    reviewer_1_done: number;
    reviewer_2_done: number;
    both_done: number;
    agreements: number;
    conflicts: number;
  };
  top_articles: Array<{
    id: number;
    title: string;
    year: number | null;
    journal: string | null;
    authors: string | null;
    doi: string | null;
    study_design: string | null;
    citation_count: number | null;
    screening_status: string | null;
    quality_score: number | null;
    similarity_score: number | null;
    abstract_excerpt: string;
    pico_summary: {
      population: string;
      intervention: string;
      outcome: string;
      key_finding: string;
    } | null;
  }>;
  pico_table: Array<{
    id: number;
    title: string;
    year: number | null;
    journal: string | null;
    citation_count: number | null;
    study_design: string;
    screening_status: string | null;
    similarity_score: number | null;
    pico: {
      population: string;
      intervention: string;
      comparator: string;
      outcome: string;
      study_design: string;
      key_finding: string;
      limitations: string;
      evidence_level: string;
    };
  }>;
  study_design_distribution: Array<{ design: string; design_en?: string; count: number }>;
  year_distribution: Array<{ year: number; count: number }>;
  source_distribution: Array<{ source: string; count: number }>;
  evidence_level_distribution: Array<{ level: string; level_en?: string; count: number }>;
}

export async function fetchEvidenceBrief(scenarioId: string): Promise<EvidenceBriefData> {
  const base = scenarioBase(scenarioId);
  const r = await safeFetch(`${base}/${scenarioId}/evidence-brief`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Knowledge Graph (co-citations / similarité cosinus) ─────────────────────

export interface KGNode {
  id: number;
  title: string;
  year: number | null;
  journal: string | null;
  design: string;
  quality: number;
  cluster: number;
  degree: number;
}

export interface KGEdge {
  source: number;
  target: number;
  weight: number;
}

export interface KGCluster {
  id: number;
  size: number;
  /** Étiquette thématique (mots-clés des titres). */
  label?: string;
  years: number[];
  designs: string[];
  top_articles: string[];
}

export interface KnowledgeGraphData {
  scenario_id: string;
  n_nodes: number;
  n_edges: number;
  n_clusters: number;
  /** Nombre total d'articles éligibles (n_nodes peut être un sous-ensemble). */
  n_total?: number;
  min_similarity: number;
  nodes: KGNode[];
  edges: KGEdge[];
  clusters: KGCluster[];
}

// ─── Concept map (typed concepts, co-occurrence links, articles behind each) ──

export type ConceptType =
  | "pathogen" | "vector" | "host" | "population" | "exposure" | "intervention"
  | "outcome" | "method" | "place" | "design" | "setting" | "topic";

export interface ConceptNode {
  id: number;
  type: ConceptType;
  /** Canonical labels in both interface languages; a place carries its ISO2 code. */
  label: { en: string; fr: string };
  /** Articles citing the concept (all of them), and those of the corpus's latest year. */
  count: number;
  new_count: number;
  /** Up to 40 article ids, most relevant first (keys of `articles`). */
  articles: number[];
  /** How many `articles` the payload actually carries. Below `count` on a frequent
   *  concept: the panel says so rather than showing 40 under a header announcing 1200. */
  articles_listed?: number;
}

export interface ConceptEdge {
  source: number;
  target: number;
  /** Number of articles citing both concepts. */
  weight: number;
  /** Up to 20 of them; `articles_listed` says how many the payload carries. */
  articles: number[];
  articles_listed?: number;
}

export interface ConceptArticle {
  t: string;
  y: number | null;
  q: number;
  doi: string | null;
  pmid: string | null;
}

export interface ConceptGraphData {
  kind: "concepts";
  version: number;
  scenario_id: string;
  n_articles: number;
  n_total: number;
  n_with_concepts: number;
  n_missing_concepts: number;
  source: "llm" | "structured";
  /** True while the API normalises the missing concepts with the LLM; re-fetch later. */
  enriching?: boolean;
  latest_year: number | null;
  types: { type: ConceptType; count: number }[];
  nodes: ConceptNode[];
  edges: ConceptEdge[];
  triples: { nodes: number[]; count: number }[];
  gaps: { nodes: number[]; expected: number }[];
  articles: Record<string, ConceptArticle>;
}

export async function fetchConceptGraph(scenarioId: string, refresh = false): Promise<ConceptGraphData> {
  const base = scenarioBase(scenarioId);
  const r = await safeFetch(`${base}/${scenarioId}/concept-graph${refresh ? "?refresh=true" : ""}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchKnowledgeGraph(
  scenarioId: string,
  maxNodes = 400,
  minSimilarity = 0.35,
): Promise<KnowledgeGraphData> {
  const base = scenarioBase(scenarioId);
  const r = await safeFetch(
    `${base}/${scenarioId}/knowledge-graph?max_nodes=${maxNodes}&min_similarity=${minSimilarity}`,
  );
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Streaming RAG SSE ────────────────────────────────────────────────────────

/** Corpus counts behind an AI answer. `papers_used` is the relevant subset SEARCHED;
 *  `papers_retrieved` is how many of them were pulled in to compose the answer. Neither
 *  is the number of articles the answer QUOTES, which nothing measures: the field used
 *  to be called papers_quoted and the interface, the stored record and the exported
 *  document all presented the retrieval depth as the citation count.
 *  `digest_complete` says whether the answer's figures were backed by the whole-corpus
 *  digest (SQL over every relevant article) rather than by the excerpts alone. */
export interface RagMeta {
  papers_used: number;
  papers_with_fulltext: number;
  papers_retrieved?: number;
  digest_complete?: boolean;
  threshold: number;
}

export interface RagStreamCallbacks {
  onSources: (sources: ScenarioRagSource[]) => void;
  onToken: (token: string) => void;
  onMeta?: (meta: RagMeta) => void;
  onDone: () => void;
  onError: (err: string) => void;
}

export interface KappaStats {
  scenario_id: string;
  n_evaluated: number;
  kappa: number | null;
  po_observed: number;
  pe_expected: number;
  interpretation: string;
  conflicts: number;
  agreements: Record<string, number>;
  matrix: Record<string, Record<string, number>>;
}

export interface DoubleBlindDecision {
  article_id: number;
  /** Kept for backward compatibility and ignored by the API: the role is derived from
   *  the reviewer code, server side. A client that picks its own role ends up giving
   *  the same one to two people. */
  reviewer?: 1 | 2;
  status: "included" | "excluded" | "pending";
  reason?: string;
  reviewer_code: string;
}

export async function submitDoubleBlindDecision(
  scenarioId: string,
  payload: DoubleBlindDecision,
): Promise<{ id: number; agreement: boolean | null; final_status: string | null }> {
  const base = scenarioBase(scenarioId);
  const r = await safeFetch(
    `${base}/${scenarioId}/double-blind/decision`,
    {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    },
  );
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export interface DoubleBlindQueue {
  scenario_id: string;
  reviewer: 1 | 2;
  reviewer_code: string;
  remaining: number;
  returned: number;
  articles: Array<{
    id: number;
    title: string;
    abstract?: string | null;
    year?: number | null;
    journal?: string | null;
    doi?: string | null;
    similarity_score?: number | null;
    rerank_score?: number | null;
    reviewer_1_status?: string | null;
    reviewer_2_status?: string | null;
  }>;
}

/** The articles THIS reviewer has not voted on yet. Until now there was no way to cast
 *  a double-blind vote at all: the panel's only buttons were the arbitration ones, and
 *  the conflicts list can only fill once both reviewers have voted. */
export async function fetchDoubleBlindQueue(
  scenarioId: string,
  reviewerCode: string,
  limit = 25,
): Promise<DoubleBlindQueue> {
  const base = scenarioBase(scenarioId);
  const qs = new URLSearchParams({ reviewer_code: reviewerCode, limit: String(limit) });
  const r = await safeFetch(`${base}/${scenarioId}/double-blind/queue?${qs}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

/** Register a reviewer code on a scenario and get back THE role the server assigns:
 *  first code registered is reviewer 1, second is reviewer 2, a third is refused (409).
 *  The role used to be decided by the browser from a per-tab sessionStorage, so two
 *  reviewers on two machines both became reviewer 1 and the second overwrote the first. */
export async function registerDoubleBlindReviewer(
  scenarioId: string,
  reviewerCode: string,
): Promise<{ reviewer: 1 | 2; reviewer_code: string; registered: Record<string, string> }> {
  const base = scenarioBase(scenarioId);
  const r = await safeFetch(
    `${base}/${scenarioId}/double-blind/register?reviewer_code=${encodeURIComponent(reviewerCode)}`,
    { method: "POST", headers: authHeaders() },
  );
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.detail || httpMessage(r.status));
  }
  return r.json();
}

/** Arbitrate a disagreement. This is NOT the decision endpoint: pressing the
 *  arbitration buttons used to rewrite a reviewer's own vote, which manufactured the
 *  agreement the kappa then counted. */
export async function resolveDoubleBlindConflict(
  scenarioId: string,
  articleId: number,
  finalStatus: "included" | "excluded",
  arbitratorNotes?: string,
): Promise<{ id: number; final_status: string; resolved: boolean }> {
  const base = scenarioBase(scenarioId);
  const qs = new URLSearchParams({ article_id: String(articleId), final_status: finalStatus });
  if (arbitratorNotes) qs.set("arbitrator_notes", arbitratorNotes);
  const r = await safeFetch(`${base}/${scenarioId}/double-blind/resolve?${qs}`,
                            { method: "POST", headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchKappaStats(scenarioId: string): Promise<KappaStats> {
  const base = scenarioBase(scenarioId);
  const r = await safeFetch(
    `${base}/${scenarioId}/double-blind/kappa`,
  );
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchDoubleBlindConflicts(
  scenarioId: string,
): Promise<any[]> {
  const base = scenarioBase(scenarioId);
  const r = await safeFetch(
    `${base}/${scenarioId}/double-blind/conflicts`,
  );
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Evidence Brief PDF côté serveur ─────────────────────────────────────────

export function getEvidenceBriefPdfUrl(scenarioId: string): string {
  const base = scenarioBase(scenarioId);
  return `${base}/${scenarioId}/evidence-brief/pdf`;
}

// ─────────────────────────────────────────────────────────────────────────────
// USER SCENARIOS : Helpers de routage et CRUD
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retourne true si l'ID est un scénario utilisateur (préfixe "usr-").
 * Utilisé pour router vers /user-scenarios/... au lieu de /gesica/scenarios/...
 */
export function isUserScenario(scenarioId: string): boolean {
  return scenarioId.startsWith('usr-');
}

/**
 * Retourne le préfixe d'URL correct selon le type de scénario.
 * - Scénario GESICA : /gesica/scenarios
 * - Scénario utilisateur : /user-scenarios
 */
export function scenarioBase(scenarioId: string): string {
  return isUserScenario(scenarioId)
    ? `${API_BASE_URL}/user-scenarios`
    : `${API_BASE_URL}/gesica/scenarios`;
}

// ─── Types user-scenarios ─────────────────────────────────────────────────────

export interface UserScenario extends GesicaScenario {
  query: string;
  // Multi-facet search: full expression "(A) AND (B)" for display, plus the saved
  // facets/combinator so a replay can restore them. Absent on single-query searches.
  combined_query?: string;
  sub_queries?: SubQuery[] | null;
  combinator?: "union" | "intersection" | null;
  mode: string;
  /** La nature de la question ; absente d'une ancienne réponse, elle vaut tout. */
  kind?: ScenarioKind;
  filters: Record<string, any>;
  result_count: number;
  resultCount: number;           // alias camelCase de result_count
  pinned: boolean;
  created_at: string | null;
  /** L'adresse du poste qui a créé la recherche. Nulle pour tout ce qui n'a pas de
   *  requête derrière (amorçage, scripts) et pour les lignes antérieures. */
  created_ip?: string | null;
  updated_at: string | null;
  is_user_scenario: true;
  populate_status?: string;
  pipeline_status?: string;
  pipeline_step?: string | null;
  pipeline_progress?: number;
}

/** One facet of a multi-query search: a boolean or natural-language sub-query. */
export interface SubQuery {
  // "auto" = let the backend detect boolean vs natural from syntax (default);
  // "boolean"/"natural" = explicit user override.
  kind: "boolean" | "natural" | "auto";
  text: string;
  // Per-facet combine operator vs the running result, folded LEFT-TO-RIGHT:
  // "or" = union, "and" = intersection. Omitted ⇒ the backend uses the global
  // `combinator` for this facet. The main query (facet 0) is the base, no op.
  op?: "and" | "or";
}

export interface FacetPreview {
  kind: "boolean" | "natural";   // resolved kind (auto → detected)
  text: string;
  boolean: string;               // the boolean actually matched (natural → translated)
  count: number;                 // lexical matches in the indexed library
  op?: "and" | "or" | null;      // per-facet operator echoed back (null for the main facet / unset)
}

export interface FacetPreviewResponse {
  facets: FacetPreview[];
  union: number;
  intersection: number;
  combined: number;
  combinator: "union" | "intersection";
}

export interface UserScenarioCreatePayload {
  name: string;
  query: string;
  mode: string;
  filters: Record<string, any>;
  result_count?: number;
  pinned?: boolean;
  search_strategy?: SearchStrategy | null;
  // Multi-sub-query search: ≥2 facets combined by `combinator` (union = OR,
  // intersection = AND). Omitted for a normal single-query search.
  sub_queries?: SubQuery[];
  combinator?: "union" | "intersection";
}

export interface PipelineStepStatus {
  status: 'pending' | 'running' | 'done' | 'error' | 'skipped';
  ingested?: number;
  api_results_raw?: number;
  extracted?: number;
  extracted_this_run?: number;
  fetched?: number;
  n_clusters?: number;
  n_docs?: number;
  found?: number;
  errors?: number;
  reason?: string;
  error?: string;
  // embed step (docs and chunks)
  docs_done?: number;
  docs_total?: number;
  docs_embedded?: number;
  chunks_done?: number;
  chunks_total?: number;
  chunks_embedded?: number;
  pct?: number;
  // pico/metadata coverage
  total_with_pico?: number;
  total_with_metadata?: number;
  total_articles?: number;
  // rerank step
  updated?: number;
  // clustering step
  method?: string;
}

export interface UserScenarioPipelineStatus {
  scenario_id: string;
  overall_status: 'not_started' | 'starting' | 'running' | 'done' | 'error';
  current_step?: string;
  message?: string;
  error?: string;
  steps: {
    ingest?: PipelineStepStatus;
    fulltext?: PipelineStepStatus;
    embed?: PipelineStepStatus;
    rerank?: PipelineStepStatus;
    pico?: PipelineStepStatus;
    metadata?: PipelineStepStatus;
    clustering?: PipelineStepStatus;
  };
}

export interface EmbeddingChunkType {
  type: string;
  total: number;
  embedded: number;
  pct: number;
}

export interface EmbeddingStatus {
  scenario_id: string;
  status: 'none' | 'partial' | 'complete';
  status_label: string;
  corpus_total?: number;
  chunkless?: number;
  abstract_only: {
    total_docs: number;
    embedded_docs: number;
    pending_docs: number;
  };
  title_abstract_chunks: {
    total_docs: number;
    embedded_docs: number;
    pending_docs: number;
  };
  fulltext: {
    total_docs: number;
    docs_fully_embedded: number;
    docs_pending: number;
    total_chunks: number;
    embedded_chunks: number;
    pending_chunks: number;
  };
  total_pending_chunks: number;
  /** Le jeu de compteurs commun, d'un seul instantané (cf. CorpusCounts). */
  counts?: CorpusCounts;
  // Pertinence (ranking) - scores réellement présents sur le corpus (≠ indexation RAG).
  ranking?: {
    total: number;
    scored: number;
    reranked: number;
    complete: boolean;
  };
  score_availability: {
    semantic: boolean;            // vert seulement quand TOUT le corpus est scoré
    cohere: boolean;              // vert seulement quand le rerank a réellement tourné
    cohere_configured?: boolean;  // clé présente (distingue "pas de clé" de "pas encore")
  };
}

// ─── CRUD user-scenarios ──────────────────────────────────────────────────────

function _mapUserScenario(u: any): UserScenario {
  // article_count = articles réellement en DB après ingestion + nettoyage
  // result_count  = snapshot du nombre de résultats au moment de la recherche
  // On ne fait PLUS de fallback result_count → articleCount pour éviter
  // la confusion 129 (recherche) → 75 (corpus réel après rerank)
  const articleCount = u.article_count ?? u.articleCount ?? 0;
  return {
    ...u,
    articleCount,
    resultCount: u.result_count ?? u.resultCount ?? 0,
    livingEvidenceNote: u.living_evidence_note ?? u.livingEvidenceNote ?? '',
    recommendedActions: u.recommended_actions ?? u.recommendedActions ?? [],
    relevantArticles: u.relevant_articles ?? u.relevantArticles ?? [],
    hidden: u.hidden ?? false,
    cluster: u.cluster ?? 'user',
    title: u.title ?? u.name ?? '',
    description: u.description ?? `Recherche : ${u.query ?? ''}`,
    populate_status: u.populate_status ?? 'idle',
    pipeline_status: u.pipeline_status ?? 'idle',
    pipeline_step: u.pipeline_step ?? null,
    pipeline_progress: u.pipeline_progress ?? 0,
    kind: (u.kind === 'review' ? 'review' : 'predictive') as ScenarioKind,
  };
}

export async function fetchUserScenarios(): Promise<UserScenario[]> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  const data: any[] = await r.json();
  return data.map(_mapUserScenario);
}

export async function createUserScenario(
  payload: UserScenarioCreatePayload,
): Promise<UserScenario> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios?lang=${currentLang()}`, {
    method: 'POST',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return _mapUserScenario(await r.json());
}

export async function deleteUserScenario(scenarioId: string): Promise<{ deleted: boolean; id: string }> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}`, { method: 'DELETE', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function patchUserScenario(
  scenarioId: string,
  patch: { name?: string; pinned?: boolean; mode?: string; kind?: ScenarioKind; filters?: Record<string, any>; folder_id?: string | null },
): Promise<UserScenario> {
  // `lang`: pinning starts the full pipeline; everything it caches is produced in the
  // interface language, so no tab has to generate at its first opening.
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}?lang=${currentLang()}`, {
    method: 'PATCH',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(patch),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return _mapUserScenario(await r.json());
}

export async function startUserScenarioPipeline(
  scenarioId: string,
  maxResults = 100000,
): Promise<{ scenario_id: string; status: string; message: string; steps: string[] }> {
  const r = await safeFetch(
    `${API_BASE_URL}/user-scenarios/${scenarioId}/pipeline?max_results=${maxResults}&lang=${currentLang()}`,
    { method: 'POST', headers: authHeaders() },
  );
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchUserScenarioPipelineStatus(
  scenarioId: string,
): Promise<UserScenarioPipelineStatus> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}/pipeline/status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

/** LE jeu de compteurs du corpus, compté par une seule instruction SQL donc un seul
 *  instantané (api/scenario_store.py : scenario_counts). Tout panneau qui affiche un
 *  nombre d'articles lit cet objet ; aucun ne compte pour son compte, sinon deux
 *  nombres du même écran se remettent à diverger - l'en-tête annonçait 433 articles
 *  pendant que le titre du corpus en annonçait 449 et le voyant « 441 scorés sur 433 ».
 *  Renvoyé à l'identique par /counts, /detail, /corpus et /embedding-status. */
export interface CorpusCounts {
  threshold: number;
  total: number;
  above_threshold: number;
  /** Scorés ET sous le seuil. Les non scorés sont à part : les trois font le total. */
  below_threshold: number;
  unscored: number;
  scored: number;
  reranked: number;
  /** Ce que les extractions lisent vraiment (porte commune), pas un partage par le score. */
  relevant: number;
  included: number;
  excluded: number;
  pending: number;
  with_fulltext: number;
  chunkless: number;
  newly_fetched: number;
  from_local: number;
  years_covered: number;
  journals_count: number;
  year_min: number | null;
  year_max: number | null;
}

/** Les nombres d'articles affichés pour un scénario (liste, en-tête, PRISMA, étape
 *  sémantique) comparés entre eux, et si un pipeline/populate tourne encore. */
export interface ScenarioCounts {
  scenario_id: string;
  in_progress: boolean;
  pipeline_status?: string | null;
  populate_status?: string | null;
  current_step?: string | null;
  threshold: number;
  /** Copie stockée, lue par la liste des scénarios (mise à jour par étapes pendant une recherche). */
  article_count: number;
  /** Référence : liens en base hors doublons (onglet Corpus). */
  corpus_links: number;
  /** « Passés au screening » du PRISMA, figé à la fin de la dernière recherche ; null avant. */
  prisma_screened: number | null;
  above_threshold: number;
  below_threshold: number;
  embedded: number;
  /** Le jeu complet, d'un seul instantané : ce que TOUS les panneaux affichent. */
  counts: CorpusCounts;
  consistent: boolean;
  mismatches: Array<{ field: string; value: number; expected: number }>;
  checked_at: string;
}

export async function fetchScenarioCounts(scenarioId: string): Promise<ScenarioCounts> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}/counts`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

/** Recherche ou pipeline en cours, tous scénarios confondus (indicateur global de
 *  l'en-tête) : une recherche continue côté serveur quand on change de page. */
export interface ActivityItem {
  scenario_id: string;
  name: string;
  query: string;
  pinned: boolean;
  kind: "search" | "pipeline";
  step?: string | null;
  article_count: number;
}

export async function fetchActivity(): Promise<{ running: ActivityItem[]; count: number; checked_at: string }> {
  const r = await safeFetch(`${API_BASE_URL}/activity`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

/** Construit le corpus (= requête booléenne sur base locale ∪ live) en arrière-plan. */
export async function populateUserScenario(
  scenarioId: string,
  opts?: { includeLive?: boolean; maxResults?: number },
): Promise<{ scenario_id: string; status: string; message?: string }> {
  const params = new URLSearchParams();
  params.set('max_results', String(opts?.maxResults ?? 2000));
  params.set('include_live', String(opts?.includeLive ?? true));
  params.set('lang', currentLang());          // cluster summaries precomputed in this language
  const r = await safeFetch(
    `${API_BASE_URL}/user-scenarios/${scenarioId}/populate?${params}`,
    { method: 'POST', headers: authHeaders() },
  );
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchUserScenarioPopulateStatus(
  scenarioId: string,
): Promise<{
  scenario_id: string;
  /** 'not_started' | 'running' | 'done' | 'unranked' | 'error'. `unranked` = the corpus
   *  IS built but the semantic scoring produced no score: the articles are real and
   *  readable, their ORDER and the similarity threshold are not. Not an error. */
  status: string;
  ingested?: number;
  // Phase RÉELLE du backend (et non un minuteur côté client).
  phase?: 'local' | 'federation' | 'scoring' | 'done';
  // Statut du cross-encoder Cohere qui réordonne en arrière-plan après l'affichage.
  rerank_status?: 'idle' | 'running' | 'done' | 'skipped';
  /** Present only when `status === 'unranked'`: why no score was produced. */
  scoring?: { ok: boolean; reason_code: string; reason: string };
  sources?: Record<string, number>;
  /** Sources replayed from the cache of the last identical search (no network call). */
  cached_sources?: string[];
}> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}/populate/status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Alertes email ────────────────────────────────────────────────────────────

export async function subscribeAlerts(
  email: string,
  scenarioId: string,
  frequency: "daily" | "weekly" | "immediate" = "weekly",
): Promise<{ status: string; message: string; owner_set?: boolean }> {
  const r = await safeFetch(`${API_BASE_URL}/alerts/subscribe`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ email, scenario_id: scenarioId, frequency }),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Living Review ────────────────────────────────────────────────────────────

export async function triggerLivingReview(
  scenarioId?: string,
  dryRun = true,
): Promise<{ status: string; message: string; scenarios: any[] }> {
  const params = new URLSearchParams({ dry_run: String(dryRun), lang: currentLang() });
  if (scenarioId) params.set("scenario_id", scenarioId);
  const r = await safeFetch(`${API_BASE_URL}/gesica/living-review/trigger?${params}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Enrichissement LLM Batch ─────────────────────────────────────────────────

export interface EnrichmentBatchResult {
  extracted: number;
  skipped: number;
  errors: number;
  message: string;
}

/** La portée d'un lot d'enrichissement sur un scénario. */
export type EnrichmentScope = 'all' | 'relevant';

export interface EnrichmentJobStatus {
  count: number;
  pct: number;
  /** Ce qu'une exécution traiterait réellement, donc ce qu'elle coûterait. */
  todo: number;
}

export interface EnrichmentScopeStatus {
  total: number;
  pico: EnrichmentJobStatus;
  metadata: EnrichmentJobStatus;
  fulltext: EnrichmentJobStatus;
}

export interface EnrichmentStatus {
  scenario_id: string | null;
  scope: EnrichmentScope;
  total: number;
  pico: EnrichmentJobStatus;
  metadata: EnrichmentJobStatus;
  fulltext: EnrichmentJobStatus;
  /** Les deux portées d'un coup, pour annoncer chaque choix avant de le lancer.
   *  Nul hors scénario : à l'échelle du corpus la question ne se pose pas. */
  by_scope: Record<EnrichmentScope, EnrichmentScopeStatus> | null;
}

export async function extractPicoBatchGlobal(
  scenarioId?: string,
  limit = 100000,
  scope?: EnrichmentScope,
): Promise<EnrichmentBatchResult> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (scenarioId) params.set('scenario_id', scenarioId);
  if (scope) params.set('scope', scope);
  const r = await safeFetch(`${API_BASE_URL}/pico/extract?${params}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function extractMetadataBatch(
  scenarioId?: string,
  limit = 100000,
  scope?: EnrichmentScope,
): Promise<EnrichmentBatchResult> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (scenarioId) params.set('scenario_id', scenarioId);
  if (scope) params.set('scope', scope);
  const r = await safeFetch(`${API_BASE_URL}/metadata/extract?${params}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchFulltextBatch(
  scenarioId?: string,
  limit = 100000,
  scope?: EnrichmentScope,
): Promise<EnrichmentBatchResult & { fetched: number; not_available: number }> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (scenarioId) params.set('scenario_id', scenarioId);
  if (scope) params.set('scope', scope);
  const r = await safeFetch(`${API_BASE_URL}/fulltext/fetch?${params}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchEnrichmentStatus(
  scenarioId?: string,
  scope?: EnrichmentScope,
): Promise<EnrichmentStatus> {
  const params = new URLSearchParams();
  if (scenarioId) params.set('scenario_id', scenarioId);
  if (scope) params.set('scope', scope);
  const r = await safeFetch(`${API_BASE_URL}/enrichment/status?${params}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Dossiers de scénarios ────────────────────────────────────────────────────

export interface ScenarioFolder {
  id: string;
  name: string;
  color: string;
  sort_order: number;
  scenario_count: number;
  created_at: string | null;
}

export async function fetchFolders(): Promise<ScenarioFolder[]> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenario-folders`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function createFolder(
  name: string,
  color = '#6366f1',
  sort_order = 0,
): Promise<ScenarioFolder> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenario-folders`, {
    method: 'POST',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ name, color, sort_order }),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function updateFolder(
  folderId: string,
  name: string,
  color: string,
  sort_order: number,
): Promise<ScenarioFolder> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenario-folders/${folderId}`, {
    method: 'PATCH',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ name, color, sort_order }),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function deleteFolder(folderId: string): Promise<{ deleted: boolean; id: string }> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenario-folders/${folderId}`, { method: 'DELETE', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function assignScenarioToFolder(
  scenarioId: string,
  folderId: string | null,
): Promise<UserScenario> {
  return patchUserScenario(scenarioId, { folder_id: folderId ?? '' });
}

// ─── Scoring sémantique + Paramètres par scénario ────────────────────────────

export type CurveScore = 'similarity' | 'rerank';

export interface ScenarioSettings {
  scenario_id: string;
  similarity_threshold: number;
  /** The second gate, on the rerank score. 0 means no filtering, which is the default
   *  and exactly the behaviour that existed before it was added. */
  rerank_threshold: number;
  brief_generated_at: string | null;
  variables_validated: boolean;
  variables_generated_at: string | null;
  // Presence of the cached artifacts (the artifacts themselves come from their own
  // endpoints; the settings call no longer ships the clustering/graph/brief blobs).
  cached?: Record<string, boolean>;
}

export async function getScenarioSettings(scenarioId: string): Promise<ScenarioSettings> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/settings`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// The threshold picked by NUMBER of articles rather than by feel. `kept` is what the
// threshold really leaves (ties make a round target rarely reachable, hence `exact`),
// and the parameter columns say what that choice costs in articles that report an
// epidemiological measurement.
export interface ThresholdCurvePoint {
  threshold: number;
  kept: number;
  kept_scored: number;
  /** Where the slider stands today. A property of the point, not a row of its own: a
   *  ladder target can land exactly on it, and then there is only one point. */
  is_current: boolean;
  requested: number | null;
  exact: boolean | null;
  with_parameter_kept: number;
  with_parameter_cut: number;
}

export interface ThresholdCurve {
  scenario_id: string;
  /** Which score the curve is drawn over. The two differ on one point, below. */
  score: CurveScore;
  current_threshold: number;
  candidates: number;
  included: number;
  unscored: number;
  /** The asymmetry, stated rather than implied: an article with no similarity score
   *  counts as 0 and leaves, while an article the rerank has not judged yet stays. */
  unscored_are_kept: boolean;
  /** Everything that passes whatever the threshold: hand-included articles, plus the
   *  not-yet-reranked ones when this is the rerank curve. */
  always_kept: number;
  corpus: number;
  /** What the threshold can actually produce: below `min` the hand-included articles
   *  pass whatever happens, above `max` there is nothing left. Null on an empty corpus. */
  reachable: { min: number; max: number } | null;
  with_parameter_total: number;
  scoring_in_progress: boolean;
  /** A cluster or concept narrowing in force. It only ever judged the articles relevant
   *  at `judged_above_threshold`, so a threshold below that brings back articles it never
   *  saw. Null when nothing is narrowed. */
  scope: { excluded_by_scope: number; judged_above_threshold: number | null } | null;
  curve: ThresholdCurvePoint[];
  suggestion?: ThresholdCurvePoint | null;
}

/**
 * The gap matrix: how many relevant articles pair each row concept with each column
 * concept, counted in SQL over the WHOLE relevant subset with no LLM in the path. An
 * empty cell inside the shown grid is therefore a fact about this corpus.
 *
 * `coverage` is not decoration: articles whose concepts were never extracted are
 * invisible here, and a gap figure that hides its own denominator is worthless.
 */
export interface EvidenceGaps {
  row_type: string;
  col_type: string;
  threshold: number;
  coverage: { relevant: number; with_concepts: number };
  available_types: Array<{ value: string; n: number }>;
  rows: Array<{ label: string; n: number }>;
  cols: Array<{ label: string; n: number }>;
  rows_total: number;
  cols_total: number;
  cells: Array<{ row: string; col: string; n: number }>;
  gaps: Array<{ row: string; col: string }>;
  pairs_observed: number;
  cells_shown: number;
  complete: boolean;
  error?: string;
  note?: string;
}

/**
 * The citable report as a download URL rather than a fetch: the endpoint already returns
 * a markdown attachment with its own filename, and the browser does that better than we
 * would. Built here because every other URL in the app is, and `API_BASE_URL` stays
 * private to this module.
 */
/**
 * Which study design maps to which level of evidence, and why. Served from the same table
 * the distribution charts and the claim grading are computed from, so the legend cannot
 * drift from the behaviour it explains.
 */
export interface StudyDesignVocabulary {
  /** Les niveaux : `value` est celle du serveur, `label` est affichable. */
  levels: Array<{ value: string; label: string }>;
  note: string;
  sources: string[];
  /** Groupé PAR NIVEAU : seize lignes répétaient six explications. */
  groups: Array<{
    level: string | null;          // null pour la synthèse, qui hérite
    label: string;
    why: string;
    designs: Array<{ key: string; label: string; mesh: string }>;
  }>;
}

export async function fetchStudyDesignVocabulary(lang: string): Promise<StudyDesignVocabulary> {
  const r = await safeFetch(`${API_BASE_URL}/study-design-vocabulary?lang=${encodeURIComponent(lang)}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export function evidenceReportUrl(scenarioId: string): string {
  return `${API_BASE_URL}/user-scenarios/${scenarioId}/evidence-report?download=true`;
}

/** Downloads the citable report, and REFUSES instead of saving a refusal.
 *
 *  It was a plain `<a download>`: when no brief had been generated the endpoint used to
 *  answer 200 with a JSON error body, so the browser saved a 166-byte file bearing the
 *  report's name. The endpoint now answers 409; this reads it and throws the detail, so
 *  the page can say why. */
export async function downloadEvidenceReport(scenarioId: string): Promise<void> {
  const r = await safeFetch(evidenceReportUrl(scenarioId));
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.detail || httpMessage(r.status));
  }
  const blob = await r.blob();
  const name = (r.headers.get("Content-Disposition") || "")
    .match(/filename="?([^"]+)"?/)?.[1] || `report-${scenarioId}.md`;
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function fetchEvidenceGaps(
  scenarioId: string,
  rows?: string,
  cols?: string,
): Promise<EvidenceGaps> {
  const qs = new URLSearchParams();
  if (rows) qs.set('rows', rows);
  if (cols) qs.set('cols', cols);
  const suffix = qs.toString() ? `?${qs.toString()}` : '';
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}/evidence-gaps${suffix}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchThresholdCurve(
  scenarioId: string,
  target?: number,
  score: CurveScore = 'similarity',
): Promise<ThresholdCurve> {
  const p = new URLSearchParams();
  if (target && target > 0) p.set('target', String(Math.round(target)));
  if (score !== 'similarity') p.set('score', score);
  const qs = p.toString() ? `?${p.toString()}` : '';
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/threshold-curve${qs}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function patchScenarioSettings(
  scenarioId: string,
  payload: { similarity_threshold?: number; rerank_threshold?: number; variables_json?: Record<string, unknown> | null; variables_validated?: boolean },
): Promise<{ status: string; scenario_id: string; updated: string[] }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/settings`, {
    method: 'PATCH',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchUserScenarioEmbeddingStatus(
  scenarioId: string,
): Promise<EmbeddingStatus> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}/embedding-status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function triggerRerank(
  scenarioId: string,
  query?: string,
  missingOnly = false,
): Promise<{ status: string; scenario_id: string; query?: string }> {
  const p = new URLSearchParams();
  if (query) p.set('query', query);
  if (missingOnly) p.set('missing_only', 'true');
  const params = p.toString() ? `?${p.toString()}` : '';
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/rerank${params}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function getRerankStatus(
  scenarioId: string,
): Promise<{
  status: string; updated?: number; error?: string;
  /** Read from the database, not from the in-memory job: "idle" used to mean "I have
   *  forgotten", since every deploy wipes the job. These say what is true now. */
  scorable?: number; missing?: number; unscorable?: number;
  reranked?: number; rerank_batches_failed?: number; rerank_skipped_no_key?: boolean;
}> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/rerank/status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Evidence Brief LLM ──────────────────────────────────────────────────────

export interface BriefReference {
  n: number;
  id: number;
  title?: string | null;
  authors?: string | null;
  year?: number | null;
  journal?: string | null;
  doi?: string | null;
  url?: string | null;
}

export interface LlmEvidenceBrief {
  /** Les citations du texte, résolues en références numérotées et cliquables. */
  references?: BriefReference[];
  /** Les identifiants cités que le corpus ne résout pas : laissés visibles. */
  unresolved_citations?: number[];
  executive_summary?: string;
  clinical_context?: string;
  key_findings?: string[];
  recommended_actions?: string[];
  evidence_synthesis?: string;
  population_summary?: string;
  intervention_summary?: string;
  outcome_summary?: string;
  methodological_quality?: string;
  limitations?: string[];
  research_gaps?: string[];
  clinical_implications?: string;
  implementation_recommendations?: string[];
  evidence_level?: string;
  grade_recommendation?: string;
  future_research?: string;
  key_references?: Array<{ title: string; year: number | null; journal: string; key_contribution: string }>;
  /**
   * One row per claim, with the certainty it is allowed to assert. `strength` is NOT
   * written by the model: it is computed from the study designs of the cited articles and
   * capped by what the corpus as a whole supports, so `basis` carries the inputs and a
   * reader can disagree with the label. `unverified_ids` lists citations that are not in
   * this scenario's corpus, reported rather than hidden.
   */
  claims?: Array<{
    claim: string;
    reasoning?: string;
    strength: string;
    article_ids: number[];
    articles: Array<{ id: number; title: string; year: number | null; study_design: string | null }>;
    basis: {
      n_articles: number;
      designs: Record<string, number>;
      from_designs: string;
      downgraded_single_study: boolean;
      capped_by_corpus: boolean;
      note?: string;
    };
    unverified_ids?: Array<number | string | null>;
  }>;
  _meta?: {
    scenario_id: string;
    scenario_name: string;
    generated_at: string;
    articles_used: number;
    articles_above_threshold: number;
    threshold: number;
    human_validated: number;
    year_range: string;
    study_designs: Record<string, number>;
    auto_generated: boolean;
    model: string;
    reasoning_effort?: string;
    /** The ceiling the claim grading applied, so "Faible" can be told apart from capped. */
    grade_ceiling?: string;
  };
  _cached?: boolean;
  _generated_at?: string | null;
  status?: string;
  message?: string;
  error?: string;
}

export async function getLlmEvidenceBrief(scenarioId: string): Promise<LlmEvidenceBrief> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/evidence-brief/llm?lang=${currentLang()}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function generateEvidenceBrief(
  scenarioId: string,
  force = false,
): Promise<{ status: string; scenario_id: string }> {
  const r = await safeFetch(
    `${API_BASE_URL}/scenarios/${scenarioId}/evidence-brief/generate?force=${force}&lang=${currentLang()}`,
    { method: 'POST', headers: authHeaders() },
  );
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function getBriefGenerationStatus(
  scenarioId: string,
): Promise<{ status: string; generated_at?: string; error?: string }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/evidence-brief/generate/status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Variables & Modele auto-rempli ──────────────────────────────────────────

export interface ScenarioVariables {
  primary_outcome?: {
    name: string;
    definition: string;
    measurement: string;
    timeframe: string;
    unit?: string;
  };
  secondary_outcomes?: Array<{ name: string; definition: string }>;
  predictor_variables?: Array<{
    name: string;
    type: string;
    definition: string;
    data_source: string;
    importance: 'high' | 'medium' | 'low';
    evidence_level: string;
    machine_name?: string;
    source?: 'user' | 'public_api' | 'seir';
    // Rôle vis-à-vis du sous-modèle SEIR (annoté côté serveur, scénarios épidémiques) :
    // 'derived' = sortie du modèle remplie automatiquement ; 'parameter' = input de
    // simulation (R0/CFR…), pas une feature du prédicteur.
    _seir_role?: 'derived' | 'parameter';
    _seir_column?: string;
  }>;
  recommended_algorithm?: {
    primary: string;
    alternatives: string[];
    rationale: string;
    validation_method: string;
  };
  required_databases?: string[];
  sample_size_recommendation?: string;
  update_frequency?: string;
  alert_thresholds?: {
    green: { label?: string; range?: string; rationale?: string; description?: string; provenance?: number[] };
    orange: { label?: string; range?: string; rationale?: string; description?: string; provenance?: number[] };
    red: { label?: string; range?: string; rationale?: string; description?: string; provenance?: number[] };
  };
  implementation_notes?: string;
  validation_status?: string;
  _meta?: {
    scenario_id: string;
    generated_at: string;
    pico_articles_used: number;
    relevant_total?: number;
    corpus_total?: number;
    auto_generated: boolean;
    validation_status: string;
  };
  _validated?: boolean;
  _generated_at?: string | null;
  status?: string;
  message?: string;
  error?: string;
}

/** Un paramètre épidémiologique mis en commun sur le corpus pertinent : la valeur
 *  pondérée par la qualité, son intervalle, le nombre d'études. Produit de REVUE
 *  autant que de prévision, d'où un accès qui ne passe pas par le modèle. */
export interface PooledEpidemicParameter {
  value: number | null;
  ci_low: number | null;
  ci_high: number | null;
  n_studies: number | null;
  unit: string | null;
}

export interface EpidemicParameterCandidates {
  scenario_id: string;
  /** Compté sur TOUT le corpus pertinent, jamais sur un échantillon. */
  n_candidates: number;
  by_parameter: Record<string, number>;
  articles_listed: number;
  articles: {
    id: number; title: string; year: number | null; doi: string | null;
    study_design: string | null; quality_score: number | null; parameters: string[];
  }[];
}

export interface EpidemicParameterExtraction {
  status: string;
  scenario_id?: string;
  n_candidates?: number;
  n_articles_with_values?: number;
  disease?: string | null;
  applicable?: boolean;
  parameters?: Record<string, PooledEpidemicParameter>;
  message?: string;
}

/** Lecture seule, sans appel de modèle : ce que la littérature du scénario contient
 *  avant toute extraction. */
export async function fetchEpidemicParameterCandidates(
  scenarioId: string,
  limit = 0,
): Promise<EpidemicParameterCandidates> {
  const r = await safeFetch(
    `${API_BASE_URL}/scenarios/${scenarioId}/epidemic-parameters/candidates?limit=${limit}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function extractEpidemicParameters(
  scenarioId: string,
): Promise<EpidemicParameterExtraction> {
  const r = await safeFetch(
    `${API_BASE_URL}/scenarios/${scenarioId}/epidemic-parameters/extract`,
    { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function getScenarioVariables(scenarioId: string): Promise<ScenarioVariables> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/variables?lang=${currentLang()}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function generateScenarioVariables(
  scenarioId: string,
): Promise<{ status: string; scenario_id?: string }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/variables/generate?lang=${currentLang()}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function getVariablesGenerationStatus(
  scenarioId: string,
): Promise<{ status: string; generated_at?: string; variables_count?: number; error?: string }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/variables/generate/status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function validateScenarioVariables(
  scenarioId: string,
  payload: { variables_json?: Record<string, unknown> },
): Promise<{ status: string; scenario_id: string; validated_at: string }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/variables/validate`, {
    method: 'POST',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Modèle entraîné : run, monitoring live, évolution (Phases 3-5) ───────────

export interface ModelRun {
  status: string; // ready | empty
  run_id?: number;
  family?: string;
  task_type?: string;
  metric?: string;
  metrics?: Record<string, number>;
  best_params?: Record<string, unknown>;
  feature_importances?: { feature: string; importance: number }[];
  summary?: Record<string, unknown>;
  has_artifact?: boolean;
  created_at?: string;
  message?: string;
}

export interface ModelMonitor {
  status: string; // ready | unavailable | error
  status_color?: 'green' | 'orange' | 'red' | 'unavailable';
  status_label?: string;
  value?: number;
  kind?: string;
  unit?: string | null;
  outcome?: string | null;
  positive_class?: string | null;
  bands?: { orange: number | null; red: number | null };
  n_scored?: number;
  window?: number;
  model?: { run_id: number; family: string; task_type: string; metric: string; metrics: Record<string, number> };
  alert_thresholds?: Record<string, { label?: string; condition?: string }>;
  generated_at?: string;
  message?: string;
}

export interface SpecDiff {
  has_changes: boolean;
  outcome_changed: boolean;
  outcome_fields: Record<string, { old: unknown; new: unknown }>;
  features_added: string[];
  features_removed: string[];
  features_changed: { machine_name: string; fields: Record<string, { old: unknown; new: unknown }> }[];
  algorithm_changed: boolean;
  algorithm_fields: Record<string, { old: unknown; new: unknown }>;
  /** The SEIR inputs. Absent from the diff until now, so a regeneration could take R0
   *  from 2.1 to 4.8 and the screen show no difference. */
  epidemic_parameters?: {
    params_added: string[];
    params_removed: string[];
    params_shifted: { param: string; old: number | null; new: number | null; relative: number | null }[];
    applicable_changed: boolean;
    applicable: { old: boolean; new: boolean };
    has_changes: boolean;
  };
  summary: {
    added: number; removed: number; changed: number;
    outcome_changed: boolean; algorithm_changed: boolean;
    epidemic_parameters_changed?: boolean;
  };
}

/** What a regeneration would change, and by how much: the full answer to "does this
 *  change the evidence or the model". Unlike the digest's cheap signals it compares two
 *  specs that were really generated. It applies nothing. */
export interface ChangeReport {
  status: "ready" | "empty" | "generating" | "error";
  scenario_id: string;
  has_changes?: boolean;
  /** What moved, in plain language, most consequential first. */
  changes?: string[];
  diff?: SpecDiff;
  corpus?: { active_articles: number | null; proposal_articles: number | null; delta?: number };
  active_generated_at?: string | null;
  proposal_generated_at?: string | null;
  applied?: boolean;
  apply_endpoint?: string;
  message?: string;
  error?: string;
}

export async function getChangeReport(scenarioId: string): Promise<ChangeReport> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/change-report`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export interface SpecProposal {
  status: string; // ready | empty | generating | error
  diff?: SpecDiff;
  proposal_spec?: Record<string, unknown>;
  active_version?: number;
  generated_at?: string;
  message?: string;
  error?: string;
}

export async function getModelRun(scenarioId: string): Promise<ModelRun> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/run`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function trainModel(scenarioId: string): Promise<{ status: string; scenario_id?: string; n_trials?: number }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/train`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// Entraîne plusieurs familles (LightGBM/XGBoost/GB/RF/linéaire) et garde la
// meilleure. Partage le même job + endpoint de statut que trainModel.
export async function compareModels(scenarioId: string): Promise<{ status: string; scenario_id?: string; n_trials?: number; mode?: string }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/compare`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function generateSyntheticData(
  scenarioId: string,
  nRows = 400,
): Promise<{ status: string; n_rows?: number; n_cols?: number }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/data/synthetic?n_rows=${nRows}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export interface ModelDataset {
  status: string; // ready | empty
  dataset_id?: number;
  n_rows?: number;
  n_cols?: number;
  validation?: {
    matched_features?: string[];
    missing_user?: string[];
    missing_public?: string[];
    missing_seir?: string[];
    target_present?: boolean;
    readiness?: { can_train: boolean; reasons: string[]; auto_fetchable?: string[] };
  };
  is_synthetic?: boolean;
  created_at?: string;
  message?: string;
}

export async function getModelDataset(scenarioId: string): Promise<ModelDataset> {
  // Authentifié : le schéma des données uploadées n'est pas public.
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/data`, { headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ── Phase 2: public-data connectors + auto-fetch ────────────────────────────
export interface ConnectorVariable { machine_name: string; label: string; unit?: string; dtype?: string; }
export interface DataConnector {
  id: string; name: string; provider: string; license: string; geo: string;
  commercial_ok: boolean; variables: ConnectorVariable[];
  params_schema?: Record<string, string>; notes?: string;
}

/** List the public-data connectors the app can auto-fetch variables from. */
export async function listDataConnectors(): Promise<DataConnector[]> {
  const r = await safeFetch(`${API_BASE_URL}/model/connectors`, { headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return (await r.json()).connectors ?? [];
}

export interface AutoFetchMapping { template_column: string; connector_id: string; connector_variable: string; }
export interface AutoFetchResponse {
  status: string; dataset_id?: number; n_rows: number; n_cols: number; frequency: string;
  filled_columns: string[]; still_needed_user_columns: string[];
  fetch_errors: Record<string, string>; validation?: ModelDataset["validation"];
  preview: Array<Record<string, unknown>>; training_started?: boolean;
}

/** Assemble the model dataset from public connectors instead of a CSV upload. */
export async function autoFetchModelData(scenarioId: string, payload: {
  region?: string; lat?: number; lon?: number; start_date: string; end_date: string;
  frequency?: string; mappings: AutoFetchMapping[]; auto_train?: boolean;
}): Promise<AutoFetchResponse> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/data/auto-fetch`, {
    method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function getModelTrainStatus(scenarioId: string): Promise<{ status: string; error?: string; metrics?: Record<string, number> }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/train/status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function getModelMonitor(scenarioId: string): Promise<ModelMonitor> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/monitor?lang=${currentLang()}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ── SEIR projection (literature-parameterized compartmental model) ──────────
export interface SeirBand { median: number[]; lower: number[]; upper: number[]; }
export interface SeirSummaryBand { median: number; lower: number; upper: number; }
export interface SeirParamValue {
  value: number;
  ci_low: number | null;
  ci_high: number | null;
  unit?: string;
  n_studies?: number;
  provenance?: number[];
  overridden?: boolean;
}
export interface SeirOverride { value: number; ci_low?: number | null; ci_high?: number | null; }
export interface SeirProjection {
  applicable: boolean;
  scenario_id: string;
  /** Texte français du backend (API / logs). L'UI affiche la traduction de `reason_code`. */
  reason?: string;
  /** Porte fermée, identifiant stable : traduit dans la langue choisie (seirReasonText). */
  reason_code?: "no_parameters" | "not_transmissible" | "no_transmission_parameter" | string;
  model?: string;
  disease?: string | null;
  /** true = projection obtenue via des paramètres SAISIS, pas extraits de la littérature. */
  forced?: boolean;
  /** Provenance du R₀ effectivement simulé - l'UI ne doit pas présenter "assumed"/"user" comme sourcé. */
  r0_source?: "literature" | "user" | "assumed";
  /** The infectious period decides gamma, hence the peak day, the peak height, the
   *  growth rate and the epidemic duration. It was silently set to 7 days on any corpus
   *  that does not report one, and the resulting curve was labelled "from the
   *  literature". Its provenance now travels with it. */
  infectious_period_days?: number | null;
  infectious_period_source?: "literature" | "user" | "assumed";
  parameter_sources?: Record<string, "literature" | "user" | "assumed">;
  assumed_parameters?: string[];
  /** Paramètres extraits mais inexploitables (le backend nomme ce qui manque). */
  missing?: string[];
  available_parameters?: string[];
  n_samples?: number;
  /** Tirages écartés de l'ensemble (divergence numérique) - diagnostic d'un IC d'entrée trop large. */
  n_dropped?: number;
  population?: number;
  initial_infected?: number;
  geography?: string | null;
  dates?: (string | number)[];
  series?: {
    incidence: SeirBand;
    prevalence: SeirBand;
    cumulative: SeirBand;
    deaths: SeirBand;
    r_eff: SeirBand;
    vaccinated?: SeirBand;     // présent si le modèle a une vaccination (V)
    quarantined?: SeirBand;    // présent si le modèle a une quarantaine (Q)
  };
  summary?: {
    model: string;
    r0: SeirSummaryBand;
    peak_incidence: SeirSummaryBand;
    peak_incidence_day: SeirSummaryBand;
    peak_prevalence: SeirSummaryBand;
    peak_prevalence_day: SeirSummaryBand;
    attack_rate: SeirSummaryBand;
    total_deaths: SeirSummaryBand;
    total_vaccinated?: SeirSummaryBand;
    peak_quarantine?: SeirSummaryBand;
  };
  parameters?: Record<string, SeirParamValue>;
  effective_parameters?: Record<string, SeirParamValue>;
  overrides_applied?: Record<string, number>;
  observed?: SeirObserved | null;        // série réelle superposée (si attachée)
}

export interface SeirObservedPoint { day: number; value: number; }
export interface SeirObserved {
  column: string; label?: string | null; source?: string | null; n: number;
  start_date?: string | null; scale?: number | null; fit_r2?: number | null;
  shift_days?: number | null;
  points: SeirObservedPoint[];           // en unités MODÈLE, au jour aligné
}
export interface SeirCalibration {
  ok: boolean; fitted_r0?: number; scale?: number; shift_days?: number;
  rmse?: number; r2?: number; n_points?: number; column?: string;
  horizon_days?: number; literature_r0?: number | null; detail?: string;
}

export async function fetchSeirProjection(
  scenarioId: string,
  params: { days?: number; start_date?: string; population?: number; initial_infected?: number; n_samples?: number } = {},
): Promise<SeirProjection> {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v != null) q.set(k, String(v)); });
  const qs = q.toString();
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/seir/projection${qs ? `?${qs}` : ""}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// Bundle de reproductibilité d'un modèle (spec + runs + hyperparamètres + dataset +
// prédiction) - authentifié (expose le schéma/les données du scénario).
export async function exportModelBundle(scenarioId: string, includeData = true): Promise<Record<string, unknown>> {
  const r = await safeFetch(
    `${API_BASE_URL}/scenarios/${scenarioId}/model/export?include_data=${includeData}`,
    { headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Relevant articles export (csv, xlsx, ris, bibtex, json, md) ─────────────
export const RELEVANT_EXPORT_FORMATS = ["csv", "xlsx", "ris", "bibtex", "json", "md"] as const;
export type RelevantExportFormat = typeof RELEVANT_EXPORT_FORMATS[number];

/** The relevant articles of a scenario as a file; the filename comes from the API. */
export async function exportRelevantArticles(
  scenarioId: string,
  format: RelevantExportFormat,
  opts: { includeAbstract?: boolean; threshold?: number } = {},
): Promise<{ blob: Blob; filename: string }> {
  const params = new URLSearchParams({ format });
  if (opts.includeAbstract === false) params.set("include_abstract", "false");
  // The threshold currently shown, so the file matches the count next to the button.
  if (typeof opts.threshold === "number") params.set("threshold", String(opts.threshold));
  const r = await safeFetch(`${scenarioBase(scenarioId)}/${scenarioId}/relevant/export?${params}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  const disposition = r.headers.get("Content-Disposition") ?? "";
  const m = /filename="?([^";]+)"?/.exec(disposition);
  return { blob: await r.blob(), filename: m?.[1] ?? `relevant-articles.${format === "bibtex" ? "bib" : format}` };
}

/** Shared tail of every subset export: read the server's filename, hand back the blob. */
async function _exportResponse(
  url: string, format: RelevantExportFormat, fallback: string,
): Promise<{ blob: Blob; filename: string }> {
  const r = await safeFetch(url);
  if (!r.ok) throw new Error(httpMessage(r.status));
  const m = /filename="?([^";]+)"?/.exec(r.headers.get("Content-Disposition") ?? "");
  const ext = format === "bibtex" ? "bib" : format;
  return { blob: await r.blob(), filename: m?.[1] ?? `${fallback}.${ext}` };
}

// ─── Narrowing the corpus by cluster or concept ──────────────────────────────
// The narrowing is a screening decision, written as `excluded` on this scenario's link
// rows, so PRISMA, the extractions, the exports and the threshold curve all pick it up
// without being told. `undecided` is the number to read before applying: articles the
// selection cannot judge (outside the capped clustering, or with no extracted concepts).
// They are kept unless `unassigned: "exclude"` says otherwise.
//
// These four go to /user-scenarios directly rather than through `scenarioBase`: that is
// the only prefix the routes are registered under, and the lookup behind them resolves a
// system scenario by id just as well, so the gesica prefix would only produce a 404.

export interface SubsetSelection {
  clusters?: number[];
  concepts?: string[];                       // "type:label", the map's canonical English
  concept_mode?: "any" | "all";
  /** Vocabulary labels from `api/study_design`, as the charts display them. */
  designs?: string[];
  levels?: string[];
  combine?: "any" | "all";
  unassigned?: "keep" | "exclude";
  reason?: string;
}

export interface SubsetPlan {
  relevant: number;
  keep: number;
  exclude: number;
  undecided: number;
  by_dimension: Record<string, { in: number; out: number; unknown: number }>;
  combine: string;
  unassigned: string;
  reason: string;
  meta: Record<string, any>;
  exclude_sample: number[];
  undecided_sample: number[];
  applied?: number;
  status?: string;
  caveat?: string;
  message?: string;
}

export interface SubsetState {
  scenario_id: string;
  narrowed: boolean;
  excluded_by_scope: number;
  /** The boundary: the narrowing only judged articles relevant at this threshold. */
  judged_above_threshold: number | null;
  steps: Array<{
    reason: string; articles: number; applied_at: string | null;
    applied_at_threshold: number | null;
  }>;
}

export async function previewScenarioSubset(
  scenarioId: string, selection: SubsetSelection,
): Promise<SubsetPlan> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}/subset/preview`, {
    method: 'POST',
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(selection),
  });
  if (!r.ok) throw new Error((await _detail(r)) || httpMessage(r.status));
  return r.json();
}

export async function applyScenarioSubset(
  scenarioId: string, selection: SubsetSelection,
): Promise<SubsetPlan> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}/subset/apply`, {
    method: 'POST',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(selection),
  });
  if (!r.ok) throw new Error((await _detail(r)) || httpMessage(r.status));
  return r.json();
}

export async function fetchScenarioSubsetState(scenarioId: string): Promise<SubsetState> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}/subset`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function undoScenarioSubset(
  scenarioId: string, reason?: string,
): Promise<{ restored: number; status: string }> {
  const qs = reason ? `?reason=${encodeURIComponent(reason)}` : '';
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}/subset/undo${qs}`, {
    method: 'POST', headers: authHeaders(),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

/** The API's own `detail` when it has one: "cluster 42 unknown" is worth more to the
 *  reader than "404". */
async function _detail(r: Response): Promise<string> {
  try {
    const body = await r.json();
    return typeof body?.detail === "string" ? body.detail : "";
  } catch {
    return "";
  }
}

/** One cluster's articles. Complete for that cluster; the clustering itself is a
 *  projection over the most relevant CLUSTER_MAX_DOCS, which the file states. */
export async function exportClusterArticles(
  scenarioId: string, clusterId: number, format: RelevantExportFormat,
  opts: { includeAbstract?: boolean; lang?: string } = {},
): Promise<{ blob: Blob; filename: string }> {
  const params = new URLSearchParams({ format });
  if (opts.includeAbstract === false) params.set("include_abstract", "false");
  if (opts.lang) params.set("lang", opts.lang);
  return _exportResponse(
    `${scenarioBase(scenarioId)}/${scenarioId}/clusters/${clusterId}/export?${params}`,
    format, `cluster-${clusterId}`);
}

/** The articles behind one concept or a selection of them. `mode` is "any" for the
 *  union a filtered map shows, "all" for the intersection. Never capped at the map's
 *  40-article display limit: the server recomputes the graph uncapped. */
export async function exportConceptArticles(
  scenarioId: string, concepts: Array<{ type: string; label: string }>,
  format: RelevantExportFormat,
  opts: { includeAbstract?: boolean; mode?: "any" | "all" } = {},
): Promise<{ blob: Blob; filename: string }> {
  const params = new URLSearchParams({
    format, concepts: concepts.map(c => `${c.type}:${c.label}`).join("|"),
  });
  if (opts.mode) params.set("mode", opts.mode);
  if (opts.includeAbstract === false) params.set("include_abstract", "false");
  return _exportResponse(
    `${scenarioBase(scenarioId)}/${scenarioId}/concepts/export?${params}`, format, "concepts");
}

/** An explicit list of articles: the RAG sources, a filtered map, a hand-picked set.
 *  Ids outside this scenario are dropped server-side, never exported. */
export async function exportArticleIds(
  scenarioId: string, ids: number[], format: RelevantExportFormat,
  opts: { includeAbstract?: boolean; label?: string } = {},
): Promise<{ blob: Blob; filename: string }> {
  const params = new URLSearchParams({ format, ids: ids.join(",") });
  if (opts.label) params.set("label", opts.label);
  if (opts.includeAbstract === false) params.set("include_abstract", "false");
  return _exportResponse(
    `${scenarioBase(scenarioId)}/${scenarioId}/articles/export?${params}`, format,
    opts.label ?? "selection");
}

/** Hand a blob to the browser as a download. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Même export, en classeur Excel (.xlsx) : feuilles Variables / Dataset (valeurs +
// issue) / Model runs. Renvoie le binaire à télécharger.
export async function exportModelXlsx(scenarioId: string): Promise<Blob> {
  const r = await safeFetch(
    `${API_BASE_URL}/scenarios/${scenarioId}/model/export.xlsx`,
    { headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.blob();
}

// Projection AVEC paramètres modifiés par l'utilisateur (onglet SEIR) - explore des
// variantes sans altérer les paramètres source extraits de la littérature.
export async function postSeirProjection(
  scenarioId: string,
  body: {
    days?: number; start_date?: string; population?: number; initial_infected?: number;
    n_samples?: number; overrides?: Record<string, SeirOverride>;
  } = {},
): Promise<SeirProjection> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/seir/projection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// Attache une série OBSERVÉE (réelle) au SEIR : upload de points {date|day, value}
// OU tirage d'un connecteur (foph-sentinella-ili, foph-wastewater…). Superposée au
// graphe + base de la calibration. Au moins 3 points.
export async function postSeirObserved(
  scenarioId: string,
  body: {
    points?: { date?: string | number; day?: number; value: number }[];
    connector_id?: string; connector_variable?: string; region?: string;
    start_date?: string; end_date?: string; column?: string; label?: string;
  },
): Promise<{ ok: boolean; n: number; column: string; source: string; label?: string; start_date?: string | null }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/seir/observed`, {
    method: "POST", headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function deleteSeirObserved(scenarioId: string): Promise<{ ok: boolean }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/seir/observed`, {
    method: "DELETE", headers: authHeaders(),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// Calibre R0 (+ échelle + décalage temporel) du SEIR sur la série observée attachée.
export async function calibrateSeir(
  scenarioId: string,
  body: { column?: string; overrides?: Record<string, SeirOverride>; days?: number } = {},
): Promise<SeirCalibration> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/seir/calibrate`, {
    method: "POST", headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export interface ProvArticle {
  id: number;
  title: string;
  year?: number | null;
  doi?: string | null;
  citation_count?: number | null;
  url?: string | null;
}

export interface ModelSpecResponse {
  status: string; // ready | empty | legacy
  outcome?: { name?: string; machine_name?: string; task_type?: string; unit?: string; best_article?: ProvArticle | null };
  features?: Array<{ name?: string; machine_name?: string; dtype?: string; source?: string; importance?: string; best_article?: ProvArticle | null }>;
  algorithm?: { family?: string; metric?: string; candidates?: string[]; best_article?: ProvArticle | null };
  // Modalités d'alerte (seuils green/orange/red) enrichies côté serveur : chaque
  // niveau porte l'article source le plus pertinent + la liste résolue, pour lier
  // la modalité aux articles du pool pertinent qui la justifient.
  alert_thresholds?: Record<string, {
    label?: string; range?: string; rationale?: string; description?: string;
    best_article?: ProvArticle | null;
    provenance_articles?: ProvArticle[];
  }>;
  provenance_index?: Record<string, ProvArticle>;
  validated?: boolean;
  message?: string;
}

export async function getScenarioModelSpec(scenarioId: string): Promise<ModelSpecResponse> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/spec?lang=${currentLang()}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export interface EditSpecPayload {
  algorithm_family?: string;
  metric?: string;
  task_type?: string;
  positive_class?: string | null;
  remove_features?: string[];                                  // machine_names to drop
  add_features?: { name: string; dtype?: string; importance?: string }[];
  retrain?: boolean;
}

export interface EditSpecResponse {
  status: string;                                              // updated | unchanged
  new_version?: number;
  outcome?: { machine_name?: string; name?: string; task_type?: string; positive_class?: string | null };
  algorithm?: { family?: string; metric?: string };
  features?: { name: string; machine_name: string; dtype: string; importance?: string }[];
  warnings?: string[];
  retrain_started?: boolean;
}

// Édite directement le spec actif (famille, task_type, ajout/suppression de
// variables) et ré-entraîne si demandé. lang → alertes task↔cible localisées.
export async function editModelSpec(scenarioId: string, payload: EditSpecPayload): Promise<EditSpecResponse> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/spec/edit?lang=${currentLang()}`, {
    method: 'POST',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// Outcomes prêts à l'emploi (GESICA) : surcharge urgences, occupation lits, appels, pic d'appels.
export interface OutcomeTemplate {
  id: string; name: string; description: string;
  outcome: { name: string; machine_name: string; task_type: string; unit?: string; positive_class?: string | null };
  algorithm: { family: string; metric?: string; quantile?: number };
  features: { name: string; machine_name: string; dtype: string }[];
}
export async function listOutcomeTemplates(): Promise<OutcomeTemplate[]> {
  const r = await safeFetch(`${API_BASE_URL}/model/outcome-templates`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return (await r.json()).templates ?? [];
}
export async function applyOutcomeTemplate(scenarioId: string, templateId: string): Promise<{ status: string; model_spec?: unknown }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/outcome-template`, {
    method: 'POST',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ template_id: templateId }),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function proposeSpec(scenarioId: string): Promise<{ status: string; scenario_id?: string }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/spec/propose?lang=${currentLang()}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function getSpecProposal(scenarioId: string): Promise<SpecProposal> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/spec/proposal`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function validateSpecProposal(
  scenarioId: string,
  action: 'accept' | 'reject',
): Promise<{ status: string; new_version?: number; retrain_started?: boolean }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/spec/proposal/validate`, {
    method: 'POST',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ action }),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Heatmap avec vrais noms ──────────────────────────────────────────────────

export async function fetchCorpusStatsByYearNamed(): Promise<CorpusStatsByYear> {
  const r = await safeFetch(`${API_BASE_URL}/corpus/stats/by-year/named`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  const data = await r.json();
  return {
    byYear: data.by_year,
    scenarioByYear: data.scenario_by_year,
    heatmapScenarioSource: data.heatmap_scenario_source,
  };
}

// ─── Assistant IA filtre par seuil ───────────────────────────────────────────

export function askScenarioRagStreamFiltered(
  scenarioId: string,
  question: string,
  callbacks: RagStreamCallbacks,
): () => void {
  let aborted = false;
  const controller = new AbortController();

  (async () => {
    try {
      const resp = await safeFetch(`${API_BASE_URL}/ask/stream/filtered`, {
        method: 'POST',
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          question,
          project_context: 'literev',
          scenario_id: scenarioId,
          top_k: 12,
          lang: currentLang(),
        }),
        signal: controller.signal,
      });

      if (!resp.ok) throw new Error(httpMessage(resp.status));
      if (!resp.body) throw new Error('No response body');

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (!aborted) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (line.startsWith('event: sources')) continue;
          if (line.startsWith('event: meta')) continue;
          if (line.startsWith('event: error')) continue;
          if (line.startsWith('event: done')) {
            callbacks.onDone();
            return;
          }
          if (line.startsWith('data: ')) {
            const raw = line.slice(6).trim();
            if (!raw || raw === '{}') continue;
            try {
              const parsed = JSON.parse(raw);
              if (parsed.token !== undefined) {
                callbacks.onToken(parsed.token);
              } else if (Array.isArray(parsed)) {
                callbacks.onSources(parsed as ScenarioRagSource[]);
              } else if (parsed.papers_used !== undefined) {
                callbacks.onMeta?.(parsed as RagMeta);
              } else if (parsed.error) {
                callbacks.onError(parsed.error);
              }
            } catch {}
          }
        }
      }
      callbacks.onDone();
    } catch (e: any) {
      if (!aborted) callbacks.onError(e.message ?? 'Erreur streaming');
    }
  })();

  return () => {
    aborted = true;
    controller.abort();
  };
}

// ─── Live Federated Search ────────────────────────────────────────────────────

export interface LiveSearchResult {
  title: string;
  abstract?: string | null;
  doi?: string | null;
  year?: number | null;
  authors?: string[];
  journal?: string | null;
  url?: string | null;
  external_id?: string | null;
  source_name: string;
  in_local_db: boolean;
  semantic_score?: number | null;
  lexical_score?: number | null;
  hybrid_score?: number | null;
  also_in_sources?: string[];
}

export interface LiveSearchResponse {
  results: LiveSearchResult[];
  total: number;
  new_count: number;
  corpus_total?: number;
  corpus_above_threshold?: number;
  threshold?: number;
  /** Only the sources whose answer is in `results`: a source that failed is no longer
   *  listed here, because the panel printed it among the sources it had searched. */
  sources_queried: string[];
  /** Per source: ok, empty, partial, error, timeout, with its latency and count. Built
   *  and returned by the API since the start, and rendered nowhere until now. */
  source_status?: Record<string, {
    status: string;
    count?: number;
    fetched?: number;
    latency_ms?: number | null;
    error?: string;
  }>;
  source_raw_counts?: Record<string, number>;
  ingesting_background: boolean;
}

export async function searchLive(
  scenarioId: string,
  maxPerSource = 50,
): Promise<LiveSearchResponse> {
  const r = await safeFetch(
    // `lang`: this starts a corpus build and the pipeline that follows it, so it carries
    // the interface language like the other four entry points.
    `${API_BASE_URL}/user-scenarios/${scenarioId}/search/live?max_per_source=${maxPerSource}&lang=${currentLang()}`,
    { method: 'POST', headers: authHeaders() },
  );
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export interface SearchStrategy {
  general: string;
  pubmed: string;
  explanation: string;
  synonyms: string[][];
}

/** Traduit une requête en langage naturel en stratégie booléenne (LLM). */
export async function fetchSearchStrategy(query: string): Promise<SearchStrategy> {
  const r = await safeFetch(`${API_BASE_URL}/search-strategy`, {
    method: 'POST',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ query }),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function getSearchStrategy(scenarioId: string): Promise<SearchStrategy> {
  const r = await safeFetch(
    `${API_BASE_URL}/user-scenarios/${scenarioId}/search-strategy`,
    { headers: authHeaders() },
  );
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

/** Prévisualise (lexical, bibliothèque locale) le compte d'articles par facette et le
 *  total par union (OU) / intersection (ET) - AVANT de lancer la recherche complète. */
export async function previewSearchFacets(
  subQueries: SubQuery[],
  combinator: "union" | "intersection",
  filters: Record<string, any> = {},
): Promise<FacetPreviewResponse> {
  const r = await safeFetch(`${API_BASE_URL}/search-facets`, {
    method: 'POST',
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ sub_queries: subQueries, combinator, filters }),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ── ReliefWeb : rapports de situation (flux SÉPARÉ du corpus scientifique) ────
// Littérature GRISE : jamais mélangée à literature_document, jamais comptée dans
// PRISMA, et plafonnée en crédibilité pour ne pas peser comme une étude publiée.
export interface SituationReport {
  id: number;
  rw_id: string;
  title: string;
  url: string | null;
  published_at: string | null;
  sources: string[];
  format: string | null;
  primary_country: string | null;
  primary_iso3: string | null;
  glide: string | null;
  disaster_types: string[];
  themes: string[];
  language: string | null;
  /** 0..1, plafonnée à 0.45 - très en dessous de toute étude revue par les pairs. */
  credibility: number;
  excerpt: string;
}

export interface SituationReportsPage {
  total: number;
  limit: number;
  offset: number;
  reports: SituationReport[];
  grey_literature: boolean;
  max_credibility: number;
}

export interface ReliefWebStatus {
  configured: boolean;
  base_url?: string;
  quota_calls_per_day?: number;
  calls_used_today?: number;
  calls_remaining_today?: number;
  reports?: number;
  countries?: number;
  newest_published_at?: string | null;
  error?: string;
}

export async function fetchSituationReports(
  options?: { scenarioId?: string; iso3?: string; glide?: string; q?: string; limit?: number; offset?: number },
): Promise<SituationReportsPage> {
  const params = new URLSearchParams();
  if (options?.scenarioId) params.set("scenario_id", options.scenarioId);
  if (options?.iso3) params.set("iso3", options.iso3);
  if (options?.glide) params.set("glide", options.glide);
  if (options?.q) params.set("q", options.q);
  if (options?.limit) params.set("limit", String(options.limit));
  if (options?.offset) params.set("offset", String(options.offset));
  const r = await safeFetch(`${API_BASE_URL}/reliefweb/reports?${params}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchReliefWebStatus(): Promise<ReliefWebStatus> {
  const r = await safeFetch(`${API_BASE_URL}/reliefweb/status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function ingestSituationReports(
  body: { scenario_id?: string; pathogens?: string[]; countries?: string[]; date_from?: string; limit?: number; max_calls?: number },
): Promise<Record<string, unknown>> {
  const r = await safeFetch(`${API_BASE_URL}/reliefweb/ingest`, {
    method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Questions asked of a scenario ───────────────────────────────────────────
// Une réponse de l'assistant n'est pas un tour de discussion qui défile : elle a
// une PORTÉE (quel scénario, quel seuil, quel resserrement), une DATE et des
// SOURCES. Conservée avec les trois, elle redevient comparable, citable et
// vérifiable, et peut proposer une mise à jour du scénario.

export interface QuestionProposal {
  key: string;
  value: number | null;
  low: number | null;
  high: number | null;
  unit: string | null;
  /** La valeur que le scénario tient aujourd'hui, ou null s'il n'en tient aucune. */
  current: number | null;
  kind: "update" | "new";
  quote?: string;
  decision?: "accepted" | "rejected";
  decided_at?: string;
}

export interface ScenarioQuestion {
  id: number;
  scenario_id: string;
  question: string;
  answer: string;
  lang: string | null;
  threshold: number | null;
  scope: Record<string, unknown> | null;
  /** La portée en une phrase, telle que le serveur la formule. */
  scope_label: string;
  sources: Array<{ document_id?: number; title?: string; authors?: string;
                   year?: number; doi?: string; score?: number }> | null;
  papers_used: number | null;
  /** The number of articles RETRIEVED to compose the answer. The server still stores it
   *  in a column named papers_quoted; nothing measures how many the answer quotes. */
  papers_quoted: number | null;
  digest_complete: boolean;
  proposals: QuestionProposal[] | null;
  created_at: string | null;
  /** Combien de fois ce même libellé apparaît dans la page listée. */
  asked_times_in_page?: number;
}

export interface ScenarioQuestionsPage {
  scenario_id: string;
  total: number;
  items: ScenarioQuestion[];
}

export async function saveScenarioQuestion(
  scenarioId: string,
  body: {
    question: string; answer: string; lang?: string | null; threshold?: number | null;
    scope?: Record<string, unknown>; sources?: unknown[];
    papers_used?: number | null; papers_retrieved?: number | null;
    digest_complete?: boolean;
  },
): Promise<ScenarioQuestion> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}/questions`, {
    method: "POST",
    // La clé, comme pour la suppression de la même ligne : l'écriture dans
    // l'historique ne demandait rien, pendant que le badge disait « lecture seule ».
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchScenarioQuestions(
  scenarioId: string, options?: { limit?: number; offset?: number },
): Promise<ScenarioQuestionsPage> {
  const params = new URLSearchParams();
  if (options?.limit) params.set("limit", String(options.limit));
  if (options?.offset) params.set("offset", String(options.offset));
  const r = await safeFetch(
    `${API_BASE_URL}/user-scenarios/${scenarioId}/questions?${params}`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function deleteScenarioQuestion(scenarioId: string, questionId: number): Promise<void> {
  const r = await safeFetch(
    `${API_BASE_URL}/user-scenarios/${scenarioId}/questions/${questionId}`,
    { method: "DELETE", headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
}

export async function decideQuestionProposal(
  scenarioId: string, questionId: number, key: string,
  decision: "accepted" | "rejected",
): Promise<{ proposals: QuestionProposal[] }> {
  const r = await safeFetch(
    `${API_BASE_URL}/user-scenarios/${scenarioId}/questions/${questionId}/proposals`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ key, decision }),
    });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

/** Lien de téléchargement d'une réponse : Markdown, Word ou PDF. */
export function questionExportUrl(
  scenarioId: string, questionId: number, format: "md" | "docx" | "pdf",
): string {
  return `${API_BASE_URL}/user-scenarios/${scenarioId}/questions/${questionId}/export?format=${format}`;
}
