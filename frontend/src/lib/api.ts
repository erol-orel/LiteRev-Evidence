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
  // Respecte Retry-After mais le plafonne : inutile de figer l'UI 60 s — mieux
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

// --- EMS / STATS TYPES ---

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
    legacy_full_text_to_retype: number;
    junk_to_delete: number;
    substantive_kept_reported: number;
    retyped?: number;
    deleted_junk?: number;
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
  const response = await safeFetch(`${API_BASE_URL}/gesica/scenarios`);
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

export interface ScenarioDetail {
  id: string;
  title: string;
  description: string;
  cluster: string;
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
  below_threshold?: number;
  unscored?: number;
  from_local?: number | null;
  newly_fetched?: number | null;
  docs_with_fulltext?: number;
  docs_abstract_only?: number;
  source_breakdown?: Record<string, number>;
  rerank_running?: boolean;
  threshold?: number;
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

export interface ClusterResult {
  cluster_id: number;
  cluster_name: string;
  is_noise: boolean;
  n_docs: number;
  center_x: number;
  center_y: number;
  top_words: string[];
  summary: string;
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
  n_clusters?: number;
  clusters: ClusterResult[];
  topics: ClusterTopic[];
  message?: string;
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
    manually_rescued: number;
    manually_vetoed: number;
  };
  evidence: {
    total: number;
    ai_auto_selected: number;
    manually_rescued: number;
    with_fulltext: number;
    screening_complete: boolean;
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
  const response = await safeFetch(`${base}/${scenarioId}/detail`);
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
  articles: PicoBulkArticle[];
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
  study_design_distribution: Array<{ design: string; count: number }>;
  year_distribution: Array<{ year: number; count: number }>;
  source_distribution: Array<{ source: string; count: number }>;
  evidence_level_distribution: Array<{ level: string; count: number }>;
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

export interface RagStreamCallbacks {
  onSources: (sources: ScenarioRagSource[]) => void;
  onToken: (token: string) => void;
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
  reviewer: 1 | 2;
  status: "included" | "excluded" | "pending";
  reason?: string;
  reviewer_code?: string;
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
  mode: string;
  filters: Record<string, any>;
  result_count: number;
  resultCount: number;           // alias camelCase de result_count
  pinned: boolean;
  created_at: string | null;
  updated_at: string | null;
  is_user_scenario: true;
  populate_status?: string;
  pipeline_status?: string;
  pipeline_step?: string | null;
  pipeline_progress?: number;
}

export interface UserScenarioCreatePayload {
  name: string;
  query: string;
  mode: string;
  filters: Record<string, any>;
  result_count?: number;
  pinned?: boolean;
  search_strategy?: SearchStrategy | null;
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
  // Pertinence (ranking) — scores réellement présents sur le corpus (≠ indexation RAG).
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
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios`, {
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
  patch: { name?: string; pinned?: boolean; mode?: string; filters?: Record<string, any>; folder_id?: string | null },
): Promise<UserScenario> {
  const r = await safeFetch(`${API_BASE_URL}/user-scenarios/${scenarioId}`, {
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
    `${API_BASE_URL}/user-scenarios/${scenarioId}/pipeline?max_results=${maxResults}`,
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

/** Construit le corpus (= requête booléenne sur base locale ∪ live) en arrière-plan. */
export async function populateUserScenario(
  scenarioId: string,
  opts?: { includeLive?: boolean; maxResults?: number },
): Promise<{ scenario_id: string; status: string; message?: string }> {
  const params = new URLSearchParams();
  params.set('max_results', String(opts?.maxResults ?? 2000));
  params.set('include_live', String(opts?.includeLive ?? true));
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
  status: string;
  ingested?: number;
  // Phase RÉELLE du backend (et non un minuteur côté client).
  phase?: 'local' | 'federation' | 'scoring' | 'done';
  // Statut du cross-encoder Cohere qui réordonne en arrière-plan après l'affichage.
  rerank_status?: 'idle' | 'running' | 'done' | 'skipped';
  sources?: Record<string, number>;
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
): Promise<{ status: string; message: string }> {
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
  const params = new URLSearchParams({ dry_run: String(dryRun) });
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

export interface EnrichmentStatus {
  scenario_id: string | null;
  total: number;
  pico: { count: number; pct: number };
  metadata: { count: number; pct: number };
  fulltext: { count: number; pct: number };
}

export async function extractPicoBatchGlobal(
  scenarioId?: string,
  limit = 100000,
): Promise<EnrichmentBatchResult> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (scenarioId) params.set('scenario_id', scenarioId);
  const r = await safeFetch(`${API_BASE_URL}/pico/extract?${params}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function extractMetadataBatch(
  scenarioId?: string,
  limit = 100000,
): Promise<EnrichmentBatchResult> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (scenarioId) params.set('scenario_id', scenarioId);
  const r = await safeFetch(`${API_BASE_URL}/metadata/extract?${params}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchFulltextBatch(
  scenarioId?: string,
  limit = 100000,
): Promise<EnrichmentBatchResult & { fetched: number; not_available: number }> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (scenarioId) params.set('scenario_id', scenarioId);
  const r = await safeFetch(`${API_BASE_URL}/fulltext/fetch?${params}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function fetchEnrichmentStatus(
  scenarioId?: string,
): Promise<EnrichmentStatus> {
  const params = new URLSearchParams();
  if (scenarioId) params.set('scenario_id', scenarioId);
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

export interface ScenarioSettings {
  scenario_id: string;
  similarity_threshold: number;
  evidence_brief_json: Record<string, unknown> | null;
  brief_generated_at: string | null;
  variables_json: Record<string, unknown> | null;
  variables_validated: boolean;
  variables_generated_at: string | null;
}

export async function getScenarioSettings(scenarioId: string): Promise<ScenarioSettings> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/settings`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function patchScenarioSettings(
  scenarioId: string,
  payload: Partial<Pick<ScenarioSettings, 'similarity_threshold' | 'variables_json' | 'variables_validated'>>,
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
): Promise<{ status: string; scenario_id: string; query?: string }> {
  const params = query ? `?query=${encodeURIComponent(query)}` : '';
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/rerank${params}`, { method: 'POST', headers: authHeaders() });
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function getRerankStatus(
  scenarioId: string,
): Promise<{ status: string; updated?: number; error?: string }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/rerank/status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

// ─── Evidence Brief LLM ──────────────────────────────────────────────────────

export interface LlmEvidenceBrief {
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
  };
  _cached?: boolean;
  _generated_at?: string | null;
  status?: string;
  message?: string;
  error?: string;
}

export async function getLlmEvidenceBrief(scenarioId: string): Promise<LlmEvidenceBrief> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/evidence-brief/llm`);
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

export async function getScenarioVariables(scenarioId: string): Promise<ScenarioVariables> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/variables`);
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
  summary: { added: number; removed: number; changed: number; outcome_changed: boolean; algorithm_changed: boolean };
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
    target_present?: boolean;
    readiness?: { can_train: boolean; reasons: string[]; auto_fetchable?: string[] };
  };
  created_at?: string;
  message?: string;
}

export async function getModelDataset(scenarioId: string): Promise<ModelDataset> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/data`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function getModelTrainStatus(scenarioId: string): Promise<{ status: string; error?: string; metrics?: Record<string, number> }> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/train/status`);
  if (!r.ok) throw new Error(httpMessage(r.status));
  return r.json();
}

export async function getModelMonitor(scenarioId: string): Promise<ModelMonitor> {
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/monitor`);
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
  algorithm?: { family?: string; metric?: string; best_article?: ProvArticle | null };
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
  const r = await safeFetch(`${API_BASE_URL}/scenarios/${scenarioId}/model/spec`);
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
  sources_queried: string[];
  ingesting_background: boolean;
}

export async function searchLive(
  scenarioId: string,
  maxPerSource = 50,
): Promise<LiveSearchResponse> {
  const r = await safeFetch(
    `${API_BASE_URL}/user-scenarios/${scenarioId}/search/live?max_per_source=${maxPerSource}`,
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
