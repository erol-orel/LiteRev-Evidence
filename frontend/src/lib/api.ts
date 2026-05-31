import type {
  SearchRequest,
  SearchResponse,
  SearchResult,
  ApiSearchRequest,
  ApiSearchResponse,
  ApiSearchResult,
  ApiSearchFilters,
} from "../types/search";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

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

function mapFiltersToApi(filters?: SearchRequest["filters"]): ApiSearchFilters | undefined {
  if (!filters) return undefined;
  return {
    project_context: filters.projectContext,
    source_type: filters.sourceType,
    disease_or_condition: filters.diseaseOrCondition,
    scenario_type: filters.scenarioType,
    geographic_scope: filters.geographicScope,
    evidence_category: filters.evidenceCategory,
    year_min: filters.yearMin,
    year_max: filters.yearMax,
  };
}

function mapSearchResultFromApi(apiResult: ApiSearchResult): SearchResult {
  return {
    id: apiResult.id,
    documentId: apiResult.document_id,
    chunkId: apiResult.chunk_id,
    chunkIndex: apiResult.chunk_index,
    title: apiResult.title,
    abstract: apiResult.abstract,
    content: apiResult.content,
    highlight: apiResult.highlight,
    score: apiResult.score,
    source: apiResult.source,
    year: apiResult.year,
    url: apiResult.url,
    projectContext: apiResult.project_context,
    sourceType: apiResult.source_type,
    diseaseOrCondition: apiResult.disease_or_condition,
    scenarioType: apiResult.scenario_type,
    geographicScope: apiResult.geographic_scope,
    evidenceCategory: apiResult.evidence_category,
    chunkType: apiResult.chunk_type,
  };
}

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

export async function searchDocuments(
  payload: SearchRequest,
): Promise<SearchResponse> {
  const apiPayload: ApiSearchRequest = {
    query_text: payload.queryText,
    mode: payload.mode,
    limit: payload.limit,
    filters: mapFiltersToApi(payload.filters),
  };

  const response = await fetch(`${API_BASE_URL}/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(apiPayload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Search failed with status ${response.status}`);
  }

  const apiData: ApiSearchResponse = await response.json();
  return {
    results: (apiData.results || []).map(mapSearchResultFromApi),
  };
}

export async function getFilterOptions(): Promise<FilterOptions> {
  const response = await fetch(`${API_BASE_URL}/filters-options`);

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
  const response = await fetch(`${API_BASE_URL}/documents/${documentId}`);

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

// --- GESICA / STATS TYPES ---

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
  description: string;
  cluster: string;
  articleCount: number;
  livingEvidenceNote: string;
  recommendedActions: string[];
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
  };
  embeddings: {
    total_chunks: number;
    chunks_with_embedding: number;
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
  const response = await fetch(`${API_BASE_URL}/corpus/fulltext-stats`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
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

export interface ScreeningDocument {
  id: number;
  title: string;
  abstract: string | null;
  year: number | null;
  source: string;
  url: string | null;
  externalId: string | null;
  projectContext: string;
  screeningStatus: "included" | "excluded" | "pending" | null;
  screeningReason: string | null;
  screeningNotes: string | null;
}

export interface ScreeningDecision {
  documentId: number;
  status: "included" | "excluded" | "pending";
  reason?: string;
  notes?: string;
}

export interface PrismaFlow {
  recordsIdentified: number;
  recordsScreened: number;
  recordsExcluded: number;
  recordsIncluded: number;
  exclusionReasons: Record<string, number>;
  bySource: Record<string, number>;
}

export async function fetchEvidenceSummary(
  documentId: number,
): Promise<EvidenceSummaryResponse> {
  const response = await fetch(`${API_BASE_URL}/evidence-summary/${documentId}`);
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
  const response = await fetch(`${API_BASE_URL}/corpus/stats`);
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

export async function fetchGesicaStats(): Promise<GesicaStats> {
  const response = await fetch(`${API_BASE_URL}/gesica/stats`);
  if (!response.ok) throw new Error(`GESICA stats failed with status ${response.status}`);
  const data = await response.json();
  return {
    totalDocuments: data.total_documents,
    evidenceStrengthDistribution: data.evidence_strength_distribution,
    uncertaintyMethods: data.uncertainty_methods,
    forecastHorizons: data.forecast_horizons,
  };
}

export async function fetchGesicaScenarios(): Promise<GesicaScenario[]> {
  const response = await fetch(`${API_BASE_URL}/gesica/scenarios`);
  if (!response.ok) throw new Error(`GESICA scenarios failed with status ${response.status}`);
  const data: Array<{
    id: string;
    title: string;
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
    description: s.description,
    cluster: s.cluster,
    articleCount: s.article_count,
    livingEvidenceNote: s.living_evidence_note,
    recommendedActions: s.recommended_actions,
    relevantArticles: s.relevant_articles,
  }));
}

export async function fetchScreeningList(projectContext?: string): Promise<ScreeningDocument[]> {
  const url = projectContext 
    ? `${API_BASE_URL}/screening?project_context=${projectContext}`
    : `${API_BASE_URL}/screening`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Screening list failed with status ${response.status}`);
  const data = await response.json();
  return data.map((d: any) => ({
    id: d.id,
    title: d.title,
    abstract: d.abstract,
    year: d.year,
    source: d.source,
    url: d.url ?? null,
    externalId: d.external_id ?? null,
    projectContext: d.project_context,
    screeningStatus: d.screening_status ?? "pending",
    screeningReason: d.screening_reason ?? null,
    screeningNotes: d.screening_notes ?? null,
  }));
}

export async function submitScreeningDecision(decision: ScreeningDecision): Promise<void> {
  const token = localStorage.getItem("api_key") || "";
  const response = await fetch(`${API_BASE_URL}/screening/decision`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": token
    },
    body: JSON.stringify({
      document_id: decision.documentId,
      status: decision.status,
      reason: decision.reason,
      notes: decision.notes
    })
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Screening decision failed with status ${response.status}`);
  }
}

export async function fetchPrismaFlow(projectContext?: string): Promise<PrismaFlow> {
  const url = projectContext
    ? `${API_BASE_URL}/screening/prisma?project_context=${projectContext}`
    : `${API_BASE_URL}/screening/prisma`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`PRISMA flow failed with status ${response.status}`);
  const d = await response.json();
  // Support both flat (new) and nested (legacy) API response shapes
  if (typeof d.records_identified !== "undefined") {
    // New flat format from main.py
    return {
      recordsIdentified: d.records_identified ?? 0,
      recordsScreened: d.records_screened ?? 0,
      recordsExcluded: d.records_excluded ?? 0,
      recordsIncluded: d.records_included ?? 0,
      exclusionReasons: d.exclusion_reasons ?? {},
      bySource: d.by_source ?? {},
    };
  }
  // Legacy nested format fallback
  return {
    recordsIdentified: d.identification?.total_records ?? 0,
    recordsScreened: d.screening?.records_screened ?? 0,
    recordsExcluded: d.screening?.records_excluded ?? 0,
    recordsIncluded: d.included?.total_included ?? 0,
    exclusionReasons: d.eligibility?.reasons ?? {},
    bySource: d.identification?.by_source ?? {},
  };
}

export async function askAssistant(req: AskRequest): Promise<AskResponse> {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question: req.question,
      project_context: req.projectContext || null,
      filters: req.filters || null,
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
  const response = await fetch(`${API_BASE_URL}/terrain/meteo?lat=${lat}&lon=${lon}`);
  if (!response.ok) throw new Error(`Terrain meteo failed with status ${response.status}`);
  return response.json();
}

export async function fetchTerrainGeo(
  origLat = 46.2044, origLon = 6.1432,
  destLat = 46.1925, destLon = 6.2388
): Promise<TerrainGeo> {
  const url = `${API_BASE_URL}/terrain/geo?orig_lat=${origLat}&orig_lon=${origLon}&dest_lat=${destLat}&dest_lon=${destLon}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Terrain geo failed with status ${response.status}`);
  return response.json();
}

export async function fetchTerrainEpidemic(region = "transborder"): Promise<TerrainEpidemic> {
  const response = await fetch(`${API_BASE_URL}/terrain/epidemic?region=${region}`);
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
  impact_on_gesica?: string;
  impact_on_geoai4ei?: string;
}

export interface TerrainInformalSignals {
  source: string;
  active_signals: TerrainSignal[];
  architecture_note: string;
}

// ─── P5 TERRAIN EXTENDED API FUNCTIONS ────────────────────────────────────────

export async function fetchTerrainDemographics(postalCode = "74100"): Promise<TerrainDemographics> {
  const response = await fetch(`${API_BASE_URL}/terrain/demographics?postal_code=${postalCode}`);
  if (!response.ok) throw new Error(`Terrain demographics failed with status ${response.status}`);
  return response.json();
}

export async function fetchTerrainPharmacies(lat = 46.2044, lon = 6.1432): Promise<TerrainPharmacies> {
  const response = await fetch(`${API_BASE_URL}/terrain/pharmacies?lat=${lat}&lon=${lon}`);
  if (!response.ok) throw new Error(`Terrain pharmacies failed with status ${response.status}`);
  return response.json();
}

export async function fetchTerrainInformalSignals(): Promise<TerrainInformalSignals> {
  const response = await fetch(`${API_BASE_URL}/terrain/informal-signals`);
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
  const response = await fetch(`${API_BASE_URL}/terrain/climate?lat=${lat}&lon=${lon}`);
  if (!response.ok) throw new Error(`Terrain climate failed with status ${response.status}`);
  return response.json();
}

// ─── Demand Forecasting Model (Scénario 1) ───────────────────────────────────

export interface DemandForecastPrediction {
  date: string;
  ds: string;
  demand: number;
  temp_estimated: number;
  risk_level: "NORMAL" | "ÉLEVÉ" | "CRITIQUE";
  color: "green" | "orange" | "red";
  recommendation: string;
}

export interface DemandForecastResponse {
  status: "success" | "fallback";
  model: string;
  last_trained: string;
  input_features: {
    current_temperature: number;
    epidemic_index: number;
    geographical_scope: string;
  };
  predictions: DemandForecastPrediction[];
  error?: string;
}

export async function fetchDemandForecast(lat = 46.2044, lon = 6.1432, region = "Auvergne-Rhône-Alpes"): Promise<DemandForecastResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/demand-forecasting?lat=${lat}&lon=${lon}&region=${encodeURIComponent(region)}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Epidemic Early Warning Model ────────────────────────────────────────────
export interface EpidemicDailyPrediction {
  date: string;
  day_label: string;
  incidence_per_100k: number;
  alert_level: "NORMAL" | "VIGILANCE" | "ÉPIDÉMIE";
  ems_impact: string;
}
export interface EpidemicDiseaseResult {
  disease: string;
  label: string;
  current_incidence: number;
  epidemic_threshold: number;
  warning_threshold: number;
  current_alert: "NORMAL" | "VIGILANCE" | "ÉPIDÉMIE";
  max_alert_14d: "NORMAL" | "VIGILANCE" | "ÉPIDÉMIE";
  peak_incidence_14d: number;
  peak_day: number;
  ems_impact: string;
  recommendation: string;
  data_source: string;
  model_used: string;
  daily_predictions: EpidemicDailyPrediction[];
}
export interface EpidemicEarlyWarningResponse {
  model: string;
  status: "live" | "fallback" | "error";
  generated_at: string;
  region: string;
  overall_alert_level: "NORMAL" | "VIGILANCE" | "ÉPIDÉMIE";
  horizon_days: number;
  diseases: Record<string, EpidemicDiseaseResult>;
  most_critical_disease: string;
  global_recommendation: string;
  ecdc_supplement: number | null;
  data_sources: string[];
}
export async function fetchEpidemicEarlyWarning(forceRefresh = false): Promise<EpidemicEarlyWarningResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/epidemic-early-warning?force_refresh=${forceRefresh}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Response Time Optimization Model ────────────────────────────────────────
export interface ResponseTimeAssignment {
  zone_id: string;
  zone_label: string;
  zone_priority: "high" | "medium" | "low";
  base_id: string;
  base_label: string;
  base_country: "CH" | "FR";
  distance_km: number;
  base_travel_time_min: number;
  border_delay_min: number;
  border_crossing: string | null;
  total_response_time_min: number;
  response_status: "OPTIMAL" | "ACCEPTABLE" | "DÉGRADÉ";
  cross_border: boolean;
  recommendation: string;
  route_source: string;
}
export interface ResponseTimeOptimizationResponse {
  model: string;
  status: "live" | "fallback" | "error";
  generated_at: string;
  region: string;
  weather: {
    temperature: number;
    precipitation: number;
    wind_speed: number;
    weather_factor: number;
    weather_description: string;
    source: string;
  };
  metrics: {
    mean_response_time_min: number;
    max_response_time_min: number;
    min_response_time_min: number;
    cross_border_interventions: number;
    degraded_zones: number;
    coverage_rate_pct: number;
  };
  assignments: ResponseTimeAssignment[];
  critical_zones: Array<{ zone_label: string; total_response_time_min: number; recommendation: string }>;
  global_recommendation: string;
  data_sources: string[];
}
export async function fetchResponseTimeOptimization(forceRefresh = false): Promise<ResponseTimeOptimizationResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/response-time-optimization?force_refresh=${forceRefresh}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Cardiac Arrest Prediction Model (OHCA) ──────────────────────────────────
export interface OHCAForecastDay {
  date: string;
  day_name: string;
  temp_max: number;
  temp_min: number;
  season: string;
  risk_multiplier: number;
  risk_pct_above_baseline: number;
  ohca_per_100k_predicted: number;
  ohca_absolute_predicted: number;
  alert_level: "NORMAL" | "VIGILANCE" | "ÉLEVÉ" | "CRITIQUE";
  active_risk_factors: string[];
}
export interface CardiacArrestPredictionResponse {
  model: string;
  status: "live" | "fallback" | "error";
  generated_at: string;
  region: string;
  population_100k: number;
  ohca_baseline_daily_per_100k: number;
  ohca_baseline_annual_per_100k: number;
  flu_epidemic_active: boolean;
  overall_alert_level: "NORMAL" | "VIGILANCE" | "ÉLEVÉ" | "CRITIQUE";
  max_risk_multiplier_3d: number;
  avg_risk_multiplier_7d: number;
  current_weather: {
    temp_max: number;
    temp_min: number;
    temp_mean: number;
    humidity: number;
    wind_speed: number;
    season: string;
    source: string;
  };
  forecast_3d: OHCAForecastDay[];
  recommendations: string[];
  scientific_references: string[];
  data_sources: string[];
}
export async function fetchCardiacArrestPrediction(): Promise<CardiacArrestPredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/cardiac-arrest-prediction`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Heatwave EMS Impact Model (DLNM + UTCI) ─────────────────────────────────
export interface HeatwaveForecastDay {
  date: string;
  temp_max: number;
  temp_min: number;
  utci: number;
  utci_category: string;
  is_heatwave_day: boolean;
  ems_multiplier: number;
  ems_calls_predicted: number;
  ems_excess_calls: number;
  ems_excess_pct: number;
  alert_level: "NORMAL" | "VIGILANCE" | "ALERTE" | "URGENCE";
  pathology_risks: Record<string, { label: string; risk_multiplier: number }>;
}
export interface HeatwaveEMSImpactResponse {
  model: string;
  status: "live" | "fallback" | "error";
  generated_at: string;
  region: string;
  overall_alert_level: "NORMAL" | "VIGILANCE" | "ALERTE" | "URGENCE";
  ems_baseline_daily: number;
  current_weather: {
    temp_max: number;
    temp_min: number;
    apparent_temp_max: number;
    humidity: number;
    wind_speed_ms: number;
    utci: number;
    utci_category: string;
    source: string;
  };
  heatwave_status: {
    active: boolean;
    duration_days: number;
    severity: string;
    start_date: string | null;
    consecutive_hw_days: number;
    threshold_tmax: number;
    threshold_tmin: number;
  };
  dlnm_analysis: {
    multiplier: number;
    ems_calls_today: number;
    excess_calls_today: number;
    excess_pct_today: number;
    lag_contributions: Array<{
      lag: number;
      date: string;
      utci: number;
      utci_category: string;
      ems_impact_factor: number;
      weight: number;
      contribution: number;
    }>;
  };
  forecast_7d: HeatwaveForecastDay[];
  max_ems_multiplier_7d: number;
  recommendations: string[];
  scientific_references: string[];
  data_sources: string[];
}
export async function fetchHeatwaveEMSImpact(): Promise<HeatwaveEMSImpactResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/heatwave-ems-impact`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Stroke Detection ─────────────────────────────────────────────────────────
export interface StrokeDetectionResponse {
  model: string;
  status: string;
  generated_at: string;
  region: string;
  overall_alert_level: string;
  circadian_risk: { hour: number; risk_level: string; risk_factor: number; rationale: string };
  stroke_units: Array<{
    name: string; city: string; country: string;
    estimated_dtn_min: number; dtn_target_min: number; dtn_ok: boolean;
    tpa_eligible: boolean; thrombectomy_eligible: boolean;
    distance_km: number; transport_time_min: number;
  }>;
  therapeutic_windows: Record<string, { label: string; window_hours: number; evidence: string }>;
  recommendations: string[];
  scientific_references: string[];
}
export async function fetchStrokeDetection(): Promise<StrokeDetectionResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/stroke-detection`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Triage Support ───────────────────────────────────────────────────────────
export interface TriageSupportResponse {
  model: string;
  status: string;
  generated_at: string;
  region: string;
  overall_alert_level: string;
  ccmu_levels: Record<string, { label: string; description: string; target_time_min: number; color: string }>;
  current_load: { level: string; label: string; waiting_patients: number; mean_wait_min: number };
  red_flags: Record<string, { category: string; flags: string[] }>;
  news2_thresholds: Record<string, { label: string; score_range: string; action: string; color: string }>;
  recommendations: string[];
  scientific_references: string[];
}
export async function fetchTriageSupport(): Promise<TriageSupportResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/triage-support`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Undertriage Risk ─────────────────────────────────────────────────────────
export interface UndertriageRiskResponse {
  model: string;
  status: string;
  generated_at: string;
  region: string;
  overall_alert_level: string;
  undertriage_rate_target_pct: number;
  high_risk_scenarios: Array<{
    scenario: string; risk_factors: string[];
    undertriage_risk_pct: number; risk_level: string;
    recommended_action: string;
  }>;
  recommendations: string[];
  scientific_references: string[];
}
export async function fetchUndertriageRisk(): Promise<UndertriageRiskResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/undertriage-risk`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Trauma Care ──────────────────────────────────────────────────────────────
export interface TraumaCareResponse {
  model: string;
  status: string;
  generated_at: string;
  region: string;
  overall_alert_level: string;
  case_examples: Array<{
    case_name: string; mechanism: string; age: number;
    scores: { rts: number; iss: number; iss_level: string; triss_survival_pct: number; predicted_mortality_pct: number };
    damage_control_indicated: boolean;
    damage_control_triggers: string[];
    recommendations: string[];
  }>;
  cohort_summary: { n_cases: number; mean_survival_pct: number; damage_control_cases: number; damage_control_rate_pct: number };
  transfusion_protocol: Record<string, { label: string; evidence: string }>;
  recommendations: string[];
  scientific_references: string[];
}
export async function fetchTraumaCare(): Promise<TraumaCareResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/trauma-care`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Mass Casualty ────────────────────────────────────────────────────────────
export interface MassCasualtyResponse {
  model: string;
  status: string;
  generated_at: string;
  region: string;
  overall_alert_level: string;
  scenario: { n_victims: number; event_type: string; event_label: string; contamination_risk: boolean; blast_injuries: boolean };
  salt_distribution: Record<string, { mean: number; median: number; ci95_low: number; ci95_high: number; label: string; color: string }>;
  resource_needs: {
    transport: { smur_needed: number; amu_needed: number; deficit_smur: number; deficit_amu: number };
    personnel: { doctors_needed: number; nurses_needed: number };
    hospital_capacity: { icu_beds_needed: number; surgery_rooms_needed: number; total_hospital_capacity_available: number };
    mutual_aid_required: boolean;
  };
  hospital_distribution: Array<{ hospital: string; city: string; country: string; assigned_immediate: number; assigned_delayed: number; total_assigned: number; transport_time_min: number }>;
  activation_checklist: Array<{ step: number | string; action: string; responsible: string; time_target: string }>;
  mci_types: Record<string, { label: string; contamination_risk: boolean; blast_injuries: boolean }>;
  recommendations: string[];
  scientific_references: string[];
}
export async function fetchMassCasualty(nVictims = 50, eventType = "transport_accident"): Promise<MassCasualtyResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/mass-casualty?n_victims=${nVictims}&event_type=${eventType}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Clinical Deterioration ───────────────────────────────────────────────────
export interface ClinicalDeteriorationResponse {
  model: string; status: string; generated_at: string;
  overall_alert: string; overall_color: string;
  news2_score: number; mews_score: number;
  vital_signs: Record<string, { value: number; unit: string; status: string }>;
  alerts: Array<{ score: string; value: number; threshold: number; level: string; recommendation: string }>;
  recommendations: string[];
}
export async function fetchClinicalDeterioration(): Promise<ClinicalDeteriorationResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/clinical-deterioration-prediction`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Emergency Call Qualification ────────────────────────────────────────────
export interface CallQualificationResponse {
  model: string; status: string; generated_at: string;
  overall_priority: number; overall_label: string; overall_color: string;
  calls_analyzed: number;
  sample_calls: Array<{ call_id: string; chief_complaint: string; priority: number; priority_label: string; recommended_resource: string; confidence_pct: number }>;
  resource_summary: Record<string, number>;
}
export async function fetchCallQualification(): Promise<CallQualificationResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/emergency-call-qualification`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Dispatch Decision Support ────────────────────────────────────────────────
export interface DispatchDecisionResponse {
  model: string; status: string; generated_at: string;
  overall_status: string; overall_color: string;
  pending_incidents: number; available_resources: number;
  dispatch_recommendations: Array<{ incident_id: string; priority: number; category: string; recommended_resource: string; eta_min: number; rationale: string }>;
  resource_status: Record<string, { available: number; deployed: number; total: number }>;
}
export async function fetchDispatchDecision(): Promise<DispatchDecisionResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/dispatch-decision-support`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Patient Pathway Optimization ────────────────────────────────────────────
export interface PatientPathwayResponse {
  model: string; status: string; generated_at: string;
  cases: Array<{ case_id: string; condition: string; recommended: string; eta_min: number; rationale: string; cross_border: boolean }>;
  summary: { total_cases: number; cross_border_cases: number; mean_eta_min: number };
}
export async function fetchPatientPathway(): Promise<PatientPathwayResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/patient-pathway-optimization`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Ambulance Dispatch Optimization ─────────────────────────────────────────
export interface AmbulanceDispatchResponse {
  model: string; status: string; generated_at: string;
  coverage: { coverage_pct: number; covered_population: number; total_population: number; uncovered_zones: number; degraded_zones: number };
  zone_details: Array<{ zone_id: string; zone_name: string; covered: boolean; best_base: string; eta_min: number; redundancy: number }>;
  recommendations: Array<{ type: string; zone: string; action: string; priority: string }>;
  total_units_deployed: number;
}
export async function fetchAmbulanceDispatch(): Promise<AmbulanceDispatchResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/ambulance-dispatch-optimization`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Hospital Capacity & Staffing ────────────────────────────────────────────
export interface HospitalCapacityResponse {
  model: string; status: string; generated_at: string;
  current_status: { ed_occupancy_pct: number; icu_occupancy_pct: number; nedocs_score: number; nedocs_level: string; nedocs_color: string; hospital_status: string; hospital_color: string };
  staffing_now: { required_crews: number; current_crews: number; delta: number; status: string; color: string; action: string };
  peak_demand: { hour: string; calls_per_hour: number };
  max_staffing_deficit: { hour: string; required: number; current: number; deficit: number };
}
export async function fetchHospitalCapacity(): Promise<HospitalCapacityResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/hospital-capacity-forecasting`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Surveillance ─────────────────────────────────────────────────────────────
export interface SurveillanceResponse {
  model: string; status: string; generated_at: string;
  overall_status: string; overall_color: string;
  indicators: Record<string, { anomaly: boolean; zscore: number; current: number; mean: number }>;
  active_alerts: Array<{ indicator: string; zscore: number; message: string; severity: string }>;
}
export async function fetchSurveillance(): Promise<SurveillanceResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/surveillance`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Surge Management ────────────────────────────────────────────────────────
export interface SurgeManagementResponse {
  model: string; status: string; generated_at: string;
  queue_metrics: { arrival_rate_per_hour: number; utilization_pct: number; prob_waiting_pct: number; mean_wait_min: number };
  surge_status: string; surge_color: string;
  staffing: { available_crews: number; required_crews: number; additional_needed: number };
}
export async function fetchSurgeManagement(): Promise<SurgeManagementResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/surge-management`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Resource Allocation ──────────────────────────────────────────────────────
export interface ResourceAllocationResponse {
  model: string; status: string; generated_at: string;
  summary: { total_incidents: number; allocated: number; unmet: number };
  allocations: Array<{ incident_id: string; priority: number; category: string; allocated: string; status: string }>;
  unmet_incidents: Array<{ incident_id: string; priority: number; category: string; status: string }>;
  remaining_resources: Record<string, number>;
}
export async function fetchResourceAllocation(): Promise<ResourceAllocationResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/resource-allocation`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Environmental Risk ───────────────────────────────────────────────────────
export interface EnvironmentalRiskResponse {
  model: string; status: string; generated_at: string;
  air_quality: { pm2_5_ugm3: number; ozone_ugm3: number; no2_ugm3: number; iqa_level: string; iqa_color: string; source: string };
  ems_impact: { estimated_call_increase_pct: number; risk_level: string; risk_color: string };
  recommendations: string[];
}
export async function fetchEnvironmentalRisk(): Promise<EnvironmentalRiskResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/environmental-risk-forecasting`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Pandemic Preparedness ────────────────────────────────────────────────────
export interface PandemicPreparednessResponse {
  model: string; status: string; generated_at: string;
  parameters: { R0: number; population: number; current_infected: number };
  "30d_forecast": { peak_infected: number; peak_day: number; peak_icu_required: number; icu_capacity: number; total_cases: number };
  preparedness_assessment: string; preparedness_color: string;
}
export async function fetchPandemicPreparedness(): Promise<PandemicPreparednessResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/pandemic-preparedness`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Cross-Border Coordination ────────────────────────────────────────────────
export interface CrossBorderResponse {
  model: string; status: string; generated_at: string;
  coordination_status: string; coordination_color: string;
  active_agreements: number; total_daily_capacity: number; pending_incidents: number;
  available_resources: { CH: number; FR: number };
  agreements: Array<{ id: string; name: string; type: string; active: boolean; legal_basis: string }>;
}
export async function fetchCrossBorder(): Promise<CrossBorderResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/cross-border-coordination`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Situational Awareness ────────────────────────────────────────────────────
export interface SituationalAwarenessResponse {
  model: string; status: string; generated_at: string;
  overall_status: string; overall_color: string;
  real_time_indicators: { active_incidents: number; available_ems_crews: number; ed_occupancy_pct: number; pending_calls_in_queue: number; cross_border_active: number; weather_risk: string };
}
export async function fetchSituationalAwareness(): Promise<SituationalAwarenessResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/situational-awareness`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── Disaster Risk Assessment ─────────────────────────────────────────────────
export interface DisasterRiskResponse {
  model: string; status: string; generated_at: string;
  top_risk: { type: string; zone: string; probability_annual: number; severity: number; risk_score: number; risk_level: string };
  all_risks: Array<{ type: string; zone: string; probability_annual: number; severity: number; population_at_risk: number; risk_score: number; risk_level: string }>;
  overall_risk_level: string;
}
export async function fetchDisasterRisk(): Promise<DisasterRiskResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/disaster-risk-assessment`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ─── MCI Victim Estimation ────────────────────────────────────────────────────
export interface MCIVictimResponse {
  model: string; status: string; generated_at: string;
  incident: { type: string; vehicles: number; location: string; time: string };
  estimated_victims: number;
  triage_distribution: { T1_critical: number; T2_serious: number; T3_minor: number };
  recommended_resources: { SMUR: number; AMBULANCE: number; MÉDECINS: number };
}
export async function fetchMCIVictim(): Promise<MCIVictimResponse> {
  const response = await fetch(`${API_BASE_URL}/gesica/model/mci-victim-estimation`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}
