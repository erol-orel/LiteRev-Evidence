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
  recommendedActions: string[];
  relevantArticles: Array<{
    id: number;
    title: string;
    abstract: string | null;
    year: number | null;
    source: string;
  }>;
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
  identification: {
    totalRecords: number;
    bySource: Record<string, number>;
    duplicatesRemoved: number;
  };
  screening: {
    recordsScreened: number;
    recordsExcluded: number;
  };
  eligibility: {
    fulltextAssessed: number;
    fulltextExcluded: number;
    reasons: Record<string, number>;
  };
  included: {
    totalIncluded: number;
    pendingAssessment: number;
  };
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
    recommended_actions: string[];
    relevant_articles: Array<{ id: number; title: string; abstract: string | null; year: number | null; source: string }>;
  }> = await response.json();
  return data.map((s) => ({
    id: s.id,
    title: s.title,
    description: s.description,
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
    projectContext: d.project_context,
    screeningStatus: d.screening_status,
    screeningReason: d.screening_reason,
    screeningNotes: d.screening_notes
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
  return {
    identification: {
      totalRecords: d.identification.total_records,
      bySource: d.identification.by_source,
      duplicatesRemoved: d.identification.duplicates_removed
    },
    screening: {
      recordsScreened: d.screening.records_screened,
      recordsExcluded: d.screening.records_excluded
    },
    eligibility: {
      fulltextAssessed: d.eligibility.fulltext_assessed,
      fulltextExcluded: d.eligibility.fulltext_excluded,
      reasons: d.eligibility.reasons
    },
    included: {
      totalIncluded: d.included.total_included,
      pendingAssessment: d.included.pending_assessment
    }
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
