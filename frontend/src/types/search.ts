export type SearchMode = "semantic" | "boolean";
export type ProjectContext = "literev";
export type RelevanceLabel = "pertinent" | "non-pertinent" | "incertain";

// --- FRONTEND UI DOMAIN TYPES (camelCase) ---

export interface SearchFilters {
  projectContext?: string;
  sourceType?: string;
  diseaseOrCondition?: string;
  scenarioType?: string;
  geographicScope?: string;
  evidenceCategory?: string;
  yearMin?: number;
  yearMax?: number;
}

export interface SearchResult {
  id: string; // e.g., "12-3"
  documentId: number;
  chunkId?: number;
  chunkIndex?: number;
  content: string;
  score?: number;
  title: string;
  abstract?: string | null;
  source?: string | null;
  year?: number | null;
  url?: string | null;
  projectContext?: string | null;
  sourceType?: string | null;
  diseaseOrCondition?: string | null;
  scenarioType?: string | null;
  geographicScope?: string | null;
  evidenceCategory?: string | null;
  highlight?: string | null;
  chunkType?: string | null;
  [key: string]: unknown;
}

export interface SearchRequest {
  queryText: string;
  mode: SearchMode;
  limit: number;
  filters?: SearchFilters;
}

export interface SearchResponse {
  results: SearchResult[];
  totalUniqueDocs?: number;
  sourceBreakdown?: Record<string, number>;
}

// --- BACKEND API RAW TYPES (snake_case) ---

export interface ApiSearchFilters {
  project_context?: string;
  source_type?: string;
  disease_or_condition?: string;
  scenario_type?: string;
  geographic_scope?: string;
  evidence_category?: string;
  year_min?: number;
  year_max?: number;
}

export interface ApiSearchResult {
  id: string;
  document_id: number;
  chunk_id: number;
  chunk_index: number;
  title: string;
  abstract: string | null;
  content: string;
  highlight: string | null;
  score: number;
  source: string | null;
  year: number | null;
  url: string | null;
  project_context: string | null;
  source_type: string | null;
  disease_or_condition: string | null;
  scenario_type: string | null;
  geographic_scope: string | null;
  evidence_category: string | null;
  chunk_type: string | null;
}

export interface ApiSearchRequest {
  query_text: string;
  mode: SearchMode;
  limit: number;
  filters?: ApiSearchFilters;
}

export interface ApiSearchResponse {
  results: ApiSearchResult[];
  count: number;
  total_unique_docs?: number;
  source_breakdown?: Record<string, number>;
}
