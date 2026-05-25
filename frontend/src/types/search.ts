export type SearchMode = 'semantic' | 'boolean'
export type ProjectContext = 'geoai4ei' | 'gesica' | 'eva'
export type RelevanceLabel = 'pertinent' | 'non_pertinent' | 'incertain'

export interface SearchFilters {
  project_context?: string
  source_type?: string
  disease_or_condition?: string
  scenario_type?: string
  geographic_scope?: string
  evidence_category?: string
  year_min?: number
  year_max?: number
}

export interface SearchResult {
  id: number
  document_id: number
  chunk_index: number
  content: string
  score: number
  title: string
  source: string | null
  year: number | null
  url: string | null
  project_context: string | null
  source_type: string | null
  disease_or_condition: string | null
  scenario_type: string | null
  geographic_scope: string | null
  evidence_category: string | null
  highlight?: string | null
  [key: string]: unknown
}

export interface SearchRequest {
  query_text: string
  mode: SearchMode
  limit: number
  filters?: SearchFilters
}

export interface SearchResponse {
  results: SearchResult[]
}
