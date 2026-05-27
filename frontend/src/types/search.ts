export type SearchMode = "semantic" | "boolean"
export type ProjectContext = "geoai4ei" | "gesica" | "eva"
export type RelevanceLabel = "pertinent" | "non-pertinent" | "incertain"

export interface SearchFilters {
  projectcontext?: string
  sourcetype?: string
  diseaseorcondition?: string
  scenariotype?: string
  geographicscope?: string
  evidencecategory?: string
  yearmin?: number
  yearmax?: number
}

export interface SearchResult {
  id: number
  documentid: number
  chunkindex?: number
  content: string
  score?: number
  title: string
  abstract?: string | null
  source?: string | null
  year?: number | null
  url?: string | null
  projectcontext?: string | null
  sourcetype?: string | null
  diseaseorcondition?: string | null
  scenariotype?: string | null
  geographicscope?: string | null
  evidencecategory?: string | null
  highlight?: string | null
  [key: string]: unknown
}

export interface SearchRequest {
  querytext: string
  mode: SearchMode
  limit: number
  filters?: SearchFilters
}

export interface SearchResponse {
  results: SearchResult[]
}
