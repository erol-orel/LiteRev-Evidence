import type { SearchRequest, SearchResponse } from "../types/search"

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api"

export async function searchDocuments(payload: SearchRequest): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Search failed with status ${response.status}`)
  }
  return response.json()
}

export interface FilterOption {
  value: string | number
  label: string
}

export interface FilterOptions {
  source: FilterOption[]
  sourcetype: FilterOption[]
  diseaseorcondition: FilterOption[]
  scenariotype: FilterOption[]
  geographicscope: FilterOption[]
  evidencecategory: FilterOption[]
  year: FilterOption[]
}

export async function getFilterOptions(): Promise<FilterOptions> {
  const response = await fetch(`${API_BASE_URL}/filtersoptions`)
  if (!response.ok) {
    throw new Error(`Filter options failed: ${response.status}`)
  }
  return response.json()
}

export interface DocumentDetailResponse {
  id: number
  source?: string | null
  title?: string | null
  abstract?: string | null
  year?: number | null
  url?: string | null
  externalid?: string | null
  projectcontext?: string | null
  sourcetype?: string | null
  diseaseorcondition?: string | null
  scenariotype?: string | null
  geographicscope?: string | null
  evidencecategory?: string | null
}

export async function getDocumentDetail(documentId: number): Promise<DocumentDetailResponse> {
  const response = await fetch(`${API_BASE_URL}/documents/${documentId}`)
  if (!response.ok) {
    throw new Error(`Document detail failed: ${response.status}`)
  }
  return response.json()
}

export interface EvidenceSummaryResponse {
  doc_id: number
  title?: string
  project_context?: string
  metadata?: Record<string, unknown>
  gesica_extraction?: Record<string, unknown>
}

export async function getEvidenceSummary(documentId: number): Promise<EvidenceSummaryResponse> {
  const response = await fetch(`${API_BASE_URL}/evidence-summary/${documentId}`)
  if (!response.ok) {
    throw new Error(`Evidence summary failed: ${response.status}`)
  }
  return response.json()
}
