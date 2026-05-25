import type { SearchRequest, SearchResponse } from '../types/search'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api'

export async function searchDocuments(payload: SearchRequest): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
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
  source_type: FilterOption[]
  disease_or_condition: FilterOption[]
  scenario_type: FilterOption[]
  geographic_scope: FilterOption[]
  evidence_category: FilterOption[]
  year: FilterOption[]
}

export async function getFilterOptions(): Promise<FilterOptions> {
  const response = await fetch(`${API_BASE_URL}/filters/options`)
  if (!response.ok) throw new Error(`Filter options failed: ${response.status}`)
  return response.json()
}
