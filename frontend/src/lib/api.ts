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
