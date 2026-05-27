import type {
  SearchRequest,
  SearchResponse,
  SearchResult,
} from "../types/search";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

export interface FilterOption {
  value: string | number;
  label: string;
}

export interface FilterOptions {
  source?: FilterOption[];
  sourcetype?: FilterOption[];
  diseaseorcondition?: FilterOption[];
  scenariotype?: FilterOption[];
  geographicscope?: FilterOption[];
  evidencecategory?: FilterOption[];
  year?: FilterOption[];
}

export interface DocumentChunk {
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

export async function searchDocuments(
  payload: SearchRequest,
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Search failed with status ${response.status}`);
  }

  return response.json();
}

export async function getFilterOptions(): Promise<FilterOptions> {
  const response = await fetch(`${API_BASE_URL}/filters-options`);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(
      text || `Filter options failed with status ${response.status}`,
    );
  }

  return response.json();
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

  return response.json();
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