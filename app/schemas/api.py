from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    source: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    abstract: str | None = None
    year: int | None = None
    url: str | None = None
    external_id: str | None = None
    project_context: str | None = None
    source_type: str | None = None
    disease_or_condition: str | None = None
    scenario_type: str | None = None
    geographic_scope: str | None = None
    evidence_category: str | None = None


class ChunkIn(BaseModel):
    document_id: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)

    chunk_type: str | None = None
    section_label: str | None = None
    char_start: int | None = Field(None, ge=0)
    char_end: int | None = Field(None, ge=0)
    token_count: int | None = Field(None, ge=0)
    chunk_weight: float | None = Field(None, ge=0)
    metadata_json: dict[str, Any] | None = None


class SearchIn(BaseModel):
    query_text: str = Field(..., min_length=1)
    filters: dict[str, Any] | None = None
    mode: str = Field(default="semantic")
    limit: int = Field(default=10, ge=1, le=100)
