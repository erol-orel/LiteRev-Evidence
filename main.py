from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from app.schemas.api import ChunkIn, DocumentIn, SearchIn
from app.search.service import SearchService

logger = logging.getLogger("literev-api")
logging.basicConfig(level=logging.INFO)

DB_URL = os.getenv("DB_URL")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
WRITE_API_KEY = os.getenv("WRITE_API_KEY", "")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://127.0.0.1",
).split(",")

app = FastAPI(title="LiteRev API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(DB_URL, pool_pre_ping=True)
embedder = SentenceTransformer(EMBED_MODEL_NAME)
search_service = SearchService(db_engine=engine, embedder=embedder)

logger.info("Embedding model loaded: %s", EMBED_MODEL_NAME)
logger.info("Search backend selected: %s", search_service.backend_name)


# ── API Key guard (write endpoints only) ──────────────────────────────────────
def require_api_key(x_api_key: str = Header(default="")) -> None:
    if WRITE_API_KEY and x_api_key != WRITE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.on_event("startup")
def startup_event():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection OK")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/embed-info")
def embed_info() -> dict[str, Any]:
    dim = embedder.get_sentence_embedding_dimension()
    return {"model": EMBED_MODEL_NAME, "dimension": dim}


@app.get("/filters/options")
def filter_options() -> dict[str, Any]:
    return search_service.get_filter_options()


@app.post("/documents")
def create_document(doc: DocumentIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    sql = text("""
        INSERT INTO literature_document (
            source, title, abstract, year, url, external_id,
            project_context, source_type, disease_or_condition,
            scenario_type, geographic_scope, evidence_category
        )
        VALUES (
            :source, :title, :abstract, :year, :url, :external_id,
            :project_context, :source_type, :disease_or_condition,
            :scenario_type, :geographic_scope, :evidence_category
        )
        RETURNING id
    """)

    with engine.begin() as conn:
        row = conn.execute(
            sql,
            {
                "source": doc.source,
                "title": doc.title,
                "abstract": doc.abstract,
                "year": doc.year,
                "url": doc.url,
                "external_id": doc.external_id,
                "project_context": doc.project_context,
                "source_type": doc.source_type,
                "disease_or_condition": doc.disease_or_condition,
                "scenario_type": doc.scenario_type,
                "geographic_scope": doc.geographic_scope,
                "evidence_category": doc.evidence_category,
            },
        ).first()

    return {"id": row.id, "status": "ok"}


@app.post("/chunks")
def create_chunk(chunk: ChunkIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    embedding = embedder.encode(chunk.content).tolist()

    sql = text("""
        INSERT INTO document_chunk (
            document_id,
            chunk_index,
            content,
            embedding,
            search_vector,
            chunk_type,
            section_label,
            char_start,
            char_end,
            token_count,
            chunk_weight,
            metadata_json
        )
        VALUES (
            :document_id,
            :chunk_index,
            :content,
            CAST(:embedding AS vector),
            to_tsvector('simple', :content),
            :chunk_type,
            :section_label,
            :char_start,
            :char_end,
            :token_count,
            :chunk_weight,
            CAST(:metadata_json AS jsonb)
        )
        RETURNING id
    """)

    with engine.begin() as conn:
        row = conn.execute(
            sql,
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "embedding": str(embedding),
                "chunk_type": chunk.chunk_type,
                "section_label": chunk.section_label,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_count": chunk.token_count,
                "chunk_weight": chunk.chunk_weight,
                "metadata_json": json.dumps(chunk.metadata_json or {}),
            },
        ).first()

    return {"id": row.id, "status": "ok"}


@app.post("/search")
def search(payload: SearchIn) -> dict[str, Any]:
    mode = (payload.mode or "semantic").lower()

    if mode == "hybrid":
        results = search_service.hybrid_search(
            query=payload.query_text,
            boolean_filters=payload.filters,
            vector_weight=0.65,
            bm25_weight=0.25,
            limit=payload.limit,
        )
    else:
        results = search_service.search(
            query=payload.query_text,
            filters=payload.filters,
            mode=mode,
            limit=payload.limit,
        )

    return {"results": results}


@app.get("/documents/{doc_id}")
def get_document(doc_id: int) -> dict[str, Any]:
    sql = text("""
        SELECT
            d.id, d.source, d.title, d.abstract, d.year, d.url,
            d.external_id, d.project_context, d.source_type,
            d.disease_or_condition, d.scenario_type,
            d.geographic_scope, d.evidence_category,
            d.created_at, d.updated_at,
            COALESCE(
                json_agg(
                    json_build_object(
                        'id', c.id,
                        'chunk_index', c.chunk_index,
                        'content', c.content,
                        'chunk_type', c.chunk_type,
                        'section_label', c.section_label,
                        'token_count', c.token_count
                    ) ORDER BY c.chunk_index
                ) FILTER (WHERE c.id IS NOT NULL),
                '[]'
            ) AS chunks
        FROM literature_document d
        LEFT JOIN document_chunk c ON c.document_id = d.id
        WHERE d.id = :doc_id
        GROUP BY d.id
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"doc_id": doc_id}).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return dict(row._mapping)
