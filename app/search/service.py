from __future__ import annotations

from app.search.pgvector_backend import PgvectorSearchBackend
from app.search.elasticsearch_backend import ElasticsearchSearchBackend


class SearchService:
    def __init__(self, db_engine=None, embedder=None, backend_name: str = "pgvector"):
        self.backend_name = backend_name

        if backend_name == "elasticsearch":
            self.backend = ElasticsearchSearchBackend()
        else:
            self.backend = PgvectorSearchBackend(db_engine=db_engine, embedder=embedder)

    def search(self, query: str, filters: dict | None = None, mode: str = "semantic", limit: int = 10):
        return self.backend.search(query=query, filters=filters, mode=mode, limit=limit)

    def hybrid_search(
        self,
        query: str,
        boolean_filters: dict | None = None,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.25,
        limit: int = 10,
    ):
        return self.backend.hybrid_search(
            query=query,
            boolean_filters=boolean_filters,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            limit=limit,
        )

    def get_filter_options(self):
        if hasattr(self.backend, "get_filter_options"):
            return self.backend.get_filter_options()
        return {
            "source": [],
            "source_type": [],
            "disease_or_condition": [],
            "scenario_type": [],
            "geographic_scope": [],
            "evidence_category": [],
            "year": [],
        }
