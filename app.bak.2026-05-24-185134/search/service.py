from __future__ import annotations

import logging
import os

from .elasticsearch_backend import ElasticsearchSearchBackend
from .pgvector_backend import PgvectorSearchBackend

logger = logging.getLogger("literev-api")


class SearchService:
    def __init__(self, db_engine, embedder):
        backend_name = os.getenv("SEARCH_BACKEND", "pgvector").lower()

        if backend_name == "elasticsearch":
            self.backend = ElasticsearchSearchBackend()
        else:
            self.backend = PgvectorSearchBackend(db_engine=db_engine, embedder=embedder)
            backend_name = "pgvector"

        self.backend_name = backend_name
        logger.info("Search backend selected: %s", self.backend_name)

    def search(self, query: str, filters=None, mode: str = "semantic", limit: int = 10):
        return self.backend.search(query=query, filters=filters, mode=mode, limit=limit)

    def hybrid_search(self, query: str, boolean_filters=None, vector_weight: float = 0.7, bm25_weight: float = 0.3, limit: int = 10):
        return self.backend.hybrid_search(
            query=query,
            boolean_filters=boolean_filters,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            limit=limit,
        )
