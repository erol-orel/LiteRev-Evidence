from __future__ import annotations

from .base import SearchBackend


class ElasticsearchSearchBackend(SearchBackend):
    def __init__(self, client=None):
        self.client = client

    def search(self, query: str, filters=None, mode: str = "semantic", limit: int = 10):
        raise NotImplementedError("Elasticsearch backend prepared but not active in Phase 2.")

    def hybrid_search(self, query: str, boolean_filters=None, vector_weight: float = 0.7, bm25_weight: float = 0.3, limit: int = 10):
        raise NotImplementedError("Elasticsearch backend prepared but not active in Phase 2.")
