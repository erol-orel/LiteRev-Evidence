from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SearchBackend(ABC):
    @abstractmethod
    def search(self, query: str, filters: dict[str, Any] | None = None, mode: str = "semantic", limit: int = 10):
        raise NotImplementedError

    @abstractmethod
    def keyword_search(self, query: str, filters: dict[str, Any] | None = None, limit: int = 10):
        raise NotImplementedError

    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        boolean_filters: dict[str, Any] | None = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        limit: int = 10,
    ):
        raise NotImplementedError
