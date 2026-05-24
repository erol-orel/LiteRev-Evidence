from app.search.backends.pgvector import PgVectorSearchBackend

class SearchService:
    def __init__(self, backend=None):
        self.backend = backend or PgVectorSearchBackend()

    def search(self, query_text: str, limit: int = 5, mode: str = "semantic", filters: dict | None = None):
        return self.backend.search(query_text=query_text, limit=limit, mode=mode, filters=filters)

    def get_filter_options(self):
        return self.backend.get_filter_options()
