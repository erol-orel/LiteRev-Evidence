import os


DB_URL = os.environ.get("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL environment variable is required")

SEARCH_BACKEND = os.environ.get("SEARCH_BACKEND", "pgvector").strip().lower()
MODEL_NAME = os.environ.get("MODEL_NAME", "BAAI/bge-m3")
DEFAULT_SEARCH_LIMIT = int(os.environ.get("DEFAULT_SEARCH_LIMIT", "5"))
MAX_SEARCH_LIMIT = int(os.environ.get("MAX_SEARCH_LIMIT", "50"))
