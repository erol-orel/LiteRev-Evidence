from __future__ import annotations

from typing import Any
from pydantic import BaseModel

class SearchIn(BaseModel):
    query_text: str
    mode: str = "semantic"
    limit: int = 5
    filters: dict[str, Any] | None = None
