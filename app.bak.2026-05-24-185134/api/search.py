from fastapi import APIRouter, Depends

from app.deps import get_search_service
from app.schemas import SearchIn
from app.services.search_service import SearchService

router = APIRouter()


@router.post("/search")
def search_chunks(payload: SearchIn, service: SearchService = Depends(get_search_service)):
    return {
        "results": service.search(
            query_text=payload.query_text,
            limit=payload.limit,
            mode=payload.mode,
            filters=payload.filters,
        )
    }
