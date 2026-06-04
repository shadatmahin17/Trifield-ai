from fastapi import APIRouter, HTTPException

from core.config import get_settings
from models.schemas import SearchRequest, SearchResponse
from services.search_service import search_papers

router = APIRouter()


@router.post("/", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    settings = get_settings()
    limit = min(request.limit or settings.max_search_results, settings.max_search_results)

    try:
        papers = await search_papers(
            query=request.query,
            discipline=request.discipline or "all",
            year_from=request.year_from,
            year_to=request.year_to,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Paper search failed: {exc}") from exc

    return SearchResponse(
        query=request.query,
        total=len(papers),
        discipline=request.discipline or "all",
        papers=papers,
    )
