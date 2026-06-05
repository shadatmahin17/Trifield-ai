from fastapi import APIRouter, Query, HTTPException
from services.search_service import search_papers
from models.schemas import SearchResponse

router = APIRouter()

@router.get("/", response_model=SearchResponse)
async def search(
    query:      str = Query(..., description="Search query e.g. 'carbon fibre composite fatigue'"),
    discipline: str = Query("all", description="all | aerospace | materials | textile"),
    year_from:  int = Query(None, description="Filter from year e.g. 2015"),
    year_to:    int = Query(None, description="Filter to year e.g. 2024"),
    limit:      int = Query(10,   description="Number of results (max 50)", le=50, ge=1),
):
    """
    Search research papers via Semantic Scholar.
    Works directly in browser — no API key needed.

    Example:
      GET /api/search/?query=jute+flax+hybrid+composite&discipline=textile&limit=5
    """
    try:
        papers = await search_papers(
            query=query,
            discipline=discipline,
            year_from=year_from,
            year_to=year_to,
            limit=limit,
        )
        return SearchResponse(
            query=query,
            total=len(papers),
            discipline=discipline,
            papers=papers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
