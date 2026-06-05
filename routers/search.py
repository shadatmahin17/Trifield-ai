from fastapi import APIRouter, Query, HTTPException
from services.search_service import search_papers
from models.schemas import SearchResponse

router = APIRouter()

@router.get("/", response_model=SearchResponse)
async def search(
    query:      str = Query(..., description="e.g. 'carbon fibre composite fatigue'"),
    discipline: str = Query("all", description="all | aerospace | materials | textile"),
    year_from:  int = Query(None, description="Filter from year e.g. 2015"),
    year_to:    int = Query(None, description="Filter to year e.g. 2024"),
    limit:      int = Query(10, description="Number of results (max 50)", le=50, ge=1),
):
    """
    Search research papers via Semantic Scholar (free, no API key needed).
    Results are cached for 1 hour — repeated queries are instant.

    Examples:
      /api/search/?query=jute+flax+hybrid+composite&discipline=textile
      /api/search/?query=carbon+fibre+laminate+fatigue&discipline=aerospace&limit=5
      /api/search/?query=3D+woven+composites+damage&discipline=materials&year_from=2018
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
        msg = str(e)
        # Give friendly message for rate limit
        if "rate limit" in msg.lower() or "429" in msg:
            raise HTTPException(
                status_code=429,
                detail="Semantic Scholar rate limit reached. Wait 30 seconds and try again. "
                       "Tip: repeated searches are cached and never hit the rate limit."
            )
        raise HTTPException(status_code=500, detail=msg)
