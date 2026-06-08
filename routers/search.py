from fastapi import APIRouter, Query, HTTPException
from services.search_service import search_papers
from services.query_intelligence import analyse_query
from models.schemas import SearchResponse

router = APIRouter()

@router.get("/", response_model=SearchResponse)
async def search(
    query:      str = Query(..., description="Natural language query — typos, abbreviations, and synonyms handled automatically"),
    discipline: str = Query("all", description="all | aerospace | materials | textile"),
    year_from:  int = Query(None, description="Filter from year"),
    year_to:    int = Query(None, description="Filter to year"),
    limit:      int = Query(10, description="Number of results (max 50)", le=50, ge=1),
):
    """
    Intelligent paper search across OpenAlex, Crossref, arXiv, PubMed, and Unpaywall.

    Features:
    - Understands intent, not just exact keywords
    - Corrects typos automatically (e.g. 'tenslie' → 'tensile')
    - Expands abbreviations (e.g. 'CFRP' → 'carbon fibre reinforced polymer')
    - Expands synonyms (e.g. 'jute' → adds 'bast fibre', 'corchorus')
    - Detects discipline automatically if not specified
    - Reranks results by relevance, entity match, and recency

    Examples:
      /api/search/?query=jute flax hybrid composit mechanicl properteis
      /api/search/?query=CFRP fatigue damage tolerance aerospace
      /api/search/?query=fem simulation 3d woven unit cell model
    """
    try:
        papers = await search_papers(
            query=query,
            discipline=discipline,
            year_from=year_from,
            year_to=year_to,
            limit=limit,
        )

        # Include query intelligence info in response
        analysis = analyse_query(query, discipline)

        return SearchResponse(
            query=query,
            interpreted_query=analysis.primary_query,
            intent=analysis.intent,
            detected_discipline=analysis.discipline,
            total=len(papers),
            discipline=discipline,
            papers=papers,
        )
    except Exception as e:
        msg = str(e)
        if "rate limit" in msg.lower() or "429" in msg:
            raise HTTPException(
                status_code=429,
                detail="Rate limit reached. Wait 30 seconds and retry. Cached searches are instant."
            )
        raise HTTPException(status_code=500, detail=msg)
