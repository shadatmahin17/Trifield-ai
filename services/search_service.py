import httpx
from models.schemas import Paper, Author

# Journals mapped to discipline tags
DISCIPLINE_JOURNALS = {
    "aerospace": [
        "aiaa journal", "journal of aerospace engineering",
        "aerospace science and technology", "acta astronautica",
        "journal of aircraft", "composites part a", "composites part b",
    ],
    "materials": [
        "composites science and technology", "composite structures",
        "journal of composite materials", "carbon",
        "materials & design", "polymer composites",
        "journal of materials science",
    ],
    "textile": [
        "textile research journal", "journal of the textile institute",
        "fibers and polymers", "textile & apparel technology management",
        "international journal of clothing science and technology",
    ],
}

# Discipline keyword boosts for query rewriting
DISCIPLINE_KEYWORDS = {
    "aerospace": "composites aerospace laminate carbon fibre structural",
    "materials":  "composite hybrid matrix fibre reinforced polymer nanocomposite",
    "textile":    "woven fabric yarn fibre textile technical preform braided",
}


def tag_discipline(journal: str | None, title: str) -> str:
    """Best-guess discipline tag from journal name or title keywords."""
    text = ((journal or "") + " " + title).lower()
    for disc, journals in DISCIPLINE_JOURNALS.items():
        if any(j in text for j in journals):
            return disc
    for disc, kws in DISCIPLINE_KEYWORDS.items():
        if any(kw in text for kw in kws.split()):
            return disc
    return "general"


async def search_papers(
    query: str,
    discipline: str = "all",
    year_from: int | None = None,
    year_to:   int | None = None,
    limit:     int = 10,
) -> list[Paper]:
    """
    Query Semantic Scholar public API.
    Docs: https://api.semanticscholar.org/graph/v1
    Free, no API key required for basic usage.
    """

    # Boost query with discipline keywords for better relevance
    boosted_query = query
    if discipline != "all" and discipline in DISCIPLINE_KEYWORDS:
        boosted_query = f"{query} {DISCIPLINE_KEYWORDS[discipline]}"

    params = {
        "query":  boosted_query,
        "limit":  min(limit, 50),
        "fields": "paperId,title,authors,year,abstract,citationCount,"
                  "externalIds,isOpenAccess,openAccessPdf,publicationVenue",
    }
    if year_from:
        params["year"] = f"{year_from}-"
    if year_to and year_from:
        params["year"] = f"{year_from}-{year_to}"
    elif year_to:
        params["year"] = f"-{year_to}"

    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    papers = []
    for item in data.get("data", []):
        venue = item.get("publicationVenue") or {}
        journal_name = venue.get("name")
        oa_pdf = item.get("openAccessPdf") or {}

        # Build authors list
        authors = [Author(name=a.get("name", "")) for a in item.get("authors", [])]

        # External URL (DOI preferred)
        ext_ids = item.get("externalIds") or {}
        doi = ext_ids.get("DOI")
        paper_url = f"https://doi.org/{doi}" if doi else \
                    f"https://www.semanticscholar.org/paper/{item.get('paperId','')}"

        papers.append(Paper(
            paper_id        = item.get("paperId", ""),
            title           = item.get("title", "Untitled"),
            authors         = authors,
            year            = item.get("year"),
            abstract        = item.get("abstract"),
            citation_count  = item.get("citationCount", 0),
            url             = paper_url,
            open_access_url = oa_pdf.get("url"),
            journal         = journal_name,
            discipline_tag  = tag_discipline(journal_name, item.get("title", "")),
        ))

    # If discipline filter requested, prioritise matching papers
    if discipline != "all":
        papers.sort(key=lambda p: 0 if p.discipline_tag == discipline else 1)

    return papers
