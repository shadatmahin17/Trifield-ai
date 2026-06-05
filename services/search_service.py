import httpx
import asyncio
import hashlib
import time
from models.schemas import Paper, Author

# ── Cache ──────────────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}
CACHE_TTL = 3600

DISCIPLINE_KEYWORDS = {
    "aerospace": [
        "aerospace", "aeronautics", "aircraft", "spacecraft", "laminate",
        "carbon fibre", "carbon fiber", "structural composite", "sandwich structure",
        "fatigue", "damage tolerance", "airframe", "aerostructure",
    ],
    "materials": [
        "composite", "hybrid composite", "nanocomposite", "polymer matrix",
        "epoxy", "resin", "fibre reinforced", "fiber reinforced",
        "mechanical properties", "tensile strength", "flexural", "void content",
        "fibre volume fraction", "delamination", "fracture",
    ],
    "textile": [
        "textile", "woven", "woven fabric", "weave", "yarn", "braided",
        "knitted", "nonwoven", "jute", "flax", "hemp", "natural fibre",
        "natural fiber", "technical textile", "preform", "fabric structure",
    ],
}

DISCIPLINE_CONCEPTS = {
    "aerospace": "C27206212",
    "materials": "C192562407",
    "textile":   "C107038049",
}


def tag_discipline(title: str, abstract: str | None, concepts: list) -> str:
    concept_names = " ".join(c.get("display_name", "").lower() for c in concepts)
    text = (title + " " + (abstract or "") + " " + concept_names).lower()
    scores = {d: sum(1 for kw in kws if kw in text)
              for d, kws in DISCIPLINE_KEYWORDS.items()}
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "general"


def _cache_key(query, discipline, year_from, year_to, limit) -> str:
    return hashlib.md5(f"{query}|{discipline}|{year_from}|{year_to}|{limit}".encode()).hexdigest()


def _get_cached(key):
    entry = _cache.get(key)
    return entry["papers"] if entry and time.time() < entry["expires"] else None


def _set_cache(key, papers):
    _cache[key] = {"papers": papers, "expires": time.time() + CACHE_TTL}


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return None
    try:
        positions = {}
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions[pos] = word
        if not positions:
            return None
        return " ".join(positions[i] for i in sorted(positions.keys()))
    except Exception:
        return None


async def _fetch_openalex(params: dict, max_retries: int = 3) -> dict:
    url = "https://api.openalex.org/works"
    params["mailto"] = "contact@trifield.ai"
    delays = [1, 3, 6]

    async with httpx.AsyncClient(timeout=20) as client:
        for attempt in range(max_retries):
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    await asyncio.sleep(delays[attempt])
                    continue
                raise Exception("Rate limit reached. Please wait a moment and retry.")
            resp.raise_for_status()

    return {"results": [], "meta": {"count": 0}}


async def _get_oa_url_unpaywall(doi: str) -> str | None:
    """Fallback: check Unpaywall for free PDF URL."""
    if not doi:
        return None
    # Extract just the DOI part if it's a full URL
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    url = f"https://api.unpaywall.org/v2/{doi_clean}?email=contact@trifield.ai"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                # Best OA location
                best = data.get("best_oa_location") or {}
                return best.get("url_for_pdf") or best.get("url")
    except Exception:
        pass
    return None


async def search_papers(
    query: str,
    discipline: str = "all",
    year_from: int | None = None,
    year_to:   int | None = None,
    limit:     int = 10,
) -> list[Paper]:

    # Cache check
    key = _cache_key(query, discipline, year_from, year_to, limit)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    # Build params — include abstract_inverted_index explicitly
    params: dict = {
        "search":   query,
        "per_page": min(limit, 50),
        "select": (
            "id,title,authorships,publication_year,"
            "abstract_inverted_index,"           # ← key fix for abstracts
            "cited_by_count,doi,"
            "primary_location,open_access,concepts"
        ),
        "sort": "relevance_score:desc",
    }

    # Discipline concept filter
    filters = []
    if discipline != "all" and discipline in DISCIPLINE_CONCEPTS:
        filters.append(f"concepts.id:{DISCIPLINE_CONCEPTS[discipline]}")

    # Year filters
    if year_from:
        filters.append(f"publication_year:>{year_from - 1}")
    if year_to:
        filters.append(f"publication_year:<{year_to + 1}")

    if filters:
        params["filter"] = ",".join(filters)

    # Fetch from OpenAlex
    data = await _fetch_openalex(params)

    # Parse results
    papers = []
    for item in data.get("results", []):

        abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))
        authors  = [
            Author(name=a.get("author", {}).get("display_name", "Unknown"))
            for a in item.get("authorships", [])[:10]
        ]

        primary  = item.get("primary_location") or {}
        source   = primary.get("source") or {}
        journal  = source.get("display_name")

        # Open access URL — use OpenAlex first, Unpaywall as fallback
        oa       = item.get("open_access") or {}
        oa_url   = oa.get("oa_url")

        doi      = item.get("doi")  # full URL e.g. https://doi.org/10.xxxx
        paper_url = doi or f"https://openalex.org/{item.get('id','').split('/')[-1]}"
        concepts  = item.get("concepts") or []

        # Unpaywall fallback for open access (only if no OA url found)
        if not oa_url and doi:
            oa_url = await _get_oa_url_unpaywall(doi)

        papers.append(Paper(
            paper_id        = item.get("id", "").split("/")[-1],
            title           = item.get("title") or "Untitled",
            authors         = authors,
            year            = item.get("publication_year"),
            abstract        = abstract,
            citation_count  = item.get("cited_by_count", 0),
            url             = paper_url,
            open_access_url = oa_url,
            journal         = journal,
            discipline_tag  = tag_discipline(item.get("title", ""), abstract, concepts),
        ))

    if discipline != "all":
        papers.sort(key=lambda p: 0 if p.discipline_tag == discipline else 1)

    _set_cache(key, papers)
    return papers
