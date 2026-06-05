import httpx
import asyncio
import hashlib
import time
from models.schemas import Paper, Author

# ── Cache ──────────────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}
CACHE_TTL = 3600  # 1 hour

# ── Discipline config ──────────────────────────────────────────────────────
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

# OpenAlex concept IDs for discipline filtering (speeds up results)
DISCIPLINE_CONCEPTS = {
    "aerospace": "C27206212",   # Aerospace engineering
    "materials": "C192562407",  # Materials science
    "textile":   "C107038049",  # Textile engineering
}


def tag_discipline(title: str, abstract: str | None, concepts: list) -> str:
    """Tag a paper with a discipline based on concepts and text."""
    # Check OpenAlex concept tags first
    concept_names = " ".join(c.get("display_name", "").lower() for c in concepts)
    text = (title + " " + (abstract or "") + " " + concept_names).lower()

    scores = {disc: 0 for disc in DISCIPLINE_KEYWORDS}
    for disc, keywords in DISCIPLINE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[disc] += 1

    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "general"


def _cache_key(query, discipline, year_from, year_to, limit) -> str:
    raw = f"{query}|{discipline}|{year_from}|{year_to}|{limit}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached(key: str) -> list | None:
    entry = _cache.get(key)
    if entry and time.time() < entry["expires"]:
        return entry["papers"]
    return None


def _set_cache(key: str, papers: list):
    _cache[key] = {"papers": papers, "expires": time.time() + CACHE_TTL}


async def _fetch_openalex(params: dict, max_retries: int = 3) -> dict:
    """
    Fetch from OpenAlex API with retry.
    Completely free — no API key required.
    Rate limit: 10 req/sec (very generous, rarely hit).
    """
    url = "https://api.openalex.org/works"

    # Polite pool — add your email for faster responses (optional)
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
                raise Exception("OpenAlex rate limit reached. Please wait a moment and retry.")

            resp.raise_for_status()

    return {"results": [], "meta": {"count": 0}}


async def search_papers(
    query: str,
    discipline: str = "all",
    year_from: int | None = None,
    year_to:   int | None = None,
    limit:     int = 10,
) -> list[Paper]:

    # 1. Cache check
    key = _cache_key(query, discipline, year_from, year_to, limit)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    # 2. Build OpenAlex params
    # OpenAlex uses "search" for full-text search across title+abstract
    params: dict = {
        "search":     query,
        "per_page":   min(limit, 50),
        "select":     (
            "id,title,authorships,publication_year,abstract_inverted_index,"
            "cited_by_count,doi,primary_location,open_access,concepts"
        ),
        "sort": "relevance_score:desc",
    }

    # Add discipline concept filter
    if discipline != "all" and discipline in DISCIPLINE_CONCEPTS:
        params["filter"] = f"concepts.id:{DISCIPLINE_CONCEPTS[discipline]}"

    # Year filter
    year_filters = []
    if year_from:
        year_filters.append(f"publication_year:>{year_from - 1}")
    if year_to:
        year_filters.append(f"publication_year:<{year_to + 1}")
    if year_filters:
        existing = params.get("filter", "")
        combined = ",".join([existing] + year_filters) if existing else ",".join(year_filters)
        params["filter"] = combined

    # 3. Fetch
    data = await _fetch_openalex(params)

    # 4. Parse OpenAlex results
    papers = []
    for item in data.get("results", []):

        # Reconstruct abstract from inverted index
        abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))

        # Authors
        authors = [
            Author(name=a.get("author", {}).get("display_name", "Unknown"))
            for a in item.get("authorships", [])[:10]  # cap at 10 authors
        ]

        # Journal / venue
        primary = item.get("primary_location") or {}
        source  = primary.get("source") or {}
        journal = source.get("display_name")

        # Open access PDF
        oa      = item.get("open_access") or {}
        oa_url  = oa.get("oa_url")

        # DOI url
        doi     = item.get("doi")  # already full URL e.g. https://doi.org/10.xxxx
        paper_url = doi or f"https://openalex.org/{item.get('id','').split('/')[-1]}"

        # Concepts for discipline tagging
        concepts = item.get("concepts") or []

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
            discipline_tag  = tag_discipline(
                item.get("title", ""), abstract, concepts
            ),
        ))

    _set_cache(key, papers)
    return papers


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """
    OpenAlex stores abstracts as inverted index: {"word": [positions]}.
    This reconstructs the original text.
    """
    if not inverted_index:
        return None
    try:
        words = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        if not words:
            return None
        return " ".join(words[i] for i in sorted(words.keys()))
    except Exception:
        return None
