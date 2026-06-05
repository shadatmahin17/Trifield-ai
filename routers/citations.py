from fastapi import APIRouter, HTTPException
from models.schemas import CitationRequest, CitationResponse

router = APIRouter()

# ── Citation formatting logic ────────────────────────────────────────────────

def format_authors_apa(authors: list[str]) -> str:
    parts = []
    for a in authors:
        if "," in a:
            parts.append(a.strip())
        else:
            names = a.strip().split()
            if len(names) >= 2:
                last  = names[-1]
                inits = " ".join(n[0] + "." for n in names[:-1])
                parts.append(f"{last}, {inits}")
            else:
                parts.append(a.strip())
    if len(parts) == 1:
        return parts[0]
    if len(parts) <= 6:
        return ", ".join(parts[:-1]) + ", & " + parts[-1]
    return ", ".join(parts[:6]) + ", ... " + parts[-1]

def format_authors_ieee(authors: list[str]) -> str:
    parts = []
    for a in authors:
        names = a.replace(",", "").strip().split()
        if len(names) >= 2:
            inits = ". ".join(n[0] for n in names[:-1]) + "."
            parts.append(f"{inits} {names[-1]}")
        else:
            parts.append(a.strip())
    return ", ".join(parts)

def format_authors_simple(authors: list[str]) -> str:
    return ", ".join(a.strip() for a in authors)

def make_citation(req: CitationRequest) -> str:
    authors = req.authors or ["Unknown Author"]
    title   = req.title   or "Untitled"
    year    = req.year    or "n.d."
    journal = req.journal or ""
    volume  = req.volume  or ""
    pages   = req.pages   or ""
    doi     = req.doi     or ""
    doi_str = f" https://doi.org/{doi}" if doi else ""

    style = req.style.lower()

    if style == "apa":
        auth = format_authors_apa(authors)
        j    = f"*{journal}*" if journal else ""
        vol  = f", *{volume}*" if volume else ""
        pg   = f", {pages}"   if pages  else ""
        return f"{auth} ({year}). {title}. {j}{vol}{pg}.{doi_str}"

    elif style == "ieee":
        auth = format_authors_ieee(authors)
        j    = f"*{journal}*" if journal else ""
        vol  = f", vol. {volume}" if volume else ""
        pg   = f", pp. {pages}"  if pages  else ""
        return f'{auth}, "{title}," {j}{vol}{pg}, {year}.{doi_str}'

    elif style == "aiaa":
        # AIAA style: Last, F. M., "Title," Journal, Vol. X, No. Y, Year, pp. Z.
        auth = format_authors_simple(authors)
        j    = journal or ""
        vol  = f"Vol. {volume}" if volume else ""
        pg   = f"pp. {pages}"  if pages  else ""
        parts = [p for p in [j, vol, pg] if p]
        return f'{auth}, "{title}," {", ".join(parts)}, {year}.{doi_str}'

    elif style == "mla":
        auth = format_authors_simple(authors)
        j    = f'*{journal}*' if journal else ""
        vol  = f"vol. {volume}" if volume else ""
        pg   = f"pp. {pages}"  if pages  else ""
        parts = [p for p in [j, vol, pg] if p]
        return f'{auth}. "{title}." {", ".join(parts)}, {year}.{doi_str}'

    elif style == "chicago":
        auth = format_authors_simple(authors)
        j    = f'*{journal}*' if journal else ""
        vol  = f"{volume}" if volume else ""
        pg   = f"{pages}"  if pages  else ""
        return f'{auth}. "{title}." {j} {vol} ({year}): {pg}.{doi_str}'

    elif style == "harvard":
        auth = format_authors_simple(authors)
        j    = f'*{journal}*' if journal else ""
        vol  = f", {volume}"  if volume else ""
        pg   = f", pp.{pages}" if pages  else ""
        return f'{auth} ({year}) "{title}", {j}{vol}{pg}.{doi_str}'

    else:
        raise ValueError(f"Unsupported citation style: '{style}'. "
                         "Supported: apa, ieee, aiaa, mla, chicago, harvard")


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/", response_model=CitationResponse)
async def generate_citation(req: CitationRequest):
    """
    Generate a formatted citation.
    Supports: apa | ieee | aiaa | mla | chicago | harvard

    Body example:
    {
      "title": "Mechanical behaviour of jute-glass hybrid composites",
      "authors": ["Mahin, S.H.", "Islam, M."],
      "year": 2024,
      "journal": "Composites Science and Technology",
      "volume": "250",
      "pages": "110532",
      "doi": "10.1016/j.compscitech.2024.110532",
      "style": "apa"
    }
    """
    try:
        citation = make_citation(req)
        return CitationResponse(
            style=req.style,
            citation=citation,
            paper_id=req.paper_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/styles")
async def list_styles():
    """List all supported citation styles."""
    return {
        "supported_styles": [
            {"id": "apa",     "name": "APA 7th Edition",     "used_in": "Most journals, psychology, social science"},
            {"id": "ieee",    "name": "IEEE",                 "used_in": "Electrical engineering, computer science"},
            {"id": "aiaa",    "name": "AIAA",                 "used_in": "Aerospace engineering journals"},
            {"id": "mla",     "name": "MLA 9th Edition",      "used_in": "Humanities"},
            {"id": "chicago", "name": "Chicago 17th Edition", "used_in": "History, arts"},
            {"id": "harvard", "name": "Harvard",              "used_in": "UK/Australian universities, materials journals"},
        ]
    }
