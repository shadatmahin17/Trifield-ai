from fastapi import APIRouter, HTTPException
from models.schemas import CitationRequest, CitationResponse
import re

router = APIRouter()


def _apa(authors, title, year, journal, volume, pages, doi):
    def fmt(a):
        if "," in a:
            return a.strip()
        p = a.strip().split()
        return f"{p[-1]}, {'. '.join(n[0] for n in p[:-1])}." if len(p) > 1 else a.strip()
    auth = ", ".join(fmt(a) for a in authors)
    if len(authors) > 1:
        parts = [fmt(a) for a in authors]
        auth  = ", ".join(parts[:-1]) + ", & " + parts[-1]
    j   = f"*{journal}*" if journal else ""
    vol = f", *{volume}*" if volume else ""
    pg  = f", {pages}" if pages else ""
    d   = f" https://doi.org/{doi}" if doi else ""
    return f"{auth} ({year}). {title}. {j}{vol}{pg}.{d}"


def _ieee(authors, title, year, journal, volume, pages, doi):
    def fmt(a):
        p = a.replace(",","").strip().split()
        return f"{'. '.join(n[0] for n in p[:-1])}. {p[-1]}" if len(p)>1 else a
    auth = ", ".join(fmt(a) for a in authors)
    j    = f"*{journal}*" if journal else ""
    vol  = f", vol. {volume}" if volume else ""
    pg   = f", pp. {pages}" if pages else ""
    d    = f" https://doi.org/{doi}" if doi else ""
    return f'{auth}, "{title}," {j}{vol}{pg}, {year}.{d}'


def _aiaa(authors, title, year, journal, volume, pages, doi):
    auth  = ", ".join(a.strip() for a in authors)
    parts = [p for p in [journal, f"Vol. {volume}" if volume else "", f"pp. {pages}" if pages else ""] if p]
    d     = f" https://doi.org/{doi}" if doi else ""
    return f'{auth}, "{title}," {", ".join(parts)}, {year}.{d}'


def _mla(authors, title, year, journal, volume, pages, doi):
    auth  = ", ".join(a.strip() for a in authors)
    j     = f"*{journal}*" if journal else ""
    parts = [p for p in [j, f"vol. {volume}" if volume else "", f"pp. {pages}" if pages else ""] if p]
    d     = f" https://doi.org/{doi}" if doi else ""
    return f'{auth}. "{title}." {", ".join(parts)}, {year}.{d}'


def _chicago(authors, title, year, journal, volume, pages, doi):
    auth  = ", ".join(a.strip() for a in authors)
    j     = f"*{journal}*" if journal else ""
    d     = f" https://doi.org/{doi}" if doi else ""
    return f'{auth}. "{title}." {j} {volume or ""} ({year}): {pages or ""}.{d}'


def _harvard(authors, title, year, journal, volume, pages, doi):
    auth  = ", ".join(a.strip() for a in authors)
    j     = f"*{journal}*" if journal else ""
    vol   = f", {volume}" if volume else ""
    pg    = f", pp. {pages}" if pages else ""
    d     = f" https://doi.org/{doi}" if doi else ""
    return f'{auth} ({year}) "{title}", {j}{vol}{pg}.{d}'


STYLE_FN = {"apa": _apa, "ieee": _ieee, "aiaa": _aiaa, "mla": _mla, "chicago": _chicago, "harvard": _harvard}


@router.post("/", response_model=CitationResponse)
async def generate_citation(req: CitationRequest):
    try:
        fn = STYLE_FN.get(req.style.lower())
        if not fn:
            raise ValueError(f"Unsupported style: {req.style}. Use: apa, ieee, aiaa, mla, chicago, harvard")
        citation = fn(
            authors = req.authors or ["Unknown Author"],
            title   = req.title   or "Untitled",
            year    = str(req.year) if req.year else "n.d.",
            journal = req.journal or "",
            volume  = req.volume  or "",
            pages   = req.pages   or "",
            doi     = req.doi     or "",
        )
        return CitationResponse(style=req.style, citation=citation, paper_id=req.paper_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/styles")
async def list_styles():
    return {"supported_styles": [
        {"id": "apa",     "name": "APA 7th Edition",     "used_in": "Most journals, psychology"},
        {"id": "ieee",    "name": "IEEE",                 "used_in": "Electrical engineering, CS"},
        {"id": "aiaa",    "name": "AIAA",                 "used_in": "Aerospace engineering"},
        {"id": "mla",     "name": "MLA 9th Edition",      "used_in": "Humanities"},
        {"id": "chicago", "name": "Chicago 17th Edition", "used_in": "History, arts"},
        {"id": "harvard", "name": "Harvard",              "used_in": "UK universities, materials"},
    ]}
