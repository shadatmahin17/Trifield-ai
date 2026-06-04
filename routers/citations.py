from fastapi import APIRouter, HTTPException

from models.schemas import CitationRequest, CitationResponse

router = APIRouter()


def _format_authors(authors: list[str] | None, style: str) -> str:
    if not authors:
        return "Unknown author"
    if style == "ieee":
        return ", ".join(authors)
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} & {authors[1]}"
    return f"{authors[0]} et al."


def _build_citation(request: CitationRequest) -> str:
    style = request.style.lower()
    authors = _format_authors(request.authors, style)
    title = request.title or "Untitled"
    year = request.year or "n.d."
    journal = request.journal or ""
    doi = f" https://doi.org/{request.doi}" if request.doi else ""
    volume = f", {request.volume}" if request.volume else ""
    pages = f", {request.pages}" if request.pages else ""

    if style == "ieee":
        return f'{authors}, "{title}," {journal}{volume}{pages}, {year}.{doi}'.strip()
    if style == "mla":
        return f'{authors}. "{title}." {journal}{volume}{pages}, {year}.{doi}'.strip()
    if style == "chicago":
        return f'{authors}. "{title}." {journal}{volume}{pages} ({year}).{doi}'.strip()
    if style in {"apa", "aiaa", "harvard"}:
        return f"{authors} ({year}). {title}. {journal}{volume}{pages}.{doi}".strip()

    raise ValueError("Unsupported citation style. Use apa, ieee, aiaa, mla, chicago, or harvard.")


@router.post("/", response_model=CitationResponse)
def create_citation(request: CitationRequest) -> CitationResponse:
    try:
        citation = _build_citation(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CitationResponse(
        style=request.style,
        citation=citation,
        paper_id=request.paper_id,
    )
