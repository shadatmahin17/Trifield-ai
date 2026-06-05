from pydantic import BaseModel
from typing import Optional


# ── Search ──────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    discipline: Optional[str] = "all"   # all | aerospace | materials | textile
    year_from: Optional[int] = None
    year_to:   Optional[int] = None
    limit:     Optional[int] = 10

class Author(BaseModel):
    name: str

class Paper(BaseModel):
    paper_id:       str
    title:          str
    authors:        list[Author]
    year:           Optional[int]
    abstract:       Optional[str]
    citation_count: Optional[int]
    url:            Optional[str]
    open_access_url: Optional[str]
    journal:        Optional[str]
    discipline_tag: Optional[str]

class SearchResponse(BaseModel):
    query:       str
    total:       int
    discipline:  str
    papers:      list[Paper]


# ── PDF Chat ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question:   str

class ChatMessage(BaseModel):
    role:    str   # user | assistant
    content: str

class ChatResponse(BaseModel):
    session_id: str
    answer:     str
    sources:    list[str]   # page references
    history:    list[ChatMessage]


# ── Citations ─────────────────────────────────────────────
class CitationRequest(BaseModel):
    paper_id: Optional[str] = None
    # Or provide raw metadata:
    title:    Optional[str] = None
    authors:  Optional[list[str]] = None
    year:     Optional[int] = None
    journal:  Optional[str] = None
    volume:   Optional[str] = None
    pages:    Optional[str] = None
    doi:      Optional[str] = None
    style:    str = "apa"   # apa | ieee | aiaa | mla | chicago | harvard

class CitationResponse(BaseModel):
    style:      str
    citation:   str
    paper_id:   Optional[str]


# ── Property Extraction ───────────────────────────────────
class PropertyRow(BaseModel):
    property_name:  str
    value:          str
    unit:           Optional[str]
    test_standard:  Optional[str]
    page_ref:       Optional[str]

class PropertyExtractionResponse(BaseModel):
    session_id:  str
    properties:  list[PropertyRow]
