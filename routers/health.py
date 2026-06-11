from fastapi import APIRouter
from datetime import datetime
from core.config import get_settings

router = APIRouter()

@router.get("/health")
async def health():
    s = get_settings()
    has_anthropic = bool(s.anthropic_api_key)
    has_groq      = bool(s.groq_api_key)

    # BUG FIX: was `bool(s.qdrant_url) or True` — always True regardless of config.
    has_qdrant_cloud = bool(s.qdrant_url and s.qdrant_api_key)

    # Probe local/cloud Qdrant liveness
    qdrant_status = "unchecked"
    try:
        from vectorstore.qdrant_store import get_store
        get_store()._client_().get_collections()
        qdrant_status = "reachable"
    except Exception as e:
        qdrant_status = f"unreachable: {e}"

    return {
        "status":      "healthy",
        "timestamp":   datetime.utcnow().isoformat(),
        "version":     "2.0.0",
        "platform":    "TriField AI",
        "disciplines": ["Aerospace", "Materials Science", "Textile Engineering"],
        "llm": {
            "primary":        "Anthropic Claude Haiku" if has_anthropic else "Groq Llama 3.3 70B",
            "fallback":       "Groq Llama 3.3 70B" if has_groq else "None",
            "routing":        "task-aware (Claude=quality, Groq=speed)",
            "anthropic_key":  "set" if has_anthropic else "missing",
            "groq_key":       "set" if has_groq else "missing",
        },
        "vector_db": {
            "engine":  "Qdrant Cloud" if has_qdrant_cloud else "Qdrant Local",
            "status":  qdrant_status,
        },
        "features": {
            "search":          "OpenAlex + Crossref + arXiv + PubMed + Unpaywall",
            "query_rewriting": "LLM (Groq) + rule-based fallback",
            "paper_scoring":   "weighted: relevance 40% + citations 25% + recency 15% + journal 10% + OA 10%",
            "streaming_search":"SSE — live source progress",
            "pdf_rag":         "Qdrant semantic retrieval + Claude",
            "copilot":         "Research gaps + trends + experiments",
            "citations":       "APA, IEEE, AIAA, Harvard, MLA, Chicago",
            "analytics":       "search latency, top queries, success rate",
        },
        "endpoints": {
            "search":          "GET  /api/search/?query=...",
            "search_stream":   "GET  /api/search/stream?query=...",
            "copilot_analyse": "POST /api/copilot/analyse",
            "copilot_summary": "POST /api/copilot/summary",
            "pdf_upload":      "POST /api/pdf/upload",
            "pdf_chat":        "POST /api/pdf/chat",
            "properties":      "GET  /api/pdf/extract-properties/{id}",
            "citations":       "POST /api/citations/",
            "analytics":       "GET  /api/analytics/",
            "docs":            "/docs",
        }
    }
