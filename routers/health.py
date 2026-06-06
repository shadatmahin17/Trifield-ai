from fastapi import APIRouter
from datetime import datetime
from core.config import get_settings

router = APIRouter()

@router.get("/health")
async def health():
    settings = get_settings()

    # Determine active LLM
    has_anthropic = bool(settings.anthropic_api_key)
    has_groq      = bool(settings.groq_api_key)

    if has_anthropic:
        llm_primary  = "Anthropic Claude Haiku"
        llm_fallback = "Groq Llama 3.3 70B" if has_groq else "None configured"
    elif has_groq:
        llm_primary  = "Groq Llama 3.3 70B (Anthropic key missing)"
        llm_fallback = "None"
    else:
        llm_primary  = "None configured — set ANTHROPIC_API_KEY or GROQ_API_KEY"
        llm_fallback = "None"

    return {
        "status":      "healthy",
        "timestamp":   datetime.utcnow().isoformat(),
        "version":     "1.0.0",
        "platform":    "TriField AI",
        "disciplines": ["Aerospace", "Materials Science", "Textile Engineering"],
        "llm": {
            "primary":       llm_primary,
            "fallback":      llm_fallback,
            "anthropic_key": "set" if has_anthropic else "missing",
            "groq_key":      "set" if has_groq      else "missing",
        },
        "endpoints": {
            "search":     "/api/search/?query=your+query&discipline=aerospace",
            "pdf_upload": "POST /api/pdf/upload",
            "pdf_chat":   "POST /api/pdf/chat",
            "properties": "GET  /api/pdf/extract-properties/{session_id}",
            "citations":  "POST /api/citations/",
            "docs":       "/docs",
        }
    }
