from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
async def health():
    return {
        "status":     "healthy",
        "timestamp":  datetime.utcnow().isoformat(),
        "version":    "1.0.0",
        "platform":   "TriField AI",
        "disciplines": ["Aerospace", "Materials Science", "Textile Engineering"],
        "endpoints": {
            "search":     "/api/search/?query=your+query&discipline=aerospace",
            "pdf_upload": "POST /api/pdf/upload",
            "pdf_chat":   "POST /api/pdf/chat",
            "properties": "GET  /api/pdf/extract-properties/{session_id}",
            "citations":  "POST /api/citations/",
            "cite_styles":"GET  /api/citations/styles",
            "docs":       "/docs",
        }
    }
