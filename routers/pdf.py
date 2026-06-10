import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from rag.pipeline import (
    ingest_pdf, chat_with_pdf, extract_properties, get_session_download_url
)
from analytics.tracker import get_tracker
from models.schemas import ChatRequest, ChatResponse, PropertyExtractionResponse

router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a PDF:
    1. Stores original file in Supabase Storage (bucket: pdfs)
    2. Extracts text → chunks → embeds in Qdrant
    3. Saves session metadata in pdf_sessions table
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    contents = await file.read()
    size_mb  = len(contents) / (1024 * 1024)
    if size_mb > 20:
        raise HTTPException(400, f"File too large ({size_mb:.1f} MB). Max is 20 MB.")

    t0 = time.time()
    try:
        session_id = await ingest_pdf(contents, file.filename)
        latency    = round((time.time() - t0) * 1000, 1)
        get_tracker().record_pdf(file.filename, session_id, 0, latency)
        return {
            "session_id": session_id,
            "filename":   file.filename,
            "size_mb":    round(size_mb, 2),
            "message":    (
                "PDF uploaded to Supabase Storage and indexed in Qdrant. "
                "Use session_id to chat or extract properties."
            ),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat with an uploaded PDF. History is persisted in Supabase."""
    try:
        result = await chat_with_pdf(req.session_id, req.question)
        return ChatResponse(
            session_id=req.session_id,
            answer=result["answer"],
            sources=result["sources"],
            history=result["history"],
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/extract-properties/{session_id}", response_model=PropertyExtractionResponse)
async def extract(session_id: str):
    """Extract material properties from an uploaded PDF using Claude."""
    try:
        props = await extract_properties(session_id)
        return PropertyExtractionResponse(session_id=session_id, properties=props)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/download-url/{session_id}")
async def download_url(session_id: str):
    """
    Return a 1-hour signed Supabase Storage URL to re-download the original PDF.
    Returns null if Storage is not configured or the file is not found.
    """
    url = await get_session_download_url(session_id)
    if not url:
        raise HTTPException(404, "No stored file found for this session.")
    return {"session_id": session_id, "url": url, "expires_in_seconds": 3600}
