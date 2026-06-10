import asyncio, time
from fastapi import APIRouter, UploadFile, File, HTTPException
from rag.pipeline import (
    ingest_pdf, chat_with_pdf, extract_properties, get_session_download_url
)
from analytics.tracker import get_tracker
from models.schemas import ChatRequest, ChatResponse, PropertyExtractionResponse

router = APIRouter()


# ── Supabase query helpers (sync → run in thread) ──────────────────────────

def _list_sessions_sync(limit: int = 50) -> list[dict]:
    from db.client import get_db, is_configured
    if not is_configured():
        return []
    r = (
        get_db().table("pdf_sessions")
        .select("session_id,filename,size_bytes,chunk_count,latency_ms,created_at,last_accessed,storage_path")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return r.data or []


def _get_session_sync(session_id: str) -> dict | None:
    from db.client import get_db, is_configured
    if not is_configured():
        return None
    r = (
        get_db().table("pdf_sessions")
        .select("*")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )
    return r.data


def _get_chat_history_sync(session_id: str) -> list[dict]:
    from db.client import get_db, is_configured
    if not is_configured():
        return []
    r = (
        get_db().table("chat_messages")
        .select("role,content,created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )
    return r.data or []


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a PDF:
    1. Stores original bytes in Supabase Storage (bucket: pdfs)
    2. Extracts text → chunks → embeds in Qdrant
    3. Saves session row in pdf_sessions table
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
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
            "message": (
                "PDF uploaded to Supabase Storage and indexed in Qdrant. "
                "Use session_id to chat or extract properties."
            ),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/sessions")
async def list_sessions(limit: int = 50):
    """
    List all previously uploaded PDF sessions from Supabase.
    Frontend uses this to show the PDF library.
    """
    try:
        sessions = await asyncio.to_thread(_list_sessions_sync, limit)
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get full detail for one session: metadata + chat history + signed download URL.
    Frontend uses this to restore a previous session.
    """
    session = await asyncio.to_thread(_get_session_sync, session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found.")

    history = await asyncio.to_thread(_get_chat_history_sync, session_id)

    # Generate a fresh signed URL if the file is in Storage
    download_url = None
    if session.get("storage_path"):
        try:
            from storage.pdf_store import get_signed_url
            download_url = await get_signed_url(session["storage_path"])
        except Exception:
            pass

    return {
        **session,
        "history":      history,
        "download_url": download_url,
    }


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
    """Extract material properties using Claude."""
    try:
        props = await extract_properties(session_id)
        return PropertyExtractionResponse(session_id=session_id, properties=props)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/download-url/{session_id}")
async def download_url(session_id: str):
    """Return a 1-hour signed Supabase Storage URL to re-download the original PDF."""
    url = await get_session_download_url(session_id)
    if not url:
        raise HTTPException(404, "No stored file found for this session.")
    return {"session_id": session_id, "url": url, "expires_in_seconds": 3600}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a PDF session — removes from Supabase DB, Storage, and Qdrant."""
    from db.client import get_db, is_configured
    session = await asyncio.to_thread(_get_session_sync, session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    # Delete from Storage
    if session.get("storage_path"):
        try:
            from storage.pdf_store import delete_pdf
            await delete_pdf(session["storage_path"])
        except Exception:
            pass
    # Delete from Qdrant
    try:
        from vectorstore.qdrant_store import get_store
        get_store().delete_session(session_id)
    except Exception:
        pass
    # Delete from DB (cascades to chat_messages)
    if is_configured():
        await asyncio.to_thread(
            lambda: get_db().table("pdf_sessions").delete().eq("session_id", session_id).execute()
        )
    return {"deleted": session_id}
