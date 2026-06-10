"""
RAG pipeline — PDF ingestion, chat history, and property extraction.

Storage layer (Supabase):
  - PDF file bytes  → Supabase Storage bucket "pdfs"
  - Session metadata → pdf_sessions table
  - Chat history    → chat_messages table  (replaces in-memory dict)
  - Falls back to in-memory if Supabase is not configured (dev mode)

Vector layer (Qdrant):
  - Chunk embeddings still live in Qdrant (unchanged)
"""
import asyncio
import uuid
import logging
from vectorstore.qdrant_store import get_store
from core.llm import llm_call
from prompts.templates import PDF_CHAT_SYSTEM, PROPERTY_EXTRACT

logger = logging.getLogger(__name__)

# ── In-memory fallback (used when Supabase is not configured) ───────────────
_chat_history: dict[str, list[dict]] = {}


# ── Supabase session helpers ────────────────────────────────────────────────

def _session_exists_sync(session_id: str) -> bool:
    """Return True if this session_id exists in pdf_sessions."""
    from db.client import get_db, is_configured
    if not is_configured():
        return session_id in _chat_history
    try:
        r = (
            get_db().table("pdf_sessions")
            .select("session_id")
            .eq("session_id", session_id)
            .maybe_single()
            .execute()
        )
        return r.data is not None
    except Exception:
        return session_id in _chat_history


def _create_session_sync(
    session_id: str, filename: str, storage_path: str,
    size_bytes: int, chunk_count: int, latency_ms: float,
):
    from db.client import get_db, is_configured
    if not is_configured():
        return
    get_db().table("pdf_sessions").insert({
        "session_id":   session_id,
        "filename":     filename,
        "storage_path": storage_path,
        "size_bytes":   size_bytes,
        "chunk_count":  chunk_count,
        "latency_ms":   latency_ms,
    }).execute()


def _touch_session_sync(session_id: str):
    """Update last_accessed timestamp."""
    from db.client import get_db, is_configured
    if not is_configured():
        return
    try:
        from datetime import datetime, timezone
        get_db().table("pdf_sessions").update(
            {"last_accessed": datetime.now(timezone.utc).isoformat()}
        ).eq("session_id", session_id).execute()
    except Exception:
        pass


def _get_history_sync(session_id: str) -> list[dict]:
    """Fetch ordered chat messages from Supabase."""
    from db.client import get_db, is_configured
    if not is_configured():
        return list(_chat_history.get(session_id, []))
    try:
        r = (
            get_db().table("chat_messages")
            .select("role,content")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return [{"role": m["role"], "content": m["content"]} for m in (r.data or [])]
    except Exception as e:
        logger.warning(f"DB get_history failed, using in-memory: {e}")
        return list(_chat_history.get(session_id, []))


def _append_messages_sync(session_id: str, user_msg: str, assistant_msg: str):
    """Append user + assistant turns to chat_messages."""
    from db.client import get_db, is_configured
    if not is_configured():
        _chat_history.setdefault(session_id, [])
        _chat_history[session_id].append({"role": "user",      "content": user_msg})
        _chat_history[session_id].append({"role": "assistant", "content": assistant_msg})
        return
    try:
        get_db().table("chat_messages").insert([
            {"session_id": session_id, "role": "user",      "content": user_msg},
            {"session_id": session_id, "role": "assistant", "content": assistant_msg},
        ]).execute()
    except Exception as e:
        logger.warning(f"DB append_messages failed, using in-memory: {e}")
        _chat_history.setdefault(session_id, [])
        _chat_history[session_id].append({"role": "user",      "content": user_msg})
        _chat_history[session_id].append({"role": "assistant", "content": assistant_msg})


# ── Text chunking ────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if len(c.split()) > 20]


# ── Public API ───────────────────────────────────────────────────────────────

async def ingest_pdf(file_bytes: bytes, filename: str) -> str:
    """
    1. Upload raw PDF bytes to Supabase Storage ("pdfs" bucket).
    2. Extract text → chunk → embed into Qdrant.
    3. Save session metadata to pdf_sessions table.
    Returns session_id.
    """
    import pymupdf
    import time

    session_id = str(uuid.uuid4())
    t0         = time.time()

    # ── Extract text ──
    doc       = pymupdf.open(stream=file_bytes, filetype="pdf")
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    if not full_text.strip():
        raise ValueError("Could not extract text from PDF.")

    chunks = _chunk_text(full_text)
    if not chunks:
        raise ValueError("PDF text too short to process.")

    # ── Upload to Supabase Storage (parallel with Qdrant ingestion) ──
    storage_path = ""
    try:
        from storage.pdf_store import upload_pdf
        from db.client import is_configured
        if is_configured():
            storage_path = await upload_pdf(session_id, filename, file_bytes)
    except Exception as e:
        logger.warning(f"Storage upload failed (continuing without it): {e}")

    # ── Embed into Qdrant ──
    store = get_store()
    n     = store.ingest(session_id, chunks, filename)

    latency_ms = round((time.time() - t0) * 1000, 1)

    # ── Persist session to Supabase DB ──
    try:
        await asyncio.to_thread(
            _create_session_sync,
            session_id, filename, storage_path, len(file_bytes), n, latency_ms,
        )
    except Exception as e:
        logger.warning(f"DB session create failed (continuing): {e}")

    # ── In-memory fallback ──
    _chat_history[session_id] = []

    logger.info(f"PDF ingested: {filename} → {n} chunks, session={session_id}, storage={storage_path or 'none'}")
    return session_id


async def chat_with_pdf(session_id: str, question: str) -> dict:
    """
    Semantic retrieval from Qdrant + LLM answer.
    Chat history persisted in Supabase (falls back to in-memory).
    """
    # Validate session exists
    exists = await asyncio.to_thread(_session_exists_sync, session_id)
    if not exists:
        raise ValueError(f"Session '{session_id}' not found. Upload a PDF first.")

    # Touch last_accessed (fire-and-forget)
    asyncio.create_task(asyncio.to_thread(_touch_session_sync, session_id))

    store   = get_store()
    results = store.search(session_id, question, top_k=6)

    if not results:
        return {
            "answer":  "No relevant content found for your question in this PDF.",
            "sources": [],
            "history": await asyncio.to_thread(_get_history_sync, session_id),
        }

    results.sort(key=lambda r: r["score"], reverse=True)
    context = "\n\n---\n\n".join(
        f"[Chunk {r['chunk_index']+1}, relevance={r['score']:.2f}]\n{r['text']}"
        for r in results
    )
    sources = [f"chunk {r['chunk_index']+1} (score={r['score']:.2f})" for r in results]

    # Fetch full history for context window
    history  = await asyncio.to_thread(_get_history_sync, session_id)
    messages = history + [{"role": "user", "content": question}]
    system   = PDF_CHAT_SYSTEM.format(context=context)

    answer = await llm_call(
        system=system, messages=messages,
        max_tokens=1024, task="pdf_chat",
    )

    # Persist both turns
    await asyncio.to_thread(_append_messages_sync, session_id, question, answer)

    updated_history = await asyncio.to_thread(_get_history_sync, session_id)
    return {"answer": answer, "sources": sources, "history": updated_history}


async def extract_properties(session_id: str) -> list[dict]:
    """Semantic search for property-bearing chunks → Claude extraction."""
    exists = await asyncio.to_thread(_session_exists_sync, session_id)
    if not exists:
        raise ValueError(f"Session '{session_id}' not found.")

    store = get_store()
    queries = [
        "tensile strength flexural strength Young's modulus mechanical properties",
        "fibre volume fraction void content density weight",
        "impact strength fracture toughness interlaminar shear",
        "test standard ASTM ISO specimen dimensions",
    ]

    all_chunks, seen = [], set()
    for q in queries:
        for r in store.search(session_id, q, top_k=4):
            if r["chunk_index"] not in seen:
                seen.add(r["chunk_index"])
                all_chunks.append(r)

    if not all_chunks:
        return []

    import json
    context = "\n\n---\n\n".join(r["text"] for r in all_chunks[:10])
    raw = await llm_call(
        system="You are a materials science data extraction specialist.",
        messages=[{"role": "user", "content": f"{PROPERTY_EXTRACT}\n\nTEXT:\n{context}"}],
        max_tokens=2048, prefer_json=True, task="property_extract",
    )
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


async def get_session_download_url(session_id: str) -> str | None:
    """Return a 1-hour signed URL to re-download the original PDF from Storage."""
    try:
        from db.client import get_db, is_configured
        if not is_configured():
            return None
        r = (
            get_db().table("pdf_sessions")
            .select("storage_path")
            .eq("session_id", session_id)
            .maybe_single()
            .execute()
        )
        path = r.data and r.data.get("storage_path")
        if not path:
            return None
        from storage.pdf_store import get_signed_url
        return await get_signed_url(path)
    except Exception as e:
        logger.warning(f"get_session_download_url failed: {e}")
        return None
