"""
RAG pipeline using Qdrant for semantic retrieval.
Replaces ChromaDB with proper vector search.
"""
import uuid
import logging
from vectorstore.qdrant_store import get_store
from core.llm import llm_call
from prompts.templates import PDF_CHAT_SYSTEM, PROPERTY_EXTRACT

logger = logging.getLogger(__name__)

# In-memory chat history per session
_chat_history: dict[str, list[dict]] = {}


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks."""
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if len(c.split()) > 20]   # drop tiny chunks


async def ingest_pdf(file_bytes: bytes, filename: str) -> str:
    """
    Parse PDF → chunk → embed into Qdrant.
    Returns session_id.
    """
    import pymupdf

    session_id = str(uuid.uuid4())
    doc        = pymupdf.open(stream=file_bytes, filetype="pdf")
    full_text  = "\n".join(page.get_text() for page in doc)
    doc.close()

    if not full_text.strip():
        raise ValueError("Could not extract text from PDF.")

    chunks = _chunk_text(full_text)
    if not chunks:
        raise ValueError("PDF text too short to process.")

    store = get_store()
    n     = store.ingest(session_id, chunks, filename)

    _chat_history[session_id] = []
    logger.info(f"PDF ingested: {filename} → {n} chunks, session={session_id}")
    return session_id


async def chat_with_pdf(session_id: str, question: str) -> dict:
    """
    Semantic retrieval from Qdrant + LLM answer.
    Uses Claude for accuracy-critical PDF chat.
    """
    if session_id not in _chat_history:
        raise ValueError(f"Session '{session_id}' not found. Upload a PDF first.")

    store   = get_store()
    results = store.search(session_id, question, top_k=6)

    if not results:
        return {
            "answer":  "No relevant content found for your question in this PDF.",
            "sources": [],
            "history": _chat_history[session_id],
        }

    # Build context from retrieved chunks (sorted by score)
    results.sort(key=lambda r: r["score"], reverse=True)
    context = "\n\n---\n\n".join(
        f"[Chunk {r['chunk_index']+1}, relevance={r['score']:.2f}]\n{r['text']}"
        for r in results
    )
    sources = [f"chunk {r['chunk_index']+1} (score={r['score']:.2f})" for r in results]

    system   = PDF_CHAT_SYSTEM.format(context=context)
    history  = _chat_history[session_id]
    messages = history + [{"role": "user", "content": question}]

    answer = await llm_call(
        system=system,
        messages=messages,
        max_tokens=1024,
        task="pdf_chat",   # routes to Claude
    )

    _chat_history[session_id].append({"role": "user",      "content": question})
    _chat_history[session_id].append({"role": "assistant", "content": answer})

    return {"answer": answer, "sources": sources, "history": _chat_history[session_id]}


async def extract_properties(session_id: str) -> list[dict]:
    """
    Semantic search for property-bearing chunks → Claude extraction.
    """
    if session_id not in _chat_history:
        raise ValueError(f"Session '{session_id}' not found.")

    store = get_store()

    # Multi-query retrieval for better coverage
    queries = [
        "tensile strength flexural strength Young's modulus mechanical properties",
        "fibre volume fraction void content density weight",
        "impact strength fracture toughness interlaminar shear",
        "test standard ASTM ISO specimen dimensions",
    ]

    all_chunks = []
    seen = set()
    for q in queries:
        for r in store.search(session_id, q, top_k=4):
            if r["chunk_index"] not in seen:
                seen.add(r["chunk_index"])
                all_chunks.append(r)

    if not all_chunks:
        return []

    context = "\n\n---\n\n".join(r["text"] for r in all_chunks[:10])

    import json
    raw = await llm_call(
        system="You are a materials science data extraction specialist.",
        messages=[{"role": "user", "content": f"{PROPERTY_EXTRACT}\n\nTEXT:\n{context}"}],
        max_tokens=2048,
        prefer_json=True,
        task="property_extract",   # routes to Claude
    )
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []
