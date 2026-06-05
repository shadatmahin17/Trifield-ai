import uuid
import anthropic
import chromadb
from core.config import get_settings

_settings = get_settings()

# Lazy initialization — nothing loads at import time
# This lets uvicorn bind the port immediately on startup
_chroma_client = None
_embed_fn = None
_chat_history: dict[str, list[dict]] = {}


def _get_chroma_client():
    """Initialize ChromaDB only on first use."""
    global _chroma_client, _embed_fn
    if _chroma_client is None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        _embed_fn = ONNXMiniLM_L6_V2()
        _chroma_client = chromadb.PersistentClient(path=_settings.chroma_path)
    return _chroma_client


def _get_or_create_collection(session_id: str):
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=f"pdf_{session_id}",
        embedding_function=_embed_fn,
    )


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return chunks


async def ingest_pdf(file_bytes: bytes, filename: str) -> str:
    import pymupdf
    session_id = str(uuid.uuid4())

    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    chunks = _chunk_text(full_text)
    if not chunks:
        raise ValueError("Could not extract text from PDF.")

    col = _get_or_create_collection(session_id)
    col.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))],
    )

    _chat_history[session_id] = []
    return session_id


async def chat_with_pdf(session_id: str, question: str) -> dict:
    if session_id not in _chat_history:
        raise ValueError(f"Session '{session_id}' not found. Please upload a PDF first.")

    col     = _get_or_create_collection(session_id)
    results = col.query(query_texts=[question], n_results=5)
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    context   = "\n\n---\n\n".join(docs)
    sources   = [f"chunk {m['chunk_index']}" for m in metadatas]
    history   = _chat_history[session_id]

    system_prompt = (
        "You are TriField AI, an expert research assistant specialising in "
        "aerospace structures, advanced materials, and textile engineering. "
        "Answer using ONLY the context from the PDF. Cite the relevant section. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"CONTEXT FROM PDF:\n{context}"
    )

    client   = anthropic.Anthropic(api_key=_settings.anthropic_api_key)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=history + [{"role": "user", "content": question}],
    )
    answer = response.content[0].text

    _chat_history[session_id].append({"role": "user",      "content": question})
    _chat_history[session_id].append({"role": "assistant", "content": answer})

    return {"answer": answer, "sources": sources, "history": _chat_history[session_id]}


async def extract_properties(session_id: str) -> list[dict]:
    if session_id not in _chat_history:
        raise ValueError(f"Session '{session_id}' not found.")

    col     = _get_or_create_collection(session_id)
    results = col.query(
        query_texts=["mechanical properties tensile strength modulus fibre volume fraction"],
        n_results=8,
    )
    context = "\n\n---\n\n".join(results["documents"][0])

    prompt = (
        "Extract ALL material/mechanical properties from the text below. "
        "Return ONLY a JSON array. Each item: property_name, value, unit, "
        "test_standard (if mentioned), page_ref (if mentioned). "
        "No markdown, no explanation.\n\n"
        f"TEXT:\n{context}"
    )

    client   = anthropic.Anthropic(api_key=_settings.anthropic_api_key)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    raw = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []
