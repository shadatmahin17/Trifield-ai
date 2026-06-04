import json
import os
import uuid
from functools import lru_cache
from typing import Any

import anthropic

from core.config import get_settings

# Keep Chroma/ONNX telemetry quiet in hosted environments and set this before
# any Chroma client is created. Render should be able to bind the web port
# before expensive PDF/vector-store work starts.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# In-memory chat history per session  {session_id: [{"role":..,"content":..}]}
_chat_history: dict[str, list[dict]] = {}


@lru_cache(maxsize=1)
def _get_chroma_client() -> Any:
    """Create the ChromaDB client lazily, not while FastAPI imports the app."""
    import chromadb

    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_path)


@lru_cache(maxsize=1)
def _get_embedding_function() -> Any:
    """Load the local embedding model only when a PDF workflow needs it."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # free, runs locally, ~80MB download on first use
    )


def _get_or_create_collection(session_id: str):
    return _get_chroma_client().get_or_create_collection(
        name=f"pdf_{session_id}",
        embedding_function=_get_embedding_function(),
    )


def _get_anthropic_client() -> anthropic.Anthropic:
    settings = get_settings()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks for RAG."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


async def ingest_pdf(file_bytes: bytes, filename: str) -> str:
    """
    Parse PDF → chunk → embed → store in ChromaDB.
    Returns a session_id the frontend uses for follow-up chat.
    """
    import pymupdf  # PyMuPDF

    session_id = str(uuid.uuid4())

    # Extract text from all pages
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    for page_num, page in enumerate(doc):
        pages_text.append({
            "page": page_num + 1,
            "text": page.get_text()
        })
    doc.close()

    full_text = "\n".join(p["text"] for p in pages_text)
    chunks = _chunk_text(full_text)

    if not chunks:
        raise ValueError("The uploaded PDF did not contain extractable text.")

    # Store chunks in ChromaDB
    col = _get_or_create_collection(session_id)
    col.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))],
    )

    # Initialise empty chat history for this session
    _chat_history[session_id] = []

    return session_id


async def chat_with_pdf(session_id: str, question: str) -> dict:
    """
    RAG: retrieve relevant chunks → build context → ask Claude → return answer.
    """
    if session_id not in _chat_history:
        raise ValueError(f"Session '{session_id}' not found. Please upload a PDF first.")

    # 1. Retrieve top-5 relevant chunks
    col = _get_or_create_collection(session_id)
    results = col.query(query_texts=[question], n_results=5)
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n---\n\n".join(docs)
    sources = [f"chunk {m['chunk_index']}" for m in metadatas]

    # 2. Build conversation history for Claude
    history = _chat_history[session_id]

    system_prompt = (
        "You are TriField AI, an expert research assistant specialising in "
        "aerospace structures, advanced materials, and textile engineering. "
        "Answer the user's question using ONLY the context extracted from their PDF. "
        "Always cite the relevant section. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"CONTEXT FROM PDF:\n{context}"
    )

    messages = history + [{"role": "user", "content": question}]

    # 3. Call Claude API
    client = _get_anthropic_client()
    response = client.messages.create(
        model="claude-haiku-4-5",   # cheapest model — great for Q&A
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    answer = response.content[0].text

    # 4. Update history
    _chat_history[session_id].append({"role": "user", "content": question})
    _chat_history[session_id].append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "sources": sources,
        "history": _chat_history[session_id],
    }


async def extract_properties(session_id: str) -> list[dict[str, Any]]:
    """
    Ask Claude to extract a structured table of material/mechanical properties
    from the ingested PDF chunks.
    """
    if session_id not in _chat_history:
        raise ValueError(f"Session '{session_id}' not found.")

    col = _get_or_create_collection(session_id)

    # Pull chunks likely to contain property tables
    results = col.query(
        query_texts=["mechanical properties tensile strength modulus fibre volume fraction"],
        n_results=8,
    )
    context = "\n\n---\n\n".join(results["documents"][0])

    prompt = (
        "You are a materials science data extractor. "
        "From the text below, extract ALL material/mechanical properties you can find. "
        "Return ONLY a JSON array. Each element must have: "
        "property_name, value, unit, test_standard (if mentioned), page_ref (if mentioned). "
        "Examples of properties: tensile strength, Young's modulus, flexural strength, "
        "fibre volume fraction, void content, density, impact strength, hardness.\n\n"
        f"TEXT:\n{context}\n\n"
        "Return ONLY the JSON array, no markdown, no explanation."
    )

    client = _get_anthropic_client()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        properties = json.loads(raw)
    except json.JSONDecodeError:
        properties = []

    return properties
