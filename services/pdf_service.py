import uuid
import anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from core.config import get_settings

# In-memory chat history per session  {session_id: [{"role":..,"content":..}]}
_chat_history: dict[str, list[dict]] = {}

# ChromaDB client (persisted to disk)
_settings = get_settings()
_chroma_client = chromadb.PersistentClient(path=_settings.chroma_path)
_embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"   # free, runs locally, ~80MB download on first run
)


def _get_or_create_collection(session_id: str):
    return _chroma_client.get_or_create_collection(
        name=f"pdf_{session_id}",
        embedding_function=_embed_fn,
    )


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks for RAG."""
    words  = text.split()
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
    docs     = results["documents"][0]
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
    client = anthropic.Anthropic(api_key=_settings.anthropic_api_key)
    response = client.messages.create(
        model="claude-haiku-4-5",   # cheapest model — great for Q&A
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    answer = response.content[0].text

    # 4. Update history
    _chat_history[session_id].append({"role": "user",      "content": question})
    _chat_history[session_id].append({"role": "assistant", "content": answer})

    return {
        "answer":  answer,
        "sources": sources,
        "history": _chat_history[session_id],
    }


async def extract_properties(session_id: str) -> list[dict]:
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

    client = anthropic.Anthropic(api_key=_settings.anthropic_api_key)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        properties = json.loads(raw)
    except json.JSONDecodeError:
        properties = []

    return properties
