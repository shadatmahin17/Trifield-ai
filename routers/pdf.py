import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from rag.pipeline import ingest_pdf, chat_with_pdf, extract_properties
from analytics.tracker import get_tracker
from models.schemas import ChatRequest, ChatResponse, PropertyExtractionResponse

router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    contents = await file.read()
    size_mb  = len(contents) / (1024 * 1024)
    if size_mb > 20:
        raise HTTPException(400, f"File too large ({size_mb:.1f}MB). Max is 20MB.")
    t0 = time.time()
    try:
        session_id = await ingest_pdf(contents, file.filename)
        latency    = round((time.time() - t0) * 1000, 1)
        get_tracker().record_pdf(file.filename, session_id, 0, latency)
        return {"session_id": session_id, "filename": file.filename,
                "size_mb": round(size_mb, 2),
                "message": "PDF indexed via Qdrant vector search. Use session_id to chat."}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await chat_with_pdf(req.session_id, req.question)
        return ChatResponse(session_id=req.session_id, answer=result["answer"],
                            sources=result["sources"], history=result["history"])
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/extract-properties/{session_id}", response_model=PropertyExtractionResponse)
async def extract(session_id: str):
    try:
        props = await extract_properties(session_id)
        return PropertyExtractionResponse(session_id=session_id, properties=props)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
