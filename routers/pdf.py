from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.pdf_service import ingest_pdf, chat_with_pdf, extract_properties
from models.schemas import ChatRequest, ChatResponse, PropertyExtractionResponse

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF. Returns a session_id to use for chat and property extraction.

    - Max size: 20MB
    - Supported: any research paper PDF
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > 20:
        raise HTTPException(status_code=400, detail=f"File too large ({size_mb:.1f}MB). Max is 20MB.")

    try:
        session_id = await ingest_pdf(contents, file.filename)
        return {
            "session_id": session_id,
            "filename":   file.filename,
            "size_mb":    round(size_mb, 2),
            "message":    "PDF uploaded and indexed successfully. Use session_id to chat.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Ask a question about an uploaded PDF.
    Requires session_id from /api/pdf/upload.

    Body:
      { "session_id": "...", "question": "What fibre volume fraction was used?" }
    """
    try:
        result = await chat_with_pdf(req.session_id, req.question)
        return ChatResponse(
            session_id=req.session_id,
            answer=result["answer"],
            sources=result["sources"],
            history=result["history"],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extract-properties/{session_id}", response_model=PropertyExtractionResponse)
async def extract(session_id: str):
    """
    Auto-extract mechanical/material properties from an uploaded PDF.
    Returns a structured table: property name, value, unit, test standard.

    Example:
      GET /api/pdf/extract-properties/your-session-id
    """
    try:
        properties = await extract_properties(session_id)
        return PropertyExtractionResponse(
            session_id=session_id,
            properties=properties,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
