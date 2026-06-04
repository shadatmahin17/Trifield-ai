from fastapi import APIRouter, File, HTTPException, UploadFile

from core.config import get_settings
from models.schemas import ChatRequest, ChatResponse, PropertyExtractionResponse
from services.pdf_service import chat_with_pdf, extract_properties, ingest_pdf

router = APIRouter()


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    settings = get_settings()
    file_bytes = await file.read()
    max_bytes = settings.max_pdf_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"PDF is too large. Maximum size is {settings.max_pdf_size_mb} MB.",
        )

    try:
        session_id = await ingest_pdf(file_bytes, file.filename or "uploaded.pdf")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {exc}") from exc

    return {"session_id": session_id}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = await chat_with_pdf(request.session_id, request.question)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF chat failed: {exc}") from exc

    return ChatResponse(session_id=request.session_id, **result)


@router.get("/{session_id}/properties", response_model=PropertyExtractionResponse)
async def properties(session_id: str) -> PropertyExtractionResponse:
    try:
        properties_data = await extract_properties(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Property extraction failed: {exc}") from exc

    return PropertyExtractionResponse(session_id=session_id, properties=properties_data)
