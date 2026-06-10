"""
Supabase Storage helpers for PDF files.

Bucket: "pdfs"  (private — no public URLs)
Path convention: pdfs/{session_id}/{filename}

All functions are synchronous wrappers around supabase-py's storage API.
They are called from async FastAPI routes via asyncio.to_thread().
"""
import asyncio
import logging
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)

BUCKET = "pdfs"


def _storage():
    from db.client import get_storage
    return get_storage().from_(BUCKET)


# ── Sync helpers (run in thread pool from async callers) ────────────────────

def _upload_sync(session_id: str, filename: str, file_bytes: bytes) -> str:
    """
    Upload PDF bytes to Supabase Storage.
    Returns the storage object path (not a public URL).
    """
    path = f"{session_id}/{filename}"
    _storage().upload(
        path=path,
        file=file_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )
    logger.info(f"Storage: uploaded {path} ({len(file_bytes):,} bytes)")
    return path


def _download_sync(storage_path: str) -> bytes:
    """Download a PDF from Storage. Returns raw bytes."""
    data = _storage().download(storage_path)
    logger.info(f"Storage: downloaded {storage_path} ({len(data):,} bytes)")
    return data


def _delete_sync(storage_path: str):
    """Delete a PDF from Storage."""
    _storage().remove([storage_path])
    logger.info(f"Storage: deleted {storage_path}")


def _get_signed_url_sync(storage_path: str, expires_in: int = 3600) -> str:
    """Generate a temporary signed download URL (default: 1 hour)."""
    res = _storage().create_signed_url(storage_path, expires_in)
    return res.get("signedURL") or res.get("signedUrl", "")


# ── Async wrappers (call from FastAPI route handlers) ───────────────────────

async def upload_pdf(session_id: str, filename: str, file_bytes: bytes) -> str:
    return await asyncio.to_thread(_upload_sync, session_id, filename, file_bytes)


async def download_pdf(storage_path: str) -> bytes:
    return await asyncio.to_thread(_download_sync, storage_path)


async def delete_pdf(storage_path: str):
    await asyncio.to_thread(_delete_sync, storage_path)


async def get_signed_url(storage_path: str, expires_in: int = 3600) -> str:
    return await asyncio.to_thread(_get_signed_url_sync, storage_path, expires_in)
