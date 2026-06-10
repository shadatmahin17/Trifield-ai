"""
Supabase client — single import point for DB and Storage.

Provides:
  get_db()       → supabase.Client  (for table queries)
  get_storage()  → StorageClient    (shortcut to client.storage)
  db_execute()   → run raw SQL via the REST PostgREST interface

Connection is lazy-initialised on first call so the app starts even if
SUPABASE_URL / SUPABASE_SERVICE_KEY are not set (graceful degradation).
"""
import logging
from typing import Optional
from core.config import get_settings

logger = logging.getLogger(__name__)

_client = None   # supabase.Client singleton


def get_db():
    """
    Return the Supabase client.
    Uses the SERVICE ROLE key (bypasses RLS) so the backend has full access.
    Never expose this key to the frontend.
    """
    global _client
    if _client is not None:
        return _client

    s = get_settings()
    if not s.supabase_url or not s.supabase_service_key:
        raise RuntimeError(
            "Supabase not configured. "
            "Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables."
        )

    from supabase import create_client, Client
    _client = create_client(s.supabase_url, s.supabase_service_key)
    logger.info("Supabase client initialised")
    return _client


def get_storage():
    """Shortcut to the Supabase Storage API."""
    return get_db().storage


def is_configured() -> bool:
    """Check whether Supabase credentials are present — use for graceful degradation."""
    s = get_settings()
    return bool(s.supabase_url and s.supabase_service_key)
