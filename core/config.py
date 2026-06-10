import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    groq_api_key:      str = os.getenv("GROQ_API_KEY",      "")

    # ── Supabase ─────────────────────────────────────────────────────────
    # supabase_url:         Project URL  (Settings → API → Project URL)
    # supabase_service_key: service_role secret (Settings → API → service_role)
    #                       ⚠ Never expose the service_role key to the browser.
    # supabase_anon_key:    anon/public key — safe for client-side if needed.
    supabase_url:         str = os.getenv("SUPABASE_URL",         "")
    supabase_service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")
    supabase_anon_key:    str = os.getenv("SUPABASE_ANON_KEY",    "")

    # ── Vector DB (Qdrant) ────────────────────────────────────────────────
    qdrant_url:        str = os.getenv("QDRANT_URL",        "")   # cloud endpoint
    qdrant_api_key:    str = os.getenv("QDRANT_API_KEY",    "")
    qdrant_local_path: str = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_db")

    # ── App ───────────────────────────────────────────────────────────────
    app_env:         str = os.getenv("APP_ENV", "development")
    max_pdf_size_mb: int = 20
    # NOTE: chroma_path removed — project migrated to Qdrant in v2.

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
