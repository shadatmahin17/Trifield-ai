import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    groq_api_key:      str = os.getenv("GROQ_API_KEY", "")

    # Vector DB
    qdrant_url:        str = os.getenv("QDRANT_URL", "")          # cloud
    qdrant_api_key:    str = os.getenv("QDRANT_API_KEY", "")
    qdrant_local_path: str = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_db")

    # App
    app_env:           str = os.getenv("APP_ENV", "development")
    max_pdf_size_mb:   int = 20
    chroma_path:       str = "./chroma_db"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
