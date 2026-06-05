import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Anthropic (Claude)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # App
    app_env: str = os.getenv("APP_ENV", "development")
    max_pdf_size_mb: int = 20
    max_search_results: int = 10

    # ChromaDB persist path
    chroma_path: str = "./chroma_db"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
