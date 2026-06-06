import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    groq_api_key:      str = os.getenv("GROQ_API_KEY", "")
    app_env:           str = os.getenv("APP_ENV", "development")
    max_pdf_size_mb:   int = 20
    chroma_path:       str = "./chroma_db"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
