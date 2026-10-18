from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    data_dir: Path = Path("data")
    # A URL ("http://localhost:6333") or ":memory:" for an in-process store.
    qdrant_location: str = "http://localhost:6333"
    embed_model: str = "BAAI/bge-m3"
    max_upload_mb: int = 50
    chunk_chars: int = 1600  # ~400 tokens
