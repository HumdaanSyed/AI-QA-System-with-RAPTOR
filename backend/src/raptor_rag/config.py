from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    data_dir: Path = Path("data")
    # A URL ("http://localhost:6333") or ":memory:" for an in-process store.
    qdrant_location: str = "http://localhost:6333"
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    answer_model: str = "gpt-6.1-sol"
    summary_model: str = "gpt-6-luna"  # RAPTOR makes many summary calls; a small model is enough
    retrieve_candidates: int = 30  # hybrid search results passed to the reranker
    retrieve_top_n: int = 6  # reranked sources given to the LLM
    cors_origins: list[str] = ["http://localhost:3000"]  # the Next.js frontend
    max_upload_mb: int = 50
    chunk_chars: int = 1600  # ~400 tokens
