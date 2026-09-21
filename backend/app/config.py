"""Application settings with OpenAI or local mock fallback."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Agentic RAG Knowledge Assistant"
    api_prefix: str = "/api"
    cors_origins: list[str] = ["http://127.0.0.1:5284", "http://localhost:5284"]

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    data_dir: Path = BASE_DIR / "data"
    docs_dir: Path = BASE_DIR / "data" / "documents"
    index_dir: Path = BASE_DIR / "data" / "faiss_index"
    eval_dir: Path = BASE_DIR / "data" / "evals"
    sample_docs_dir: Path = BASE_DIR / "sample_docs"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    return settings
