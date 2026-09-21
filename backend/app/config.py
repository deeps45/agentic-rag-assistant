"""Application settings: TAMU Chat API, OpenAI, or local mock."""

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

    # TAMU System AI Chat (preferred when set)
    tamus_ai_chat_api_key: str | None = None
    tamus_ai_chat_api_endpoint: str = "https://chat-api.tamu.ai"
    tamus_chat_model: str = "protected.gemini-2.5-flash-lite"
    tamus_embedding_model: str = "protected.text-embedding-3-small"

    # Optional OpenAI fallback
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    data_dir: Path = BASE_DIR / "data"
    docs_dir: Path = BASE_DIR / "data" / "documents"
    index_dir: Path = BASE_DIR / "data" / "faiss_index"
    eval_dir: Path = BASE_DIR / "data" / "evals"
    sample_docs_dir: Path = BASE_DIR / "sample_docs"
    corpus_dir: Path = BASE_DIR / "corpus" / "wikipedia"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 6
    # FAISS L2 distance cutoff; weaker matches are dropped before synthesis.
    max_retrieval_distance: float = 1.35

    @property
    def use_tamus(self) -> bool:
        return bool(self.tamus_ai_chat_api_key)

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key) and not self.use_tamus

    @property
    def llm_provider(self) -> str:
        if self.use_tamus:
            return "tamus"
        if self.openai_api_key:
            return "openai"
        return "mock"

    @property
    def tamus_api_base(self) -> str:
        return self.tamus_ai_chat_api_endpoint.rstrip("/") + "/api"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.eval_dir.mkdir(parents=True, exist_ok=True)
    return settings
