"""Shared pytest fixtures for Groundline backend tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture(autouse=True)
def fresh_data(tmp_path, monkeypatch):
    # Force offline mock mode in tests (ignore developer .env keys).
    monkeypatch.setenv("TAMUS_AI_CHAT_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.delenv("TAMUS_AI_CHAT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    data = tmp_path / "data"
    docs = data / "documents"
    index = data / "faiss_index"
    evals = data / "evals"
    flashrank = data / "flashrank_cache"
    for p in (docs, index, evals, flashrank):
        p.mkdir(parents=True, exist_ok=True)

    get_settings.cache_clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "tamus_ai_chat_api_key", None)
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "data_dir", data)
    monkeypatch.setattr(settings, "docs_dir", docs)
    monkeypatch.setattr(settings, "index_dir", index)
    monkeypatch.setattr(settings, "eval_dir", evals)
    monkeypatch.setattr(settings, "sample_docs_dir", Path(__file__).resolve().parents[1] / "sample_docs")
    # Keep rerank on for coverage; FlashRank uses cache under data_dir.
    monkeypatch.setattr(settings, "rerank_enabled", True)

    import app.agent.graph as graph_mod
    import app.rag.rerank as rerank_mod
    import app.rag.store as store_mod

    store_mod._store = None
    graph_mod._graph = None
    rerank_mod._flashrank_ranker.cache_clear()

    yield
    store_mod._store = None
    graph_mod._graph = None
    rerank_mod._flashrank_ranker.cache_clear()
    get_settings.cache_clear()


@pytest.fixture
def client(fresh_data):
    return TestClient(app)
