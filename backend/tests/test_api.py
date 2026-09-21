"""Backend smoke tests for Groundline RAG assistant."""

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
    for p in (docs, index, evals):
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

    import app.agent.graph as graph_mod
    import app.rag.store as store_mod

    store_mod._store = None
    graph_mod._graph = None

    yield
    store_mod._store = None
    graph_mod._graph = None
    get_settings.cache_clear()


@pytest.fixture
def client(fresh_data):
    return TestClient(app)


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_seed_and_chat(client):
    status = client.get("/api/status").json()
    assert status["documents"] >= 4
    assert status["mode"] == "mock"
    chat = client.post("/api/chat", json={"question": "What is RAG and why use it?"})
    assert chat.status_code == 200
    body = chat.json()
    assert body["sources"]
    assert body["tool_trace"]
    assert "plan_step" in body["steps"]
    assert len(body["answer"]) > 40
    assert "grounded" in body
    assert "confidence" in body
    assert "memory_used" in body


def test_chat_memory_followup(client):
    first = client.post(
        "/api/chat",
        json={"question": "How does FAISS help in a knowledge assistant?"},
    )
    assert first.status_code == 200
    follow = client.post(
        "/api/chat",
        json={
            "question": "Why does that matter for retrieval?",
            "history": [
                {"role": "user", "content": "How does FAISS help in a knowledge assistant?"},
                {"role": "assistant", "content": first.json()["answer"]},
            ],
        },
    )
    assert follow.status_code == 200
    body = follow.json()
    assert body["memory_used"] is True
    assert body["tool_trace"]
    # Prefer domain FAISS/RAG sources when available.
    filenames = " ".join(s["filename"].lower() for s in body["sources"])
    assert ("faiss" in filenames) or ("rag" in filenames) or body["sources"]


def test_ingest_and_eval(client):
    up = client.post(
        "/api/documents/text",
        json={"title": "Extra", "text": "Evaluation metrics include relevance and retrieval hit-rate."},
    )
    assert up.status_code == 200
    ev = client.post("/api/eval/run")
    assert ev.status_code == 200
    report = ev.json()
    assert report["improved_score"] > report["baseline_score"]
    assert report["improvement_pct"] > 0
