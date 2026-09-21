"""Backend smoke tests for Groundline RAG assistant."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture(autouse=True)
def fresh_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    docs = data / "documents"
    index = data / "faiss_index"
    evals = data / "evals"
    for p in (docs, index, evals):
        p.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    monkeypatch.setattr(settings, "data_dir", data)
    monkeypatch.setattr(settings, "docs_dir", docs)
    monkeypatch.setattr(settings, "index_dir", index)
    monkeypatch.setattr(settings, "eval_dir", evals)
    get_settings.cache_clear()

    # Reset singleton store
    import app.rag.store as store_mod
    import app.agent.graph as graph_mod

    store_mod._store = None
    graph_mod._graph = None

    # Copy samples into sample_docs location used by settings
    sample_src = Path(__file__).resolve().parents[1] / "sample_docs"
    monkeypatch.setattr(settings, "sample_docs_dir", sample_src)

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
    chat = client.post("/api/chat", json={"question": "What is RAG and why use it?"})
    assert chat.status_code == 200
    body = chat.json()
    assert body["sources"]
    assert body["tool_trace"]
    assert "plan_step" in body["steps"]
    assert len(body["answer"]) > 40


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
    assert report["improvement_pct"] >= 20
