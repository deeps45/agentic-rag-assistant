"""Backend smoke tests for Groundline RAG assistant."""

from __future__ import annotations


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
    assert "citation_score" in report
    assert report["case_count"] >= 6
