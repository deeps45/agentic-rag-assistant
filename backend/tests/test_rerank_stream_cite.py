"""Tests for cross-encoder re-rank, citation faithfulness, and streaming."""

from __future__ import annotations

import json

from app.eval.pipeline import citation_faithfulness, is_refusal
from app.rag.rerank import _interaction_score, rerank_documents, reranker_backend
from langchain_core.documents import Document


def test_interaction_rerank_prefers_on_topic():
    docs = [
        (Document(page_content="The capital of France is Paris."), 0.4),
        (Document(page_content="FAISS is a library for efficient similarity search over vectors."), 0.5),
        (Document(page_content="Toasters heat bread using resistive coils."), 0.45),
    ]
    ranked = rerank_documents("How does FAISS help vector search?", docs, top_n=2)
    assert "FAISS" in ranked[0][0].page_content
    assert reranker_backend() in {"flashrank", "interaction"}


def test_interaction_score_monotonic():
    q = "retrieval augmented generation"
    good = _interaction_score(q, "Retrieval augmented generation reduces hallucination with documents.")
    bad = _interaction_score(q, "Mars rovers collect soil samples near craters.")
    assert good > bad


def test_citation_faithfulness_rewards_supported_cites():
    sources = [
        {
            "id": 1,
            "snippet": "FAISS stores embedding vectors for fast similarity search.",
            "content": "FAISS stores embedding vectors for fast similarity search over chunks.",
        },
        {
            "id": 2,
            "snippet": "RAG retrieves documents before generation.",
            "content": "RAG retrieves documents before generation to ground answers.",
        },
    ]
    good = (
        "FAISS enables fast similarity search over embedding vectors [1]. "
        "RAG retrieves documents before generation [2]."
    )
    metrics = citation_faithfulness(good, sources)
    assert metrics["citation_validity"] == 1.0
    assert metrics["citation_coverage"] >= 0.9
    assert metrics["citation_score"] >= 0.5

    bad = "Atlantis has a navy of twelve fleets commanded by Poseidon [9]."
    bad_m = citation_faithfulness(bad, sources)
    assert bad_m["citation_validity"] == 0.0


def test_refusal_detection_and_unanswerable_cite_score():
    refuse = "I do not have sufficiently relevant evidence in the knowledge base to answer that without guessing."
    assert is_refusal(refuse)
    metrics = citation_faithfulness(refuse, [], expect_citations=False)
    assert metrics["citation_score"] == 1.0


def test_chat_stream_sse(client):
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "How does FAISS help in a knowledge assistant?"},
    ) as res:
        assert res.status_code == 200
        body = "".join(res.iter_text())
    assert "event: plan" in body
    assert "event: token" in body or "event: replace" in body
    assert "event: final" in body
    # Parse final payload
    blocks = body.split("\n\n")
    final = None
    for block in blocks:
        if block.startswith("event: final"):
            data_line = [ln for ln in block.split("\n") if ln.startswith("data:")][0]
            final = json.loads(data_line[5:].strip())
    assert final is not None
    assert final["answer"]
    assert final["sources"]
    assert "plan_step" in final["steps"]
