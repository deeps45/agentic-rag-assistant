"""Unit tests for conversational memory and anti-hallucination helpers."""

from __future__ import annotations

from app.agent.graph import (
    _extract_terms,
    _filter_retrieved,
    _is_followup,
    _lexical_support,
    _resolve_question_with_memory,
)


def test_extract_terms_prefers_faiss_acronym():
    terms = _extract_terms("How does FAISS help in a knowledge assistant?")
    assert terms
    assert terms[0].upper() == "FAISS"


def test_followup_detection_does_not_flag_standalone_topic():
    assert not _is_followup("How does FAISS help in a knowledge assistant?")
    assert not _is_followup("What is retrieval augmented generation and why use it?")
    assert _is_followup("Why does that matter?")
    assert _is_followup("Tell me more")
    assert _is_followup("and BM25?")


def test_memory_resolves_followup_not_new_topic():
    history = [
        {"role": "user", "content": "How does FAISS help in a knowledge assistant?"},
        {"role": "assistant", "content": "FAISS indexes embeddings for fast nearest-neighbor search [1]."},
    ]
    resolved, used = _resolve_question_with_memory("Why does that matter for RAG?", history)
    assert used is True
    assert "FAISS" in resolved

    standalone, used2 = _resolve_question_with_memory(
        "What is retrieval augmented generation and why use it?",
        history,
    )
    assert used2 is False
    assert standalone.startswith("What is retrieval")


def test_lexical_support_and_filter():
    retrieved = [
        {
            "content": "FAISS is a library for efficient similarity search over embedding vectors.",
            "filename": "faiss_vector_search.txt",
            "score": 0.2,
        },
        {
            "content": "Unrelated astronomy notes about nebulae and galaxies.",
            "filename": "hf_rag_mini_wiki_000.txt",
            "score": 0.9,
        },
    ]
    good = "FAISS speeds up similarity search over embedding vectors [1]."
    bad = "Quantum flux capacitors unlock teleportation in deep space travel."
    assert _lexical_support(good, retrieved) >= 0.4
    assert _lexical_support(bad, retrieved) < 0.2

    filtered = _filter_retrieved(
        [
            {"content": "a", "filename": "faiss_vector_search.txt", "score": 0.25},
            {"content": "b", "filename": "hf_rag_mini_wiki_000.txt", "score": 0.4},
            {"content": "c", "filename": "hf_rag_mini_wiki_001.txt", "score": 1.8},
        ],
        "How does FAISS help?",
        limit=4,
    )
    assert filtered
    assert filtered[0]["filename"] == "faiss_vector_search.txt"
    assert all(float(x["score"]) < 1.5 for x in filtered)
