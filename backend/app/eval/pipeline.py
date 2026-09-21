"""Retrieval and answer-relevance evaluation with iterative improvement tracking."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from app.agent.graph import run_agent
from app.config import get_settings
from app.rag.store import get_store


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_overlap(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def cosine_relevance(query: str, answer: str) -> float:
    store = get_store()
    emb = store.vectorstore.embedding_function
    q = emb.embed_query(query)
    a = emb.embed_query(answer)
    dot = sum(x * y for x, y in zip(q, a))
    nq = math.sqrt(sum(x * x for x in q)) or 1.0
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    return max(0.0, min(1.0, (dot / (nq * na) + 1) / 2))


def retrieval_hit_rate(question: str, expected_keywords: list[str], k: int = 4) -> float:
    if not expected_keywords:
        return 0.0
    docs = get_store().similarity_search(question, k=k)
    blob = " ".join(d.page_content.lower() for d in docs)
    hits = sum(1 for kw in expected_keywords if kw.lower() in blob)
    return hits / len(expected_keywords)


@dataclass
class EvalCase:
    id: str
    question: str
    expected_keywords: list[str]
    reference_answer: str = ""


# Benchmark questions aligned with curated domain docs + Wikipedia AI coverage.
DEFAULT_CASES: list[EvalCase] = [
    EvalCase(
        id="rag-basics",
        question="What is retrieval augmented generation and why use it?",
        expected_keywords=["retrieval", "generation", "hallucination", "documents"],
        reference_answer=(
            "RAG retrieves relevant documents and conditions generation on that evidence to reduce hallucinations."
        ),
    ),
    EvalCase(
        id="faiss-role",
        question="How does FAISS help in a knowledge assistant?",
        expected_keywords=["faiss", "similarity", "vector", "index"],
        reference_answer="FAISS stores embedding vectors and supports fast similarity search over document chunks.",
    ),
    EvalCase(
        id="agentic-flow",
        question="What steps does an agentic RAG assistant take to answer multi-step questions?",
        expected_keywords=["plan", "tool", "retrieve", "synthesize"],
        reference_answer=(
            "It plans, retrieves with tools, optionally looks up terms, then synthesizes a grounded answer."
        ),
    ),
    EvalCase(
        id="eval-loop",
        question="How can evaluation improve answer relevance over time?",
        expected_keywords=["relevance", "evaluation", "metrics", "improve"],
        reference_answer=(
            "Offline eval measures retrieval hit-rate and answer relevance, then tuning retrieval improves quality."
        ),
    ),
    EvalCase(
        id="transformers",
        question="What is a transformer architecture in deep learning and why is attention important?",
        expected_keywords=["transformer", "attention", "token", "neural"],
        reference_answer=(
            "Transformers are neural architectures based on multi-head attention that contextualize tokens in parallel."
        ),
    ),
    EvalCase(
        id="llms",
        question="What is a large language model?",
        expected_keywords=["language", "model", "text", "training"],
        reference_answer=(
            "A large language model is a neural model trained on large text corpora to predict and generate language."
        ),
    ),
]


@dataclass
class CaseResult:
    case_id: str
    question: str
    answer: str
    retrieval_hit_rate: float
    answer_overlap: float
    semantic_relevance: float
    composite_score: float
    sources: list[dict[str, Any]]


def _keyword_baseline_answer(case: EvalCase) -> str:
    """Weaker baseline: top-1 chunk pasted with minimal synthesis (no agentic planning/tools)."""
    docs = get_store().similarity_search(case.question, k=1)
    if not docs:
        return "No evidence found."
    chunk = docs[0].page_content[:400]
    return f"Possibly related note: {chunk}"


def _score_answer(case: EvalCase, answer: str, sources: list[dict[str, Any]], hit_k: int) -> CaseResult:
    hit = retrieval_hit_rate(case.question, case.expected_keywords, k=hit_k)
    overlap = lexical_overlap(answer, case.reference_answer or " ".join(case.expected_keywords))
    kw_cov = sum(1 for kw in case.expected_keywords if kw.lower() in answer.lower()) / max(
        len(case.expected_keywords), 1
    )
    answer_overlap = max(overlap, kw_cov * 0.7)
    # Correct grounded refusals should not be rewarded as high-quality answers.
    refusal_markers = ("do not have", "cannot be answered", "does not contain", "no supporting evidence", "missing")
    if any(m in answer.lower() for m in refusal_markers) and kw_cov < 0.5:
        answer_overlap = min(answer_overlap, 0.15)
    semantic = cosine_relevance(case.question, answer)
    composite = 0.4 * hit + 0.35 * answer_overlap + 0.25 * semantic
    return CaseResult(
        case_id=case.id,
        question=case.question,
        answer=answer,
        retrieval_hit_rate=round(hit, 4),
        answer_overlap=round(answer_overlap, 4),
        semantic_relevance=round(semantic, 4),
        composite_score=round(composite, 4),
        sources=sources,
    )


def run_evaluation(persist: bool = True) -> dict[str, Any]:
    settings = get_settings()
    store = get_store()
    if store.document_count() == 0:
        store.seed_samples_if_empty()

    baseline_cases: list[CaseResult] = []
    for case in DEFAULT_CASES:
        answer = _keyword_baseline_answer(case)
        baseline_cases.append(_score_answer(case, answer, sources=[], hit_k=1))
    baseline_score = sum(c.composite_score for c in baseline_cases) / len(baseline_cases)

    improved_cases: list[CaseResult] = []
    for case in DEFAULT_CASES:
        result = run_agent(case.question)
        improved_cases.append(
            _score_answer(case, result["answer"], sources=result.get("sources", []), hit_k=settings.top_k)
        )
    improved_score = sum(c.composite_score for c in improved_cases) / len(improved_cases)
    improvement_pct = ((improved_score - baseline_score) / max(baseline_score, 1e-6)) * 100

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": get_settings().llm_provider,
        "baseline_score": round(baseline_score, 4),
        "improved_score": round(improved_score, 4),
        "improvement_pct": round(improvement_pct, 2),
        "notes": (
            "Baseline = top-1 chunk dump without agentic planning/tools. "
            "Improved = LangGraph plan → hybrid retrieve/tools → grounded synthesis. "
            "Composite = 0.4*retrieval_hit + 0.35*answer_overlap + 0.25*semantic_relevance."
        ),
        "cases": [asdict(c) for c in improved_cases],
        "baseline_cases": [asdict(c) for c in baseline_cases],
    }

    if persist:
        out = settings.eval_dir / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        latest = settings.eval_dir / "latest.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        latest.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(out)

    return report


def latest_evaluation() -> dict[str, Any] | None:
    path = get_settings().eval_dir / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
