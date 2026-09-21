"""LangGraph agentic RAG: plan → retrieve/tools → synthesize → ground-check.

Includes conversational memory, source filtering, and anti-hallucination gates.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agent.llm import get_chat_model, llm_mode
from app.agent.tools import run_tool
from app.config import get_settings
from app.rag.store import get_store

DOMAIN_FILES = {
    "rag_fundamentals.txt",
    "faiss_vector_search.txt",
    "agentic_rag_langgraph.txt",
    "evaluation_quality.txt",
}

DOMAIN_TERMS = {
    "faiss",
    "rag",
    "retrieval",
    "augmented",
    "embedding",
    "embeddings",
    "vector",
    "vectors",
    "langgraph",
    "langchain",
    "agentic",
    "hallucination",
    "bm25",
}


class AgentState(TypedDict):
    question: str
    history: list[dict[str, str]]
    messages: Annotated[list, add_messages]
    plan: str
    tool_trace: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    grounded: bool
    confidence: float
    memory_used: bool
    resolved_question: str


def _extract_terms(question: str) -> list[str]:
    """Prefer rare/technical terms for define_term (e.g. FAISS), skip filler words."""
    stop = {
        "what", "who", "when", "where", "why", "how", "does", "do", "is", "are",
        "the", "a", "an", "of", "to", "in", "for", "and", "or", "on", "with",
        "about", "can", "you", "me", "please", "explain", "describe", "tell",
        "help", "helps", "knowledge", "assistant", "system", "used", "use",
        "using", "into", "from", "that", "this", "these", "those", "their",
        "steps", "step", "multi", "answer", "answers", "question", "questions",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", question)
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for t in tokens:
        low = t.lower()
        if low in stop or low in seen:
            continue
        seen.add(low)
        score = 0
        if t.isupper() and len(t) >= 3:  # acronyms like FAISS
            score += 5
        elif t[:1].isupper() and t[1:].islower() and low in DOMAIN_TERMS:
            score += 3
        if low in DOMAIN_TERMS:
            score += 4
        if len(low) >= 5:
            score += 1
        if any(ch.isdigit() for ch in low):
            score += 1
        scored.append((score, t))
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    # Only keep reasonably technical terms
    return [t for s, t in scored if s >= 1][:2]


def _format_history(history: list[dict[str, str]], limit: int = 8) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for turn in history[-limit:]:
        role = (turn.get("role") or "user").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content[:700]}")
    return "\n".join(lines)


_FOLLOWUP_PHRASES = re.compile(
    r"\b("
    r"tell me more|go on|continue|elaborate|expand|clarify|how so|and then|what about|"
    r"the (first|second|third|last|previous) (one|point|step|source)"
    r")\b",
    re.I,
)
_FOLLOWUP_ANAPHORA = re.compile(
    r"\b("
    r"(why|how) (does|do|is|are|did|was|were|can|could|would|should) (that|this|it|they|those|them)|"
    r"(and|also|about) (that|this|it|them|those)|"
    r"(same|previous|above) (one|thing|topic|question|answer|point)?"
    r")\b",
    re.I,
)


def _is_followup(question: str) -> bool:
    """Detect referential follow-ups without treating every short question as one."""
    q = question.lower().strip()
    if _FOLLOWUP_PHRASES.search(q) or _FOLLOWUP_ANAPHORA.search(q):
        return True
    if re.match(r"^(and|also|same|more|again)\b", q):
        return True
    words = q.split()
    # Ultra-short replies that are not self-contained topic asks.
    if len(words) <= 5 and not re.search(
        r"\b(what|who|when|where|how|why|explain|define|describe)\b",
        q,
    ):
        return True
    if len(words) <= 4 and re.match(r"^(why|how)\b", q):
        return True
    return False


def _resolve_question_with_memory(question: str, history: list[dict[str, str]]) -> tuple[str, bool]:
    """Expand follow-ups using prior turns so retrieval stays on-topic."""
    hist = _format_history(history, limit=6)
    if not hist or not _is_followup(question):
        return question, False

    # Prefer the last topical user question (skip previous follow-ups when possible).
    last_user = ""
    for turn in reversed(history):
        if (turn.get("role") or "").lower() != "user":
            continue
        content = (turn.get("content") or "").strip()
        if not content or content.lower() == question.lower():
            continue
        if not _is_followup(content) or not last_user:
            last_user = content
            if not _is_followup(content):
                break
    if last_user:
        resolved = (
            f"{question}\n\n"
            f"(Follow-up referring to prior question: {last_user})"
        )
        return resolved, True

    return f"{question}\n\nConversation context:\n{hist}", True


def _lexical_support(answer: str, retrieved: list[dict[str, Any]]) -> float:
    """Fraction of contentful answer tokens attested in retrieved passages."""
    corpus = " ".join(str(r.get("content") or "").lower() for r in retrieved)
    if not corpus.strip() or not answer.strip():
        return 0.0
    stop = {
        "that", "this", "with", "from", "have", "been", "were", "will", "would",
        "could", "should", "about", "into", "your", "their", "there", "which",
        "while", "where", "when", "what", "based", "using", "used", "also",
        "such", "than", "then", "them", "these", "those", "only", "over",
        "source", "sources", "context", "according", "however", "therefore",
    }
    tokens = [
        t for t in re.findall(r"[a-z0-9]{4,}", answer.lower())
        if t not in stop
    ]
    if not tokens:
        return 1.0
    supported = sum(1 for t in set(tokens) if t in corpus)
    return supported / max(len(set(tokens)), 1)


def _domain_boost(filename: str, question: str) -> float:
    """Lower distance (better) for curated domain docs on domain questions."""
    q = question.lower()
    fn = filename.lower()
    if filename in DOMAIN_FILES and any(t in q for t in DOMAIN_TERMS):
        return -0.18
    if filename in DOMAIN_FILES:
        return -0.05
    if fn.startswith("hf_rag_mini_") and any(t in q for t in DOMAIN_TERMS):
        return 0.12  # demote general wiki on specialized RAG questions
    return 0.0


def _filter_retrieved(items: list[dict[str, Any]], question: str, limit: int = 4) -> list[dict[str, Any]]:
    """Keep strongest, on-topic sources only."""
    if not items:
        return []
    max_d = get_settings().max_retrieval_distance
    ranked = sorted(items, key=lambda x: float(x["score"]))
    kept: list[dict[str, Any]] = []
    for item in ranked:
        score = float(item["score"])
        if score > max_d:
            continue
        # Drop far outliers relative to the best hit.
        if kept and score > float(kept[0]["score"]) + 0.55:
            continue
        kept.append(item)
        if len(kept) >= limit:
            break

    # If domain question and we have any domain hit, prefer those first.
    q = question.lower()
    if any(t in q for t in DOMAIN_TERMS):
        domain_hits = [x for x in kept if x.get("filename") in DOMAIN_FILES]
        other = [x for x in kept if x.get("filename") not in DOMAIN_FILES]
        if domain_hits:
            kept = (domain_hits + other)[:limit]
    return kept


def plan_node(state: AgentState) -> dict[str, Any]:
    llm = get_chat_model()
    hist = _format_history(state.get("history") or [])
    resolved, memory_used = _resolve_question_with_memory(
        state["question"], state.get("history") or []
    )
    memory_block = f"\nRecent conversation:\n{hist}\n" if hist else ""
    prompt = [
        SystemMessage(
            content=(
                "You are a planning module for an agentic RAG assistant. "
                "Decompose the user question into numbered retrieval and synthesis steps. "
                "If this is a follow-up, keep continuity with conversation memory. "
                "Output only the numbered plan — no preamble."
            )
        ),
        HumanMessage(content=f"{memory_block}Question: {resolved}"),
    ]
    plan = str(llm.invoke(prompt).content)
    return {
        "plan": plan,
        "memory_used": memory_used or bool(hist and _is_followup(state["question"])),
        "resolved_question": resolved,
        "messages": prompt + [HumanMessage(content=plan)],
    }


def retrieve_and_tools_node(state: AgentState) -> dict[str, Any]:
    settings = get_settings()
    question = state.get("resolved_question") or state["question"]
    trace: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []

    list_out = run_tool("list_knowledge_documents", {})
    trace.append({"tool": "list_knowledge_documents", "args": {}, "output": list_out[:500]})

    search_out = run_tool("semantic_search", {"query": question, "top_k": settings.top_k})
    trace.append(
        {
            "tool": "semantic_search",
            "args": {"query": question, "top_k": settings.top_k},
            "output": search_out[:1200],
        }
    )

    store = get_store()
    hits = store.similarity_search_with_score(question, k=max(settings.top_k, 8))
    max_distance = settings.max_retrieval_distance
    for doc, score in hits:
        dist = float(score) + _domain_boost(str(doc.metadata.get("filename", "")), question)
        if dist > max_distance:
            continue
        retrieved.append(
            {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "doc_id": doc.metadata.get("doc_id"),
                "score": dist,
            }
        )

    retrieved = _filter_retrieved(retrieved, question, limit=4)

    for term in _extract_terms(state["question"]):
        definition = run_tool("define_term", {"term": term})
        # Skip noisy / off-topic definitions from the trace highlight.
        if definition.lower().startswith("no ") or "no on-topic" in definition.lower():
            trace.append(
                {
                    "tool": "define_term",
                    "args": {"term": term},
                    "output": definition[:300],
                }
            )
            continue
        trace.append({"tool": "define_term", "args": {"term": term}, "output": definition[:500]})

    return {"tool_trace": trace, "retrieved": retrieved}


def synthesize_node(state: AgentState) -> dict[str, Any]:
    llm = get_chat_model()
    context_blocks = []
    sources = []
    for idx, item in enumerate(state.get("retrieved", []), start=1):
        context_blocks.append(
            f"[{idx}] Source: {item['filename']} (chunk {item['chunk_index']})\n{item['content']}"
        )
        sources.append(
            {
                "id": idx,
                "filename": item["filename"],
                "chunk_index": item["chunk_index"],
                "snippet": item["content"][:320],
                "score": item["score"],
                "doc_id": item.get("doc_id"),
            }
        )

    if not context_blocks:
        refusal = (
            "I do not have sufficiently relevant evidence in the knowledge base to answer that "
            "without guessing. Please rephrase, or ingest documents that cover this topic."
        )
        return {
            "answer": refusal,
            "sources": [],
            "grounded": False,
            "confidence": 0.0,
            "messages": [],
        }

    context = "\n\n--\n\n".join(context_blocks)
    hist = _format_history(state.get("history") or [])
    memory_block = f"\nConversation memory:\n{hist}\n" if hist else ""
    prompt = [
        SystemMessage(
            content=(
                "You are a grounded knowledge assistant.\n"
                "Hard rules:\n"
                "1) Answer ONLY using the Context passages below.\n"
                "2) Every non-trivial claim must include a citation like [1] or [2].\n"
                "3) Prefer the most relevant sources; do not pad with weakly related passages.\n"
                "4) If Context is incomplete, say what is missing instead of inventing facts.\n"
                "5) Do not use outside world knowledge.\n"
                "6) Keep answers concise and precise.\n"
                "7) For follow-ups, stay consistent with conversation memory while remaining grounded.\n\n"
                f"Plan:\n{state.get('plan', '')}\n"
                f"{memory_block}\n"
                f"Context:\n{context}"
            )
        ),
        HumanMessage(content=f"Question: {state['question']}"),
    ]
    answer = str(llm.invoke(prompt).content)
    return {"answer": answer, "sources": sources, "messages": prompt}


def ground_check_node(state: AgentState) -> dict[str, Any]:
    """Post-answer anti-hallucination check + cite-only source filtering."""
    answer = state.get("answer") or ""
    sources = state.get("sources") or []
    retrieved = state.get("retrieved") or []
    if not retrieved:
        return {"grounded": False, "confidence": 0.0, "sources": []}

    cited = set(int(x) for x in re.findall(r"\[(\d+)\]", answer))
    valid_ids = set(range(1, len(retrieved) + 1))
    citation_ok = bool(cited) and cited.issubset(valid_ids)
    support = _lexical_support(answer, retrieved)
    support_ok = support >= 0.22

    best = min((float(s["score"]) for s in retrieved), default=99.0)
    max_d = get_settings().max_retrieval_distance
    retrieval_conf = max(0.0, min(1.0, 1.0 - (best / max(max_d, 1e-6))))

    needs_rewrite = (not citation_ok) or (not support_ok)
    grounded = citation_ok and support_ok
    confidence = round(
        0.45 * retrieval_conf
        + 0.35 * (1.0 if citation_ok else 0.15)
        + 0.20 * min(1.0, support / 0.45),
        3,
    )

    if needs_rewrite:
        llm = get_chat_model()
        context = "\n\n".join(
            f"[{i}] {s['filename']}: {s['content'][:500]}" for i, s in enumerate(retrieved, start=1)
        )
        fix = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Rewrite the answer so every factual sentence cites Context with [n]. "
                        "Only use the provided Context — no outside knowledge. "
                        "If the Context does not support the question, refuse briefly."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n\nDraft:\n{answer}\n\nContext:\n{context}"
                    )
                ),
            ]
        )
        answer = str(fix.content)
        cited = set(int(x) for x in re.findall(r"\[(\d+)\]", answer))
        support = _lexical_support(answer, retrieved)
        citation_ok = bool(cited) and cited.issubset(valid_ids)
        support_ok = support >= 0.18
        grounded = citation_ok and support_ok
        confidence = round(
            0.45 * retrieval_conf
            + 0.35 * (1.0 if citation_ok else 0.12)
            + 0.20 * min(1.0, support / 0.45),
            3,
        )

    # Hard refuse when rewrite still looks unsupported / uncited.
    if not grounded and support < 0.12:
        answer = (
            "I could not verify enough support in the retrieved passages to answer confidently "
            "without guessing. Please rephrase or add documents that cover this topic."
        )
        cited = set()
        confidence = round(min(confidence, 0.2), 3)
        return {
            "answer": answer,
            "grounded": False,
            "confidence": confidence,
            "sources": sources[:2],
        }

    # Expose only cited sources (fallback to top 3 if model forgot citations after rewrite).
    if cited:
        filtered = [s for s in sources if s["id"] in cited]
    else:
        filtered = sources[:3]

    return {
        "answer": answer,
        "grounded": grounded,
        "confidence": confidence,
        "sources": filtered,
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan_step", plan_node)
    graph.add_node("retrieve_and_tools", retrieve_and_tools_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("ground_check", ground_check_node)
    graph.set_entry_point("plan_step")
    graph.add_edge("plan_step", "retrieve_and_tools")
    graph.add_edge("retrieve_and_tools", "synthesize")
    graph.add_edge("synthesize", "ground_check")
    graph.add_edge("ground_check", END)
    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph() -> None:
    global _graph
    _graph = None


def run_agent(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    # Always rebuild graph in long-running servers after code reloads.
    reset_graph()
    graph = get_graph()
    result = graph.invoke(
        {
            "question": question,
            "history": history or [],
            "messages": [],
            "plan": "",
            "tool_trace": [],
            "retrieved": [],
            "answer": "",
            "sources": [],
            "grounded": False,
            "confidence": 0.0,
            "memory_used": False,
            "resolved_question": question,
        }
    )
    return {
        "answer": result["answer"],
        "plan": result["plan"],
        "sources": result["sources"],
        "tool_trace": result["tool_trace"],
        "mode": llm_mode(),
        "steps": ["plan_step", "retrieve_and_tools", "synthesize", "ground_check"],
        "grounded": bool(result.get("grounded")),
        "confidence": float(result.get("confidence") or 0.0),
        "memory_used": bool(result.get("memory_used")),
    }
