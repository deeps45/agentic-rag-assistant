"""LangGraph agentic RAG: plan → retrieve/tools → synthesize → ground-check.

Includes conversational memory and anti-hallucination gates.
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


def _extract_terms(question: str) -> list[str]:
    stop = {
        "what", "who", "when", "where", "why", "how", "does", "do", "is", "are",
        "the", "a", "an", "of", "to", "in", "for", "and", "or", "on", "with",
        "about", "can", "you", "me", "please", "explain", "describe", "tell",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", question)
    terms: list[str] = []
    for t in tokens:
        low = t.lower()
        if low not in stop and low not in terms:
            terms.append(t)
    return terms[:3]


def _format_history(history: list[dict[str, str]], limit: int = 6) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for turn in history[-limit:]:
        role = (turn.get("role") or "user").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content[:600]}")
    return "\n".join(lines)


def _resolved_question(question: str, history: list[dict[str, str]]) -> str:
    """Use recent memory so follow-ups like 'why is that?' stay searchable."""
    hist = _format_history(history, limit=4)
    if not hist:
        return question
    # Short / referential follow-ups benefit from prior context in retrieval.
    if len(question.split()) <= 12 or re.search(r"\b(it|that|this|they|those|previous|above)\b", question.lower()):
        return f"{question}\n\nConversation context:\n{hist}"
    return question


def plan_node(state: AgentState) -> dict[str, Any]:
    llm = get_chat_model()
    hist = _format_history(state.get("history") or [])
    memory_block = f"\nRecent conversation:\n{hist}\n" if hist else ""
    prompt = [
        SystemMessage(
            content=(
                "You are a planning module for an agentic RAG assistant. "
                "Decompose the user question into numbered retrieval and synthesis steps. "
                "Use conversation memory when the question is a follow-up. "
                "Output only the numbered plan — no preamble."
            )
        ),
        HumanMessage(content=f"{memory_block}Question: {state['question']}"),
    ]
    plan = str(llm.invoke(prompt).content)
    return {
        "plan": plan,
        "memory_used": bool(hist),
        "messages": prompt + [HumanMessage(content=plan)],
    }


def retrieve_and_tools_node(state: AgentState) -> dict[str, Any]:
    settings = get_settings()
    question = _resolved_question(state["question"], state.get("history") or [])
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
    hits = store.similarity_search_with_score(question, k=settings.top_k)
    # FAISS L2: lower score = closer. Drop weak matches (anti-hallucination gate).
    max_distance = settings.max_retrieval_distance
    for doc, score in hits:
        dist = float(score)
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

    for term in _extract_terms(state["question"])[:2]:
        definition = run_tool("define_term", {"term": term})
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
                "3) If Context is incomplete, say what is missing instead of inventing facts.\n"
                "4) Do not use outside world knowledge.\n"
                "5) Prefer concise, precise answers over broad speculation.\n\n"
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
    """Post-answer anti-hallucination check: citations + optional LLM verdict."""
    answer = state.get("answer") or ""
    sources = state.get("retrieved") or []
    if not sources:
        return {"grounded": False, "confidence": 0.0}

    cited = set(int(x) for x in re.findall(r"\[(\d+)\]", answer))
    valid_ids = set(range(1, len(sources) + 1))
    citation_ok = bool(cited) and cited.issubset(valid_ids)

    # Distance-based confidence (L2): map best hit into 0..1 (closer => higher).
    best = min((float(s["score"]) for s in sources), default=99.0)
    max_d = get_settings().max_retrieval_distance
    retrieval_conf = max(0.0, min(1.0, 1.0 - (best / max(max_d, 1e-6))))

    grounded = citation_ok
    confidence = round(0.55 * retrieval_conf + 0.45 * (1.0 if citation_ok else 0.2), 3)

    # If citations missing, rewrite with a strict grounded refusal/fix.
    if not citation_ok:
        llm = get_chat_model()
        context = "\n\n".join(
            f"[{i}] {s['filename']}: {s['content'][:500]}" for i, s in enumerate(sources, start=1)
        )
        fix = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Rewrite the answer so every factual sentence cites Context with [n]. "
                        "If you cannot support the answer from Context, refuse briefly."
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
        grounded = bool(cited) and cited.issubset(valid_ids)
        confidence = round(0.55 * retrieval_conf + 0.45 * (1.0 if grounded else 0.15), 3)

    return {
        "answer": answer,
        "grounded": grounded,
        "confidence": confidence,
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


def run_agent(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    graph = get_graph()
    # Rebuild graph if node set changed across reloads in long-running servers.
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
