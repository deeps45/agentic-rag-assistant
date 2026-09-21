"""LangGraph agentic RAG: plan → retrieve/tools → synthesize."""

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
    messages: Annotated[list, add_messages]
    plan: str
    tool_trace: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]


def _extract_terms(question: str) -> list[str]:
    stop = {
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "for",
        "and",
        "or",
        "on",
        "with",
        "about",
        "can",
        "you",
        "me",
        "please",
        "explain",
        "describe",
        "tell",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", question)
    terms: list[str] = []
    for t in tokens:
        low = t.lower()
        if low not in stop and low not in terms:
            terms.append(t)
    return terms[:3]


def plan_node(state: AgentState) -> dict[str, Any]:
    llm = get_chat_model()
    prompt = [
        SystemMessage(
            content=(
                "You are a planning module for an agentic RAG assistant. "
                "Decompose the user question into numbered retrieval and synthesis steps. "
                "Output only the numbered plan — no preamble or analysis."
            )
        ),
        HumanMessage(content=f"Question: {state['question']}"),
    ]
    plan = llm.invoke(prompt).content
    return {"plan": str(plan), "messages": prompt + [HumanMessage(content=str(plan))]}


def retrieve_and_tools_node(state: AgentState) -> dict[str, Any]:
    settings = get_settings()
    question = state["question"]
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
    for doc, score in store.similarity_search_with_score(question, k=settings.top_k):
        retrieved.append(
            {
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "doc_id": doc.metadata.get("doc_id"),
                "score": float(score),
            }
        )

    for term in _extract_terms(question)[:2]:
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
    context = "\n\n--\n\n".join(context_blocks) if context_blocks else ""
    prompt = [
        SystemMessage(
            content=(
                "You are a knowledge assistant. Answer only from the provided Context. "
                "Cite sources like [1], [2]. If evidence is missing, say so clearly.\n\n"
                f"Plan:\n{state.get('plan', '')}\n\nContext:\n{context}"
            )
        ),
        HumanMessage(content=f"Question: {state['question']}"),
    ]
    answer = str(llm.invoke(prompt).content)
    return {"answer": answer, "sources": sources, "messages": prompt}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan_step", plan_node)
    graph.add_node("retrieve_and_tools", retrieve_and_tools_node)
    graph.add_node("synthesize", synthesize_node)
    graph.set_entry_point("plan_step")
    graph.add_edge("plan_step", "retrieve_and_tools")
    graph.add_edge("retrieve_and_tools", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(question: str) -> dict[str, Any]:
    graph = get_graph()
    result = graph.invoke(
        {
            "question": question,
            "messages": [],
            "plan": "",
            "tool_trace": [],
            "retrieved": [],
            "answer": "",
            "sources": [],
        }
    )
    return {
        "answer": result["answer"],
        "plan": result["plan"],
        "sources": result["sources"],
        "tool_trace": result["tool_trace"],
        "mode": llm_mode(),
        "steps": ["plan_step", "retrieve_and_tools", "synthesize"],
    }
