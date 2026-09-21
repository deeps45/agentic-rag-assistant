"""Agent tools for the LangGraph knowledge assistant."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool

from app.rag.store import get_store


@tool
def semantic_search(query: str, top_k: int = 4) -> str:
    """Hybrid search (BM25 + FAISS) over the knowledge base for relevant passages."""
    store = get_store()
    hits = store.similarity_search_with_score(query, k=top_k)
    if not hits:
        return "No documents indexed yet."
    blocks: list[str] = []
    for doc, score in hits:
        blocks.append(
            f"Source: {doc.metadata.get('filename', 'unknown')} "
            f"(chunk {doc.metadata.get('chunk_index', 0)}, hybrid_distance={score:.4f})\n"
            f"{doc.page_content}"
        )
    return "\n\n--\n\n".join(blocks)


@tool
def list_knowledge_documents() -> str:
    """List ingested documents available for retrieval."""
    docs = get_store().list_documents()
    if not docs:
        return "Knowledge base is empty."
    return "\n".join(f"- {d.title} ({d.filename}) · {d.chunk_count} chunks" for d in docs)


@tool
def define_term(term: str) -> str:
    """Look up a definition-like explanation for a term from the knowledge base."""
    store = get_store()
    hits = store.similarity_search(f"definition of {term}", k=3)
    if not hits:
        return f"No definition found for '{term}'."
    joined = " ".join(h.page_content for h in hits)
    sentences = re.split(r"(?<=[.!?])\s+", joined)
    focused = [s for s in sentences if term.lower() in s.lower()]
    text = " ".join(focused[:3]) if focused else " ".join(sentences[:3])
    return text.strip() or f"No clear definition for '{term}'."


AGENT_TOOLS = [semantic_search, list_knowledge_documents, define_term]


def tool_by_name(name: str):
    for t in AGENT_TOOLS:
        if t.name == name:
            return t
    raise KeyError(name)


def run_tool(name: str, args: dict[str, Any]) -> str:
    tool_fn = tool_by_name(name)
    return tool_fn.invoke(args)
