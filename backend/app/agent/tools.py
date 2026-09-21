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
    # Prefer showing domain docs first for readability in traces.
    domain_names = {
        "rag_fundamentals.txt",
        "faiss_vector_search.txt",
        "agentic_rag_langgraph.txt",
        "evaluation_quality.txt",
    }
    ordered = sorted(docs, key=lambda d: (0 if d.filename in domain_names else 1, d.title.lower()))
    return "\n".join(f"- {d.title} ({d.filename}) · {d.chunk_count} chunks" for d in ordered[:40])


@tool
def define_term(term: str) -> str:
    """Look up a definition-like explanation for a term from the knowledge base.

    Returns only passages that actually mention the term. Skips off-topic hits.
    """
    term = (term or "").strip()
    if len(term) < 3:
        return "Term too short to define."

    store = get_store()
    # Query both the bare term and an explicit definition phrasing, then dedupe.
    seen: set[str] = set()
    hits = []
    for query in (term, f"definition of {term}", f"what is {term}"):
        for h in store.similarity_search(query, k=4):
            key = f"{h.metadata.get('filename')}:{h.metadata.get('chunk_index')}:{h.page_content[:80]}"
            if key in seen:
                continue
            seen.add(key)
            hits.append(h)
    if not hits:
        return f"No definition found for '{term}'."

    term_l = term.lower()
    # Keep only chunks that literally mention the term (word-ish match).
    term_pat = re.compile(rf"\b{re.escape(term_l)}\b", re.I)
    relevant = [h for h in hits if term_pat.search(h.page_content)]
    if not relevant:
        # Soft fallback: substring match for hyphenated / plural forms.
        relevant = [h for h in hits if term_l in h.page_content.lower()]
    if not relevant:
        return f"No on-topic definition found for '{term}' in retrieved passages."

    joined = " ".join(h.page_content for h in relevant[:3])
    sentences = re.split(r"(?<=[.!?])\s+", joined)
    focused = [s.strip() for s in sentences if term_l in s.lower() and len(s.strip()) > 20]
    if not focused:
        # Fall back to a short excerpt around the first mention.
        idx = relevant[0].page_content.lower().find(term_l)
        start = max(0, idx - 40)
        excerpt = relevant[0].page_content[start : start + 280].strip()
        return excerpt or f"No clear definition for '{term}'."
    return " ".join(focused[:3])


AGENT_TOOLS = [semantic_search, list_knowledge_documents, define_term]


def tool_by_name(name: str):
    for t in AGENT_TOOLS:
        if t.name == name:
            return t
    raise KeyError(name)


def run_tool(name: str, args: dict[str, Any]) -> str:
    tool_fn = tool_by_name(name)
    return tool_fn.invoke(args)
