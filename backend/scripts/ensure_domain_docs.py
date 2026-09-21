"""Ensure curated domain docs (RAG/FAISS/agentic/eval) are present in the live KB."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.store import get_store, reset_store  # noqa: E402


def main() -> None:
    sample_dir = ROOT / "sample_docs"
    store = reset_store()
    existing = {d.filename.lower() for d in store.list_documents()}
    added = 0
    for path in sorted(sample_dir.glob("*.txt")):
        if path.name.lower() in existing:
            print(f"skip existing {path.name}")
            continue
        meta = store.ingest_file(path)
        print(f"added {meta.filename} → {meta.chunk_count} chunks")
        added += 1
    # Also prefer curated Wikipedia pages if present on disk
    wiki = ROOT / "corpus" / "wikipedia"
    prefer = [
        "wikipedia_Semantic_search.txt",
        "wikipedia_Information_retrieval.txt",
        "wikipedia_Machine_learning.txt",
        "wikipedia_Large_language_model.txt",
        "wikipedia_Foundation_model.txt",
    ]
    for name in prefer:
        path = wiki / name
        if path.exists() and path.name.lower() not in existing:
            # may already be ingested under same filename
            if any(d.filename == path.name for d in store.list_documents()):
                continue
    print(f"done. added={added} documents={store.document_count()} chunks={store.chunk_count()}")


if __name__ == "__main__":
    main()
