"""Ingest the downloaded Wikipedia corpus into the FAISS knowledge store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.rag.store import KnowledgeStore, reset_store  # noqa: E402


def clear_store(store: KnowledgeStore) -> None:
    for doc in list(store.list_documents()):
        store.delete_document(doc.id)


def ingest_corpus(corpus_dir: Path, replace: bool = True) -> None:
    settings = get_settings()
    store = reset_store()
    if replace:
        print("Clearing existing documents…")
        clear_store(store)
        store = reset_store()

    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files in {corpus_dir}. Run download_wikipedia_corpus.py first.")

    print(f"Ingesting {len(files)} files from {corpus_dir} (provider={settings.llm_provider})…")
    for i, path in enumerate(files, start=1):
        meta = store.ingest_file(path)
        print(f"[{i}/{len(files)}] {meta.title} → {meta.chunk_count} chunks")

    print(f"Done. documents={store.document_count()} chunks={store.chunk_count()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia corpus into Groundline")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "corpus" / "wikipedia",
        help="Directory of .txt articles",
    )
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear current KB first")
    args = parser.parse_args()
    ingest_corpus(args.corpus, replace=not args.keep_existing)


if __name__ == "__main__":
    main()
