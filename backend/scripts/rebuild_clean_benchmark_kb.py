"""Rebuild a clean benchmark knowledge base.

Uses Hugging Face rag-mini-wikipedia (3,200 focused passages) plus curated
domain docs. Drops the noisy broad Wikipedia dump so eval numbers stay honest.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wipe_index() -> None:
    from app.config import get_settings

    settings = get_settings()
    # Fast reset — avoid per-document delete (that rebuilds FAISS each time).
    for path in (settings.docs_dir, settings.index_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    meta = settings.data_dir / "documents_meta.json"
    if meta.exists():
        meta.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-dir",
        type=Path,
        default=ROOT / "corpus" / "hf_rag_mini_wikipedia",
        help="Directory of hf_rag_mini_wikipedia_*.txt shards",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download HF shards when the directory is empty",
    )
    args = parser.parse_args()

    if args.download_if_missing and not list(args.hf_dir.glob("hf_rag_mini_wikipedia_*.txt")):
        from scripts.ingest_hf_rag_mini import download_corpus

        download_corpus(args.hf_dir)

    wipe_index()

    # Clear settings/store singletons after wipe.
    from app.config import get_settings
    from app.rag import store as store_mod
    import app.agent.graph as graph_mod

    get_settings.cache_clear()
    store_mod._store = None
    graph_mod._graph = None

    from app.rag.store import reset_store

    store = reset_store()

    shards = sorted(args.hf_dir.glob("hf_rag_mini_wikipedia_*.txt"))
    if not shards:
        raise SystemExit(f"No HF shards in {args.hf_dir}. Re-run with --download-if-missing.")

    print(f"Ingesting {len(shards)} HF rag-mini-wikipedia shards…")
    for i, path in enumerate(shards, start=1):
        store.ingest_file(path, persist=False)
        if i % 10 == 0 or i == len(shards):
            print(f"  [{i}/{len(shards)}] chunks so far={store.chunk_count()}")
    store.flush()

    sample_dir = ROOT / "sample_docs"
    print("Adding curated domain docs…")
    for path in sorted(sample_dir.glob("*.txt")):
        meta = store.ingest_file(path)
        print(f"  + {meta.filename} ({meta.chunk_count} chunks)")

    print(f"Done. documents={store.document_count()} chunks={store.chunk_count()}")


if __name__ == "__main__":
    main()
