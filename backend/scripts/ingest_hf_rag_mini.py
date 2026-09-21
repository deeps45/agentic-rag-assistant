"""Download Hugging Face rag-mini-wikipedia (3,200 public passages) and ingest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def download_corpus(out_dir: Path, shard_size: int = 40) -> Path:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
    print(f"Loaded {len(ds)} passages from Hugging Face rag-mini-wikipedia")

    shards: list[Path] = []
    buffer: list[str] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal buffer, shard_idx
        if not buffer:
            return
        shard_idx += 1
        path = out_dir / f"hf_rag_mini_wikipedia_{shard_idx:04d}.txt"
        header = (
            "Source: Hugging Face dataset rag-datasets/rag-mini-wikipedia (text-corpus)\n"
            "License: see dataset card on huggingface.co\n\n"
        )
        path.write_text(header + "\n\n---\n\n".join(buffer), encoding="utf-8")
        shards.append(path)
        buffer = []

    for row in ds:
        passage = (row.get("passage") or "").strip()
        if len(passage) < 40:
            continue
        pid = row.get("id", "")
        buffer.append(f"[passage_id={pid}]\n{passage}")
        if len(buffer) >= shard_size:
            flush()
    flush()

    manifest = {
        "dataset": "rag-datasets/rag-mini-wikipedia",
        "config": "text-corpus",
        "passages": len(ds),
        "shards": [p.name for p in shards],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(shards)} shard files → {out_dir}")
    return out_dir


def ingest(out_dir: Path, replace: bool) -> None:
    from app.rag.store import reset_store

    store = reset_store()
    if replace:
        print("Clearing existing documents…")
        for doc in list(store.list_documents()):
            store.delete_document(doc.id)
        store = reset_store()

    files = sorted(out_dir.glob("hf_rag_mini_wikipedia_*.txt"))
    if not files:
        raise SystemExit(f"No shard files in {out_dir}")
    print(f"Ingesting {len(files)} shards (keep_existing={not replace})…")
    for i, path in enumerate(files, start=1):
        meta = store.ingest_file(path)
        print(f"[{i}/{len(files)}] {meta.filename} → {meta.chunk_count} chunks")
    print(f"Done. documents={store.document_count()} chunks={store.chunk_count()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "corpus" / "hf_rag_mini_wikipedia")
    parser.add_argument("--shard-size", type=int, default=40)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--replace", action="store_true", help="Clear KB before ingest")
    args = parser.parse_args()

    if not args.ingest_only:
        download_corpus(args.out, shard_size=args.shard_size)
    if not args.download_only:
        ingest(args.out, replace=args.replace)


if __name__ == "__main__":
    main()
