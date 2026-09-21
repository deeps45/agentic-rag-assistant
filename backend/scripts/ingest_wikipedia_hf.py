"""Stream and ingest articles from Hugging Face wikimedia/wikipedia (20231101.en).

Keeps the existing knowledge base; skips titles already present.
Uses batched TAMU/OpenAI embeddings via KnowledgeStore.ingest_* (persist deferred).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _safe_name(title: str) -> str:
    return re.sub(r"[^\w\-]+", "_", title).strip("_")[:120]


def _article_text(title: str, text: str) -> str:
    wiki_slug = title.replace(" ", "_")
    body = text.strip()
    if len(body) > 14000:
        body = body[:14000] + "\n\n[Article truncated for corpus size.]"
    return (
        f"Title: {title}\n"
        f"Source: https://en.wikipedia.org/wiki/{wiki_slug}\n"
        f"License: Creative Commons Attribution-ShareAlike "
        f"(Hugging Face wikimedia/wikipedia 20231101.en)\n\n"
        f"{body}"
    )


def download_and_ingest(
    *,
    limit: int,
    min_chars: int,
    out_dir: Path,
    save_files: bool,
    persist_every: int,
    skip_existing: bool,
) -> None:
    from datasets import load_dataset

    from app.config import get_settings
    from app.rag.store import reset_store

    settings = get_settings()
    store = reset_store()
    known_titles = store.known_titles() if skip_existing else set()
    known_files = store.known_filenames() if skip_existing else set()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Streaming wikimedia/wikipedia 20231101.en "
        f"(limit={limit}, min_chars={min_chars}, provider={settings.llm_provider})…"
    )
    print(f"Existing KB: documents={store.document_count()} chunks={store.chunk_count()}")

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    added = 0
    skipped = 0
    scanned = 0
    started = time.time()
    manifest_articles: list[dict] = []

    for row in ds:
        scanned += 1
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        if not title or len(text) < min_chars:
            continue
        # Skip disambiguation / list dumps that add noise.
        low = title.lower()
        if low.startswith("list of ") or "(disambiguation)" in low:
            continue
        filename = f"wikipedia_hf_{_safe_name(title)}.txt"
        if skip_existing and (title.lower() in known_titles or filename.lower() in known_files):
            skipped += 1
            continue

        payload = _article_text(title, text)
        if save_files:
            path = out_dir / filename
            path.write_text(payload, encoding="utf-8")
            meta = store.ingest_file(path, original_name=filename, persist=False)
        else:
            # Stable filename for dedupe without keeping corpus shards on disk.
            tmp = out_dir / f"_tmp_{filename}"
            tmp.write_text(payload, encoding="utf-8")
            try:
                meta = store.ingest_file(tmp, original_name=filename, persist=False)
            finally:
                tmp.unlink(missing_ok=True)

        known_titles.add(title.lower())
        known_files.add(filename.lower())
        added += 1
        manifest_articles.append(
            {
                "title": title,
                "filename": meta.filename,
                "chars": meta.char_count,
                "chunks": meta.chunk_count,
                "doc_id": meta.id,
            }
        )
        if added % 25 == 0 or added == 1:
            elapsed = time.time() - started
            print(
                f"  [{added}/{limit}] {title} → {meta.chunk_count} chunks "
                f"(scanned={scanned}, skipped={skipped}, {elapsed:.0f}s)"
            )
        if persist_every > 0 and added % persist_every == 0:
            store.flush()
            print(f"  …checkpoint persist at {added} articles")

        if added >= limit:
            break

    store.flush()
    manifest = {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "requested": limit,
        "added": added,
        "skipped_existing": skipped,
        "scanned": scanned,
        "min_chars": min_chars,
        "articles": manifest_articles,
        "documents_total": store.document_count(),
        "chunks_total": store.chunk_count(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Done. added={added} skipped={skipped} scanned={scanned} "
        f"documents={store.document_count()} chunks={store.chunk_count()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest streaming Wikipedia (HF wikimedia/wikipedia) into Groundline"
    )
    parser.add_argument("--limit", type=int, default=1000, help="Articles to add (500–2000 typical)")
    parser.add_argument("--min-chars", type=int, default=800, help="Minimum article body length")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "corpus" / "wikipedia_hf",
        help="Directory for optional shard files + manifest",
    )
    parser.add_argument(
        "--save-files",
        action="store_true",
        help="Write .txt copies under --out (larger disk use)",
    )
    parser.add_argument(
        "--persist-every",
        type=int,
        default=50,
        help="Flush FAISS/meta every N articles (0 = only at end)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip titles already in the KB",
    )
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    download_and_ingest(
        limit=args.limit,
        min_chars=args.min_chars,
        out_dir=args.out,
        save_files=args.save_files,
        persist_every=args.persist_every,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
