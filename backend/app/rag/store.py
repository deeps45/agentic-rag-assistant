"""Document loading, chunking, FAISS + BM25 hybrid index management."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from app.config import Settings, get_settings
from app.rag.embeddings import get_embeddings

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class DocumentMeta:
    id: str
    filename: str
    title: str
    char_count: int
    chunk_count: int
    uploaded_at: str
    source_path: str


class KnowledgeStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.meta_path = self.settings.data_dir / "documents_meta.json"
        self._vectorstore: FAISS | None = None
        self._documents: dict[str, DocumentMeta] = {}
        # Parallel sparse index over the same chunks as FAISS.
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[Document] = []
        self._bm25_tokens: list[list[str]] = []
        self._load_meta()
        self._load_or_create_index()
        self._rebuild_bm25()

    def _load_meta(self) -> None:
        if self.meta_path.exists():
            raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self._documents = {item["id"]: DocumentMeta(**item) for item in raw}
        else:
            self._documents = {}

    def _save_meta(self) -> None:
        payload = [asdict(doc) for doc in self._documents.values()]
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _embedding_dim(self) -> int:
        probe = get_embeddings().embed_query("dimension probe")
        return len(probe)

    def _create_empty_index(self) -> FAISS:
        embeddings = get_embeddings()
        dim = self._embedding_dim()
        if faiss is None:
            raise RuntimeError("faiss is required")
        index = faiss.IndexFlatL2(dim)
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )

    def _load_or_create_index(self) -> None:
        index_file = self.settings.index_dir / "index.faiss"
        if index_file.exists():
            try:
                store = FAISS.load_local(
                    str(self.settings.index_dir),
                    get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
                # Rebuild if embedding provider/dimension changed (e.g. mock → TAMU).
                expected = self._embedding_dim()
                if store.index.d != expected:
                    raise ValueError(f"index dim {store.index.d} != embedding dim {expected}")
                self._vectorstore = store
                return
            except Exception:
                shutil.rmtree(self.settings.index_dir, ignore_errors=True)
                self.settings.index_dir.mkdir(parents=True, exist_ok=True)
                # Drop stale meta so samples re-seed with the new embeddings.
                if self.meta_path.exists():
                    self.meta_path.unlink()
                self._documents = {}
                # Also clear copied docs so seed starts clean.
                for path in self.settings.docs_dir.glob("*"):
                    if path.is_file():
                        path.unlink()
        self._vectorstore = self._create_empty_index()
        self._persist()

    def _persist(self) -> None:
        assert self._vectorstore is not None
        self._vectorstore.save_local(str(self.settings.index_dir))
        self._save_meta()

    def _iter_indexed_documents(self) -> list[Document]:
        assert self._vectorstore is not None
        docs: list[Document] = []
        for idx in range(self._vectorstore.index.ntotal):
            doc_id = self._vectorstore.index_to_docstore_id.get(idx)
            if doc_id is None:
                continue
            doc = self._vectorstore.docstore.search(doc_id)
            if isinstance(doc, Document):
                docs.append(doc)
        return docs

    def _rebuild_bm25(self) -> None:
        docs = self._iter_indexed_documents()
        self._bm25_docs = docs
        self._bm25_tokens = [_tokenize(d.page_content) for d in docs]
        if self._bm25_tokens:
            self._bm25 = BM25Okapi(self._bm25_tokens)
        else:
            self._bm25 = None

    def _append_bm25(self, documents: list[Document]) -> None:
        """Incrementally extend the sparse index (faster than full rebuild on ingest)."""
        if not documents:
            return
        for doc in documents:
            tokens = _tokenize(doc.page_content)
            self._bm25_docs.append(doc)
            self._bm25_tokens.append(tokens)
        self._bm25 = BM25Okapi(self._bm25_tokens) if self._bm25_tokens else None

    @property
    def vectorstore(self) -> FAISS:
        assert self._vectorstore is not None
        return self._vectorstore

    def list_documents(self) -> list[DocumentMeta]:
        return sorted(self._documents.values(), key=lambda d: d.uploaded_at, reverse=True)

    def document_count(self) -> int:
        return len(self._documents)

    def chunk_count(self) -> int:
        return sum(d.chunk_count for d in self._documents.values())

    def known_titles(self) -> set[str]:
        return {d.title.lower() for d in self._documents.values()}

    def known_filenames(self) -> set[str]:
        return {d.filename.lower() for d in self._documents.values()}

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown", ".csv"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix in {".docx"}:
            from docx import Document as DocxDocument

            doc = DocxDocument(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _chunk_documents(self, doc_id: str, filename: str, text: str) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        return [
            Document(
                page_content=chunk,
                metadata={
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": idx,
                    "source": filename,
                },
            )
            for idx, chunk in enumerate(chunks)
        ]

    def ingest_file(
        self,
        source: Path,
        original_name: str | None = None,
        *,
        persist: bool = True,
    ) -> DocumentMeta:
        filename = original_name or source.name
        text = self._read_file(source).strip()
        if not text:
            raise ValueError("Document is empty after extraction")

        doc_id = str(uuid.uuid4())
        dest = self.settings.docs_dir / f"{doc_id}_{filename}"
        shutil.copy2(source, dest)

        documents = self._chunk_documents(doc_id, filename, text)
        self.vectorstore.add_documents(documents)
        self._append_bm25(documents)

        meta = DocumentMeta(
            id=doc_id,
            filename=filename,
            title=Path(filename).stem.replace("_", " ").title(),
            char_count=len(text),
            chunk_count=len(documents),
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            source_path=str(dest),
        )
        self._documents[doc_id] = meta
        if persist:
            self._persist()
        return meta

    def ingest_text(self, title: str, text: str, *, persist: bool = True) -> DocumentMeta:
        safe_name = f"{title.replace(' ', '_').lower()}.txt"
        temp = self.settings.docs_dir / f"_tmp_{uuid.uuid4().hex}.txt"
        temp.write_text(text, encoding="utf-8")
        try:
            return self.ingest_file(temp, original_name=safe_name, persist=persist)
        finally:
            if temp.exists():
                temp.unlink()

    def flush(self) -> None:
        """Persist FAISS + meta after deferred ingest batches."""
        self._persist()

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self._documents:
            return False
        meta = self._documents.pop(doc_id)
        path = Path(meta.source_path)
        if path.exists():
            path.unlink()

        remaining = list(self._documents.values())
        self._documents = {}
        self._vectorstore = self._create_empty_index()
        for item in remaining:
            src = Path(item.source_path)
            if not src.exists():
                continue
            text = self._read_file(src).strip()
            if not text:
                continue
            documents = self._chunk_documents(item.id, item.filename, text)
            self.vectorstore.add_documents(documents)
            item.chunk_count = len(documents)
            item.char_count = len(text)
            self._documents[item.id] = item
        self._rebuild_bm25()
        self._persist()
        return True

    def _doc_key(self, doc: Document) -> str:
        return (
            f"{doc.metadata.get('doc_id', '')}|"
            f"{doc.metadata.get('filename', '')}|"
            f"{doc.metadata.get('chunk_index', 0)}|"
            f"{hash(doc.page_content[:240])}"
        )

    def _bm25_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if self._bm25 is None or not self._bm25_docs:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        out: list[tuple[Document, float]] = []
        for idx, score in ranked[:k]:
            if score <= 0:
                break
            out.append((self._bm25_docs[idx], float(score)))
        return out

    def _rrf_fuse(
        self,
        dense: list[tuple[Document, float]],
        sparse: list[tuple[Document, float]],
        k: int,
        rrf_k: int = 60,
    ) -> list[tuple[Document, float, float | None, float | None]]:
        """Reciprocal rank fusion → (doc, fused_rank_score, faiss_l2|None, bm25|None)."""
        fused: dict[str, dict[str, Any]] = {}

        for rank, (doc, dist) in enumerate(dense):
            key = self._doc_key(doc)
            entry = fused.setdefault(
                key,
                {"doc": doc, "rrf": 0.0, "faiss": None, "bm25": None},
            )
            entry["rrf"] += 1.0 / (rrf_k + rank + 1)
            entry["faiss"] = float(dist)
            entry["doc"] = doc

        for rank, (doc, bm25_score) in enumerate(sparse):
            key = self._doc_key(doc)
            entry = fused.setdefault(
                key,
                {"doc": doc, "rrf": 0.0, "faiss": None, "bm25": None},
            )
            entry["rrf"] += 1.0 / (rrf_k + rank + 1)
            entry["bm25"] = float(bm25_score)
            entry["doc"] = doc

        ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
        results: list[tuple[Document, float, float | None, float | None]] = []
        for entry in ranked[:k]:
            results.append(
                (entry["doc"], float(entry["rrf"]), entry["faiss"], entry["bm25"])
            )
        return results

    def _hybrid_distance(
        self,
        faiss_l2: float | None,
        bm25_score: float | None,
        max_bm25: float,
    ) -> float:
        """Map hybrid evidence to an L2-like distance (lower = better) for gating."""
        max_d = self.settings.max_retrieval_distance
        if faiss_l2 is not None and bm25_score is not None and max_bm25 > 0:
            lexical = min(1.0, bm25_score / max_bm25)
            return float(faiss_l2) * (1.0 - 0.15 * lexical)
        if faiss_l2 is not None:
            return float(faiss_l2)
        # BM25-only hits must NOT outrank dense matches with a fake ~0.2 distance.
        # Keep them near the cutoff so they only survive when dense retrieval is empty/weak.
        if bm25_score is not None and max_bm25 > 0:
            lexical = min(1.0, bm25_score / max_bm25)
            return max_d * (1.05 - 0.2 * lexical)
        return max_d + 1.0

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k)]

    def similarity_search_with_score(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Hybrid BM25 + FAISS search, then cross-encoder re-rank.

        Score is L2-like distance (lower is better).
        """
        from app.rag.rerank import rerank_documents

        k = k or self.settings.top_k
        if self.chunk_count() == 0:
            return []

        candidate_k = min(
            max(k * 4, self.settings.rerank_candidates, k),
            max(self.chunk_count(), 1),
        )
        dense = self.vectorstore.similarity_search_with_score(query, k=candidate_k)
        sparse = self._bm25_search(query, k=candidate_k)
        max_bm25 = max((s for _, s in sparse), default=0.0)

        # Rare query tokens (len>=5) → allow strong BM25-only rescue (e.g. "faiss").
        rare_tokens = {t for t in _tokenize(query) if len(t) >= 5}
        sparse_rescue: list[tuple[Document, float]] = []
        for doc, score in sparse:
            if max_bm25 <= 0:
                break
            text = doc.page_content.lower()
            if any(tok in text for tok in rare_tokens) and score >= 0.35 * max_bm25:
                sparse_rescue.append((doc, score))

        fuse_k = min(max(self.settings.rerank_candidates, k * 2), max(self.chunk_count(), 1))
        fused = self._rrf_fuse(
            dense,
            sparse_rescue if sparse_rescue else sparse[: max(k, 3)],
            k=fuse_k,
        )
        # Prefer entries that have a FAISS distance; demote pure lexical noise.
        scored: list[tuple[Document, float]] = []
        for doc, _rrf, faiss_l2, bm25_score in fused:
            dist = self._hybrid_distance(faiss_l2, bm25_score, max_bm25)
            # Extra penalty if no dense evidence and no rare-token overlap.
            if faiss_l2 is None:
                text = doc.page_content.lower()
                if not any(tok in text for tok in rare_tokens):
                    dist = max(dist, self.settings.max_retrieval_distance + 0.25)
            scored.append((doc, dist))
        scored.sort(key=lambda x: x[1])
        # Cross-encoder (or interaction fallback) over the hybrid shortlist.
        return rerank_documents(query, scored[:fuse_k], top_n=k)

    def seed_samples_if_empty(self) -> list[DocumentMeta]:
        if self.document_count() > 0:
            return []
        # Prefer clean HF rag-mini-wikipedia, then curated samples, then wiki dumps.
        candidates = [
            getattr(self.settings, "hf_corpus_dir", self.settings.data_dir / "unused"),
            self.settings.sample_docs_dir,
            self.settings.corpus_dir,
        ]
        for corpus in candidates:
            if not corpus.exists():
                continue
            files = sorted(
                p
                for p in corpus.glob("*")
                if p.is_file()
                and p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}
                and p.name != "manifest.json"
                and not p.name.upper().startswith("README")
            )
            if not files:
                continue
            added: list[DocumentMeta] = []
            for path in files:
                added.append(self.ingest_file(path, persist=False))
            self.flush()
            return added
        return []


_store: KnowledgeStore | None = None


def get_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
        _store.seed_samples_if_empty()
    return _store


def reset_store() -> KnowledgeStore:
    global _store
    _store = KnowledgeStore()
    return _store
