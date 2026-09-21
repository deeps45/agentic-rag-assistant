"""Cross-encoder style re-ranking over hybrid retrieval candidates.

Prefers FlashRank (tiny ONNX cross-encoder). Falls back to a lightweight
query–passage interaction scorer when FlashRank is unavailable.
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from app.config import get_settings

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def _flashrank_ranker():
    try:
        from flashrank import Ranker
    except ImportError:
        return None
    settings = get_settings()
    cache_dir = str(settings.data_dir / "flashrank_cache")
    try:
        return Ranker(model_name=settings.rerank_model, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FlashRank init failed (%s); using fallback reranker", exc)
        return None


def reranker_backend() -> str:
    return "flashrank" if _flashrank_ranker() is not None else "interaction"


def _interaction_score(query: str, passage: str) -> float:
    """Cheap cross-style score: coverage × density × length-normalized overlap."""
    q = _tokenize(query)
    p = _tokenize(passage)
    if not q or not p:
        return 0.0
    qset, pset = set(q), set(p)
    # Prefer contentful tokens (len >= 3) for coverage.
    q_terms = {t for t in qset if len(t) >= 3} or qset
    covered = len(q_terms & pset) / max(len(q_terms), 1)
    # Soft-tf: how often query terms appear in the passage.
    counts = {t: 0 for t in q_terms}
    for tok in p:
        if tok in counts:
            counts[tok] += 1
    density = sum(min(c, 3) for c in counts.values()) / max(len(q_terms) * 3, 1)
    # Mild length prior so tiny fragments don't dominate.
    length_prior = min(1.0, math.log1p(len(p)) / math.log1p(120))
    return 0.55 * covered + 0.35 * density + 0.10 * length_prior


def _flashrank_scores(query: str, passages: list[str]) -> list[float] | None:
    ranker = _flashrank_ranker()
    if ranker is None or not passages:
        return None
    try:
        from flashrank import RerankRequest

        req = RerankRequest(
            query=query,
            passages=[{"id": i, "text": text[:2000]} for i, text in enumerate(passages)],
        )
        ranked = ranker.rerank(req)
        scores = [0.0] * len(passages)
        for item in ranked:
            idx = int(item["id"])
            if 0 <= idx < len(scores):
                scores[idx] = float(item.get("score") or 0.0)
        return scores
    except Exception as exc:  # noqa: BLE001
        logger.warning("FlashRank rerank failed (%s); using fallback", exc)
        return None


def rerank_documents(
    query: str,
    docs_with_scores: list[tuple[Document, float]],
    *,
    top_n: int | None = None,
) -> list[tuple[Document, float]]:
    """Re-rank hybrid candidates. Returns (doc, distance) with lower = better."""
    settings = get_settings()
    if not settings.rerank_enabled or not docs_with_scores:
        return docs_with_scores[: top_n or len(docs_with_scores)]

    top_n = top_n or min(settings.top_k, len(docs_with_scores))
    passages = [d.page_content for d, _ in docs_with_scores]
    ce_scores = _flashrank_scores(query, passages)
    if ce_scores is None:
        ce_scores = [_interaction_score(query, p) for p in passages]

    max_ce = max(ce_scores) if ce_scores else 0.0
    blended: list[tuple[Document, float, float]] = []
    for (doc, hybrid_dist), ce in zip(docs_with_scores, ce_scores):
        # Normalize CE to [0,1]; boost good matches without wiping the shortlist.
        ce_norm = (ce / max_ce) if max_ce > 1e-9 else 0.0
        # Soft blend: keep hybrid scale so domain filters still see a usable spread.
        distance = float(hybrid_dist) * (1.0 - 0.55 * ce_norm)
        blended.append((doc, distance, ce_norm))

    blended.sort(key=lambda x: x[1])
    return [(doc, dist) for doc, dist, _ in blended[:top_n]]


def annotate_rerank_meta(query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach rerank backend label onto retrieved item dicts (for tool traces)."""
    backend = reranker_backend() if get_settings().rerank_enabled else "off"
    for item in items:
        item["rerank"] = backend
    return items
