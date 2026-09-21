"""Embedding providers: TAMU Chat API, OpenAI, or local hash vectors."""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

import httpx
from langchain_core.embeddings import Embeddings

from app.config import get_settings


class LocalHashEmbeddings(Embeddings):
    """Offline embedding fallback so the app runs without an API key."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vec = [0.0] * self.dim
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0)
            vec[idx] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class TamusEmbeddings(Embeddings):
    """TAMU Chat API embeddings — sends string inputs (API rejects token ids)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        batch_size: int = 32,
        max_retries: int = 6,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

    def _embed(self, texts: List[str]) -> List[List[float]]:
        import time

        vectors: List[List[float]] = []
        with httpx.Client(timeout=120.0) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                payload_input: str | list[str] = batch[0] if len(batch) == 1 else batch
                last_err: Exception | None = None
                for attempt in range(self.max_retries):
                    try:
                        resp = client.post(
                            f"{self.base_url}/embeddings",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            json={"model": self.model, "input": payload_input},
                        )
                        if resp.status_code in {429, 500, 502, 503, 504}:
                            wait = min(60.0, (2**attempt) + 0.5)
                            time.sleep(wait)
                            last_err = httpx.HTTPStatusError(
                                f"{resp.status_code} for embeddings",
                                request=resp.request,
                                response=resp,
                            )
                            continue
                        resp.raise_for_status()
                        payload = resp.json()
                        data = payload.get("data") or []
                        if len(data) != len(batch):
                            raise RuntimeError(
                                f"Expected {len(batch)} embeddings, got {len(data)}: "
                                f"{payload.get('error') or payload}"
                            )
                        # OpenAI-style responses may be unordered; sort by index when present.
                        ordered = sorted(data, key=lambda row: row.get("index", 0))
                        vectors.extend(list(row["embedding"]) for row in ordered)
                        last_err = None
                        break
                    except (httpx.TransportError, httpx.TimeoutException) as exc:
                        last_err = exc
                        time.sleep(min(60.0, (2**attempt) + 0.5))
                if last_err is not None:
                    raise last_err
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


def get_embeddings() -> Embeddings:
    settings = get_settings()
    if settings.use_tamus:
        return TamusEmbeddings(
            api_key=settings.tamus_ai_chat_api_key or "",
            base_url=settings.tamus_api_base,
            model=settings.tamus_embedding_model,
        )
    if settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    return LocalHashEmbeddings()
