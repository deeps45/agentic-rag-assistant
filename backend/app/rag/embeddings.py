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

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # API accepts a single string or list of strings; batch one-by-one for reliability.
        vectors: List[List[float]] = []
        with httpx.Client(timeout=60.0) as client:
            for text in texts:
                resp = client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.model, "input": text},
                )
                resp.raise_for_status()
                payload = resp.json()
                data = payload.get("data") or []
                if not data:
                    raise RuntimeError(f"Empty embedding response: {payload}")
                vectors.append(list(data[0]["embedding"]))
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
