"""LLM helpers: TAMU Chat API, OpenAI, or grounded mock synthesis."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Iterator, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.config import get_settings

logger = logging.getLogger(__name__)


class MockChatModel(BaseChatModel):
    """Deterministic grounded responder used when no API key is configured."""

    @property
    def _llm_type(self) -> str:
        return "mock-grounded"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any):
        text = self._respond(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any):
        return self._generate(messages, stop=stop, **kwargs)

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any):
        text = self._respond(messages)
        buf = ""
        for i, ch in enumerate(text):
            buf += ch
            if ch.isspace() or i == len(text) - 1:
                yield ChatGenerationChunk(message=AIMessageChunk(content=buf))
                buf = ""

    def _respond(self, messages: list[BaseMessage]) -> str:
        system = ""
        human = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system += str(msg.content) + "\n"
            elif isinstance(msg, HumanMessage):
                human += str(msg.content) + "\n"

        if "decompose" in system.lower() and "context:" not in (system + human).lower():
            steps = [
                "1. Clarify the information need from the user question.",
                "2. Retrieve the most relevant document passages with semantic search.",
                "3. Optionally look up definitions or related terms with tools.",
                "4. Synthesize a source-grounded answer and cite evidence.",
            ]
            return "\n".join(steps)

        context_match = re.search(r"Context:\n(.*?)(?:\n\nQuestion:|\Z)", system + "\n" + human, re.S | re.I)
        question_match = re.search(r"Question:\s*(.*)", human, re.S)
        context = (context_match.group(1).strip() if context_match else "").strip()
        question = (question_match.group(1).strip() if question_match else human.strip()).strip()

        if "rewrite the answer" in system.lower() and "draft:" in human.lower():
            draft_m = re.search(r"Draft:\n(.*?)(?:\n\nContext:|\Z)", human, re.S | re.I)
            draft = (draft_m.group(1).strip() if draft_m else "").strip()
            if draft and "[" in draft:
                return draft
            return (
                "Based on the retrieved passages, the answer is supported by the indexed evidence [1]. "
                "Additional detail is available in related chunks [2]."
            )

        if not context:
            return (
                "I could not find supporting evidence in the knowledge base for that question. "
                "Upload related documents and try again."
            )

        blocks = re.split(r"\n\n--\n\n", context)
        bullets = []
        cited_terms = []
        for idx, snip in enumerate(blocks[:4], start=1):
            clean = re.sub(r"\s+", " ", snip).strip()
            if not clean:
                continue
            bullets.append(f"- {clean[:300]} [{idx}]")
            cited_terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_\-]{3,}", clean.lower()))

        key_phrases = []
        for phrase in [
            "retrieval", "generation", "hallucination", "documents", "faiss", "similarity",
            "vector", "index", "plan", "tool", "retrieve", "synthesize", "relevance",
            "evaluation", "metrics", "improve",
        ]:
            if phrase in " ".join(cited_terms) or phrase in context.lower():
                key_phrases.append(phrase)

        return (
            f"Based on the retrieved knowledge base passages, here is a grounded answer to: {question} [1]\n\n"
            + "\n".join(bullets)
            + "\n\nKey evidence themes: "
            + ", ".join(dict.fromkeys(key_phrases) or ["retrieved context"])
            + " [1][2].\nThis synthesis stays within the retrieved evidence and cites the listed sources."
        )


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("429", "rate limit", "ratelimit", "resource_exhausted", "throttling", "quota")
    )


class ResilientChatModel(BaseChatModel):
    """Retry + optional model fallbacks for TAMU / OpenAI rate limits."""

    def __init__(
        self,
        models: Sequence[BaseChatModel],
        *,
        max_retries: int = 4,
        base_delay: float = 1.5,
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("ResilientChatModel requires at least one model")
        self._models = list(models)
        self._max_retries = max_retries
        self._base_delay = base_delay

    @property
    def _llm_type(self) -> str:
        return "resilient-chat"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> ChatResult:
        last_error: BaseException | None = None
        for model_idx, model in enumerate(self._models):
            for attempt in range(self._max_retries):
                try:
                    return model._generate(messages, stop=stop, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if not _is_rate_limit_error(exc):
                        raise
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM rate-limited on model %s (attempt %s/%s); sleeping %.1fs",
                        getattr(model, "model_name", model_idx),
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    time.sleep(delay)
            logger.warning("Exhausted retries for model index %s; trying fallback if available", model_idx)
        assert last_error is not None
        raise last_error

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any):
        last_error: BaseException | None = None
        for model_idx, model in enumerate(self._models):
            for attempt in range(self._max_retries):
                try:
                    yielded = False
                    for chunk in model.stream(messages, stop=stop, **kwargs):
                        yielded = True
                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        yield ChatGenerationChunk(message=AIMessageChunk(content=str(content or "")))
                    if yielded:
                        return
                    result = model._generate(messages, stop=stop, **kwargs)
                    text = str(result.generations[0].message.content)
                    yield ChatGenerationChunk(message=AIMessageChunk(content=text))
                    return
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if not _is_rate_limit_error(exc):
                        raise
                    delay = self._base_delay * (2 ** attempt)
                    logger.warning(
                        "LLM stream rate-limited on model %s (attempt %s/%s); sleeping %.1fs",
                        getattr(model, "model_name", model_idx),
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    time.sleep(delay)
            logger.warning("Exhausted stream retries for model index %s; trying fallback", model_idx)
        assert last_error is not None
        raise last_error

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any):
        return self._generate(messages, stop=stop, **kwargs)


def _tamus_models() -> list[BaseChatModel]:
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    model_ids = [settings.tamus_chat_model, *settings.tamus_fallback_models]
    seen: set[str] = set()
    unique: list[str] = []
    for mid in model_ids:
        if mid and mid not in seen:
            seen.add(mid)
            unique.append(mid)
    return [
        ChatOpenAI(
            model=mid,
            temperature=0.2,
            api_key=settings.tamus_ai_chat_api_key,
            base_url=settings.tamus_api_base,
            max_retries=0,
            request_timeout=60,
            streaming=True,
        )
        for mid in unique
    ]


def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    if settings.use_tamus:
        return ResilientChatModel(_tamus_models(), max_retries=4, base_delay=1.5)
    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        return ResilientChatModel(
            [
                ChatOpenAI(
                    model=settings.openai_model,
                    temperature=0.2,
                    api_key=settings.openai_api_key,
                    max_retries=0,
                    streaming=True,
                )
            ],
            max_retries=3,
            base_delay=1.0,
        )
    return MockChatModel()


def stream_chat(messages: list[BaseMessage]) -> Iterator[str]:
    """Yield text deltas from the configured chat model."""
    llm = get_chat_model()
    for chunk in llm.stream(messages):
        content = getattr(chunk, "content", None)
        if content:
            yield str(content)


def llm_mode() -> str:
    return get_settings().llm_provider


class RateLimitError(RuntimeError):
    """Raised when all LLM providers are rate-limited."""
