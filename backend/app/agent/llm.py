"""LLM helpers: TAMU Chat API, OpenAI, or grounded mock synthesis."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import get_settings


class MockChatModel(BaseChatModel):
    """Deterministic grounded responder used when no API key is configured."""

    @property
    def _llm_type(self) -> str:
        return "mock-grounded"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any):
        from langchain_core.outputs import ChatGeneration, ChatResult

        text = self._respond(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any):
        return self._generate(messages, stop=stop, **kwargs)

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

        if not context:
            return (
                "I could not find supporting evidence in the knowledge base for that question. "
                "Upload related documents and try again."
            )

        blocks = re.split(r"\n\n--\n\n", context)
        bullets = []
        cited_terms = []
        for snip in blocks[:4]:
            clean = re.sub(r"\s+", " ", snip).strip()
            if not clean:
                continue
            bullets.append(f"- {clean[:300]}")
            cited_terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_\-]{3,}", clean.lower()))

        key_phrases = []
        for phrase in [
            "retrieval",
            "generation",
            "hallucination",
            "documents",
            "faiss",
            "similarity",
            "vector",
            "index",
            "plan",
            "tool",
            "retrieve",
            "synthesize",
            "relevance",
            "evaluation",
            "metrics",
            "improve",
        ]:
            if phrase in " ".join(cited_terms) or phrase in context.lower():
                key_phrases.append(phrase)

        return (
            f"Based on the retrieved knowledge base passages, here is a grounded answer to: {question}\n\n"
            + "\n".join(bullets)
            + "\n\nKey evidence themes: "
            + ", ".join(dict.fromkeys(key_phrases) or ["retrieved context"])
            + ".\nThis synthesis stays within the retrieved evidence and cites the listed sources."
        )


def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    if settings.use_tamus:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.tamus_chat_model,
            temperature=0.2,
            api_key=settings.tamus_ai_chat_api_key,
            base_url=settings.tamus_api_base,
        )
    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=0.2,
            api_key=settings.openai_api_key,
        )
    return MockChatModel()


def llm_mode() -> str:
    return get_settings().llm_provider
