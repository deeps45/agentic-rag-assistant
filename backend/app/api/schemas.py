"""Pydantic schemas for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    filename: str
    title: str
    char_count: int
    chunk_count: int
    uploaded_at: str


class IngestTextRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list)


class SourceOut(BaseModel):
    id: int
    filename: str
    chunk_index: int
    snippet: str
    score: float
    doc_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    plan: str
    sources: list[SourceOut]
    tool_trace: list[dict[str, Any]]
    mode: str
    steps: list[str]


class StatusOut(BaseModel):
    app: str
    mode: str
    documents: int
    chunks: int
    openai_configured: bool
    tamus_configured: bool = False
    llm_configured: bool = False


class EvalSummary(BaseModel):
    created_at: str
    mode: str
    baseline_score: float
    improved_score: float
    improvement_pct: float
    notes: str
    cases: list[dict[str, Any]]
    baseline_cases: list[dict[str, Any]] | None = None
    report_path: str | None = None
