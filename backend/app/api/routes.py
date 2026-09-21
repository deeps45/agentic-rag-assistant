"""FastAPI route handlers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agent.graph import iter_agent_events, run_agent
from app.agent.llm import llm_mode
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentOut,
    EvalSummary,
    IngestTextRequest,
    StatusOut,
)
from app.config import get_settings
from app.eval.pipeline import latest_evaluation, run_evaluation
from app.rag.store import get_store

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status", response_model=StatusOut)
def status() -> StatusOut:
    settings = get_settings()
    store = get_store()
    return StatusOut(
        app=settings.app_name,
        mode=llm_mode(),
        documents=store.document_count(),
        chunks=store.chunk_count(),
        openai_configured=bool(settings.openai_api_key),
        tamus_configured=settings.use_tamus,
        llm_configured=settings.llm_provider != "mock",
    )


@router.get("/documents", response_model=list[DocumentOut])
def list_documents() -> list[DocumentOut]:
    return [DocumentOut(**d.__dict__) for d in get_store().list_documents()]


@router.post("/documents/upload", response_model=DocumentOut)
async def upload_document(file: UploadFile = File(...)) -> DocumentOut:
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    if suffix not in {".txt", ".md", ".markdown", ".pdf", ".docx", ".csv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        meta = get_store().ingest_file(tmp_path, original_name=file.filename or tmp_path.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    return DocumentOut(**meta.__dict__)


@router.post("/documents/text", response_model=DocumentOut)
def ingest_text(body: IngestTextRequest) -> DocumentOut:
    meta = get_store().ingest_text(body.title, body.text)
    return DocumentOut(**meta.__dict__)


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, bool]:
    ok = get_store().delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True}


def _normalize_history(raw: list[dict[str, str]] | None) -> list[dict[str, str]]:
    return [
        {"role": (h.get("role") or "user"), "content": (h.get("content") or "")}
        for h in (raw or [])
        if (h.get("content") or "").strip()
    ]


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    history = _normalize_history(body.history)
    try:
        result = run_agent(body.question.strip(), history=history)
        return ChatResponse(**result)
    except Exception as exc:  # noqa: BLE001
        text = str(exc).lower()
        if any(m in text for m in ("429", "rate limit", "resource_exhausted", "throttling")):
            raise HTTPException(
                status_code=429,
                detail=(
                    "The TAMU chat model is rate-limited right now (429). "
                    "Wait a few seconds and try again — the app will also auto-retry / fall back to another model."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


@router.post("/chat/stream")
def chat_stream(body: ChatRequest) -> StreamingResponse:
    """SSE stream of agent events (plan, tools, tokens, final)."""
    history = _normalize_history(body.history)
    question = body.question.strip()

    def event_gen():
        try:
            for evt in iter_agent_events(question, history=history):
                payload = json.dumps(evt.get("data") or {}, ensure_ascii=False)
                name = evt.get("event") or "message"
                yield f"event: {name}\ndata: {payload}\n\n"
        except Exception as exc:  # noqa: BLE001
            text = str(exc).lower()
            if any(m in text for m in ("429", "rate limit", "resource_exhausted", "throttling")):
                detail = (
                    "The TAMU chat model is rate-limited right now (429). "
                    "Wait a few seconds and try again."
                )
            else:
                detail = f"Chat failed: {exc}"
            yield f"event: error\ndata: {json.dumps({'detail': detail})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/eval/run", response_model=EvalSummary)
def eval_run() -> EvalSummary:
    report = run_evaluation(persist=True)
    return EvalSummary(**report)


@router.get("/eval/latest", response_model=EvalSummary | None)
def eval_latest() -> EvalSummary | None:
    report = latest_evaluation()
    if report is None:
        return None
    return EvalSummary(**report)
