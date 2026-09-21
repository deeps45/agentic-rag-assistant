# Groundline — Agentic RAG Knowledge Assistant

Python + LangGraph/LangChain/FAISS backend and React UI for document ingestion, hybrid retrieval, agentic multi-step querying, source-grounded answers, and retrieval evaluation.

**GitHub:** [https://github.com/deeps45/agentic-rag-assistant](https://github.com/deeps45/agentic-rag-assistant)

```bash
git clone https://github.com/deeps45/agentic-rag-assistant.git
cd agentic-rag-assistant
```

## Features

- **Document ingestion** — upload `.txt`, `.md`, `.pdf`, `.docx`, or paste text; chunked and embedded into FAISS
- **Hybrid retrieval** — BM25 (lexical) + FAISS (dense) fused with reciprocal rank fusion to cut off-topic sources
- **Cross-encoder re-rank** — FlashRank (TinyBERT) re-scores hybrid candidates before synthesis
- **Agentic RAG** — LangGraph pipeline: plan → retrieve + tools → synthesize → ground-check
- **Streaming chat** — SSE (`/api/chat/stream`) streams plan, tools, tokens, and final grounded answer
- **Conversational memory** — follow-ups are resolved against prior turns so retrieval stays on-topic
- **Anti-hallucination** — citation validation, lexical support check, cite-only source filtering, refuse when unsupported
- **Tools** — hybrid semantic search, list documents, on-topic `define_term` (skips off-topic hits)
- **Source-grounded chat** — answers with cited passages and full tool/plan/memory traces
- **Evaluation pipeline** — baseline vs improved score including citation faithfulness + unanswerable refusal accuracy
- **LLM modes** — TAMU Chat API (preferred), OpenAI fallback, or offline mock

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Optional: OpenAI API key for live LLM + embeddings

### Backend

```bash
cd backend
python3 -m pip install -r requirements.txt
cp .env.example .env   # add TAMUS_AI_CHAT_API_KEY or OPENAI_API_KEY
export PYTHONPATH="$(pwd)"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8472
```

API: [http://127.0.0.1:8472/docs](http://127.0.0.1:8472/docs)

### Frontend

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5284
```

UI: [http://127.0.0.1:5284](http://127.0.0.1:5284)

Or run both via `bash scripts/dev.sh`.

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `TAMUS_AI_CHAT_API_KEY` | Recommended | Texas A&M System AI Chat API key |
| `TAMUS_AI_CHAT_API_ENDPOINT` | No | Default `https://chat-api.tamu.ai` |
| `TAMUS_CHAT_MODEL` | No | Default `protected.gemini-2.5-flash-lite` |
| `TAMUS_EMBEDDING_MODEL` | No | Default `protected.text-embedding-3-small` |
| `OPENAI_API_KEY` | No | Fallback if TAMU key is unset |
| `OPENAI_MODEL` / `EMBEDDING_MODEL` | No | OpenAI model names |

**Priority:** TAMU → OpenAI → offline mock.

Copy `backend/.env.example` to `backend/.env` and set your TAMU key:

```bash
cp backend/.env.example backend/.env
# edit TAMUS_AI_CHAT_API_KEY=...
```

## Public knowledge base (clean benchmark default)

**Default corpus for demos and reported quality numbers:**
Hugging Face [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)
- 3,200 focused passages (clean RAG benchmark, not a noisy full-Wikipedia dump)
- 918 official question–answer pairs used by `/api/eval/run`
- Plus curated domain docs (RAG / FAISS / agentic / evaluation)

```bash
cd backend
PYTHONPATH=. python3 scripts/rebuild_clean_benchmark_kb.py --download-if-missing
# restart API after rebuild
```

Optional broader Wikipedia ingest (`scripts/ingest_wikipedia_hf.py`, `corpus/wikipedia/`) is available for open-domain exploration, but is **not** used for the default quality numbers.

Restart the API after rebuilding so it reloads FAISS + BM25.

### Grounding & memory

- Conversation history is sent with each chat turn (follow-ups resolve via memory).
- Hybrid retrieval (BM25 + FAISS / RRF) ranks candidates; weak matches are dropped (`MAX_RETRIEVAL_DISTANCE`).
- Answers must cite sources; a ground-check step rewrites or refuses if citations are missing.

## API surface

| Method | Path | Description |
|---|---|---|
| GET | `/api/status` | Mode, doc/chunk counts |
| GET/POST/DELETE | `/api/documents…` | List, upload, ingest text, delete |
| POST | `/api/chat` | Agentic RAG query |
| POST/GET | `/api/eval/run`, `/api/eval/latest` | Evaluation pipeline |

## Project layout

```
backend/app/
  agent/     # LangGraph graph, tools, LLM
  rag/       # embeddings + hybrid FAISS/BM25 store
  eval/      # relevance evaluation
  api/       # FastAPI routes
frontend/    # React + Vite + Tailwind UI
sample_docs/ # Seed knowledge corpus
```

## Evaluation

`POST /api/eval/run` compares:

1. **Baseline** — top-1 chunk dump, no planning/tools  
2. **Improved** — full agentic pipeline with fuller top-k retrieval  

Reports relative lift in composite answer relevance (typically ~25%+ on the sample corpus).
