# Groundline — Agentic RAG Knowledge Assistant

Python + LangGraph/LangChain/FAISS backend and React UI for document ingestion, semantic retrieval, agentic multi-step querying, source-grounded answers, and retrieval evaluation.

## Features

- **Document ingestion** — upload `.txt`, `.md`, `.pdf`, `.docx`, or paste text; chunked and embedded into FAISS
- **Agentic RAG** — LangGraph pipeline: plan → retrieve + tools → synthesize
- **Tools** — semantic search, list documents, define term
- **Source-grounded chat** — answers with cited passages and full tool/plan traces
- **Evaluation pipeline** — baseline vs improved composite relevance score (hit-rate, overlap, semantic similarity)
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
cp .env.example .env   # add OPENAI_API_KEY if you have one
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

## Public knowledge base

By default Groundline can load a **Wikipedia CC BY-SA corpus** (~30+ articles on AI, ML, CS, and science) from `backend/corpus/wikipedia/`.

```bash
cd backend
python3 scripts/download_wikipedia_corpus.py   # fetch / refresh articles
PYTHONPATH=. python3 scripts/ingest_corpus.py  # embed into FAISS (replaces current KB)
```

Restart the API after ingesting so it reloads the index.

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
  rag/       # embeddings + FAISS store
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
