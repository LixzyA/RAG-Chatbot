# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend (Python/FastAPI)

```bash
# Install dependencies (uses uv)
cd src/backend && uv sync

# Run the dev server
cd src/backend && fastapi dev main.py
# or: uvicorn main:app --reload --port 8000
# or: uv run fastapi

# RAG evaluation pipeline
cd src/backend && python evaluate_rag.py

# Batch-ingest JSONL corpus files into ChromaDB
cd src/backend && python scripts/ingest_corpus.py <corpus_dir>

# Pre-download embedding/reranker models
cd src/backend && python scripts/download_model.py
```

### Frontend (React/Vite/TypeScript)

```bash
cd src/frontend && npm install
cd src/frontend && npm run dev          # Start dev server (port 5173)
cd src/frontend && npm run build        # TypeScript check + Vite build
cd src/frontend && npm run lint         # ESLint
cd src/frontend && npm run format       # Prettier
cd src/frontend && npm run typecheck    # tsc --noEmit
cd src/frontend && npm run preview      # Preview production build
```

### Environment

- Backend requires Python == 3.11.15 with `uv` package manager
- Frontend requires Node.js >= 18
- LLM models are served via HuggingFace Inference API (configure HF_TOKEN in `.env`)
- By default LLM calls use `AsyncInferenceClient()` which reads `HF_TOKEN` from environment
- See `src/backend/pyproject.toml` for all Python dependencies
- PyTorch uses CUDA 12.8 index defined in `pyproject.toml` under `[[tool.uv.index]]`

## High-Level Architecture

This is a **RAG (Retrieval-Augmented Generation) Chatbot** with a dual-model router system, hybrid search, and cross-encoder reranking.

### Backend Structure (`src/backend/`)

```
main.py                           # FastAPI app entry: lifespan (async startup), CORS, /health
api.py                            # Router registration (mounts auth, chat, file_mgt controllers)
exception.py                      # All custom HTTP exceptions (HTTPException subclasses)
logger.py / logger.yaml           # Structured logging (JSON + console, rotating file handler)

auth/                             # JWT-based authentication
  controller.py                   # POST /auth/register, /auth/login, GET /auth/me
  service.py                      # Password hashing (argon2 via pwdlib), JWT create/decode
  dependencies.py                 # FastAPI dependency injection: AuthenticatedUser, OptionalAuthenticatedUser
  models.py                       # Pydantic request/response schemas

chat/                             # Core RAG chat logic
  controller.py                   # POST /chat/v2 (SSE streaming), history CRUD endpoints
  service.py                      # Query routing, prompt building, streaming LLM calls
  router.py                       # Llama-3.2-1B-Instruct query classifier (topic → specialist/generalist)
  core.py                         # AsyncInferenceClient singleton init, healthcheck
  history.py                      # Chat history persistence (SQLAlchemy/SQLite)
  models.py                       # Pydantic schemas (ChatQueryRequest, ChatHistoryResponse, etc.)
  prompt/                         # System prompt templates (.txt files)
    specialist.txt                # Strict RAG prompt for serious/hard topics
    generalist.txt                # Conversational RAG prompt for general topics

file_mgt/                         # Document upload & ingestion
  controller.py                   # POST /files/upload
  service.py                      # PDF parsing (pdfminer), RecursiveCharacterTextSplitter chunking
  models.py                       # UploadFileResponse schema

vectordb/                         # Vector database & search
  core.py                         # ChromaDB wrapper: embedding, hybrid search (BM25 + vector), reranked_search
  custom_embeddings.py            # LangChain Embeddings adapter for sentence-transformers/all-MiniLM-L6-v2
  reranker.py                     # FlagReranker (BAAI/bge-reranker-v2-m3) cross-encoder reranker

entity/                           # SQLAlchemy ORM models
  base.py                         # Engine, session factory, init_db (auto-creates tables)
  user.py                         # User table (id, username, email, password_hash, ...)
  chat_history.py                 # Chat sessions table (session_uuid, user_id, title, timestamps)
  chat_message.py                 # Messages table (session_id, role, content, token_count)

scripts/
  download_model.py               # Pre-cache embedding + reranker models
  ingest_corpus.py                # CLI tool to batch-ingest JSONL files into ChromaDB
```

### Frontend Structure (`src/frontend/`)

```
src/App.tsx                       # BrowserRouter, AuthProvider, Navbar, Routes
src/main.tsx                      # Entry point
src/index.css                     # Tailwind CSS v4 imports + @theme base styles

src/contexts/AuthContext.tsx       # Auth state (token, user), login/register/logout, localStorage persistence

src/lib/
  api.ts                          # apiFetch wrapper — prepends http://localhost:8000/, injects Bearer token
  utils.ts                        # Tailwind class merge utility (cn())

src/components/
  Navbar.tsx                      # Top nav with auth-aware links
  ProtectedRoute.tsx              # Route guard — redirects to /login if unauthenticated
  theme-provider.tsx               # Light/dark theme toggle
  ui/                             # shadcn/ui components (button, card, input, alert-dialog)

src/pages/
  Chat.tsx                        # Main chat interface — SSE streaming, history sidebar, markdown rendering
  Files.tsx                       # Document upload interface
  Home.tsx                        # Landing page
  Login.tsx / Register.tsx        # Auth pages
```

### Key Data Flows

**Chat Flow:** User query → `/chat/v2` SSE endpoint → parallel context retrieval (hybrid search → reranker) + query classification (router model) → routed to Specialist or Generalist LLM → streamed response back via SSE → messages saved to SQLite.

**Upload Flow:** File upload → `/files/upload` → file type validation (.txt, .pdf) → text extraction (pdfminer for PDF) → RecursiveCharacterTextSplitter (tiktoken, chunk_size=1024, overlap=20%) → embedding (all-MiniLM-L6-v2) → ChromaDB storage → BM25 cache update.

### Key Architecture Decisions

- **Dual-model router**: Llama-3.2-1B-Instruct classifies queries into serious topics (legal, medical, financial, technical, scientific, security) → routed to specialist (Llama-4-Scout-17B) vs generalist (Llama-3.1-8B). Classification uses JSON output with confidence threshold (0.7).
- **Hybrid search**: Ensemble of BM25 (keyword, weight 0.3) + vector similarity (embedding, weight 0.7), followed by cross-encoder reranking (BAAI/bge-reranker-v2-m3).
- **ChromaDB**: Local persistent vector database with SQLite backend. BM25 index rebuilt from a pickled document cache.
- **SQLite for chat history**: Async SQLAlchemy with aiosqlite, WAL mode, soft-delete for chat sessions.

### Environment Variables

Referenced in source — create a `.env` file in `src/backend/`:

| Variable | Default | Description |
|---|---|---|
| `HUGGINGFACEHUB_API_TOKEN` / `HF_TOKEN` | — | HuggingFace Inference API token |
| `SPECIALIST_MODEL` | meta-llama/Llama-4-Scout-17B-16E-Instruct | Specialist LLM |
| `GENERALIST_MODEL` | meta-llama/Llama-3.1-8B-Instruct | Generalist LLM |
| `ROUTER_MODEL` | meta-llama/Llama-3.2-1B-Instruct | Router/classifier LLM |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | Cross-encoder reranker |
| `RERANKER_ENABLED` | true | Toggle reranker on/off |
| `RERANKER_DEVICE` | cpu | Device for reranker ("cpu" or "cuda") |
| `JWT_SECRET_KEY` | change-me-in-production... | JWT signing secret |
| `JWT_EXPIRE_MINUTES` | 1440 | Token expiry (24h) |
| `DATABASE_URL` | sqlite+aiosqlite:///data/app.db | SQLite connection string |
| `DB_DIR` | src/backend/data/ | Database directory |

### Evaluation

`evaluate_rag.py` runs a custom RAG evaluation pipeline: loads questions from `evaluation.jsonl`, retrieves context, generates answers, and scores them on groundedness, relevance, and standalone quality using Llama-4-Scout as a judge.
