# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend (Python/FastAPI) — root: `src/app/`

```bash
cd src/app
uv sync                          # install (venv lives at src/app/.venv)
fastapi dev main.py              # or: uv run fastapi dev
uvicorn main:app --reload --port 8000
```

Requires **Python == 3.11.15** (pinned in `src/app/pyproject.toml`). PyTorch index is declared in the same file.

### Frontend (React/Vite/TypeScript) — root: `src/frontend/`

```bash
cd src/frontend
npm install
npm run dev          # port 5173
npm run build        # tsc -b && vite build
npm run lint         # ESLint
npm run typecheck    # tsc --noEmit
npm run format       # Prettier
```

Node.js ≥ 18 (package.json pins Vite 7 / React 19 / TS 5.9).

### Environment

Create `.env` at repo root (read by `src/app/config.py` via `PROJECT_ROOT/.env`). All LLM calls go through `huggingface_hub`'s `AsyncInferenceClient()`, which reads `HF_TOKEN`. See `.env.example` for the canonical key list. Reference documentation for all env vars is in `src/app/config.py` (one field per var, with `alias=` naming the env name).

## High-Level Architecture

A two-model **RAG chatbot**: user query → optional query transformation → hybrid retrieval (BM25 + vector) → cross-encoder rerank → streamed back from a single generation LLM via SSE.

- **`router_model`** (small, ~1B) handles all query-classification and query-transformation work in `core/orchestration/query_processor.py`: classify topic → rewrite / decompose / HyDE / passthrough.
- **`generation_model`** (large) produces the final answer in `core/orchestration/rag_chain.py:generate_stream` from the top-k reranked chunks.

The previous specialist/generalist split has been collapsed into the single generation model. `core/generation/prompt_builder.py` exposes one prompt: `get_generation_system_prompt()`.

### Backend layout (`src/app/`)

```
main.py                          # FastAPI app + lifespan (init singletons + DB, shutdown resets)
api/
  dependencies.py                 # DI providers (get_db, get_rag_chain_dep, auth guards)
  routes/
    auth.py                      # POST /auth/{register,login}, GET /auth/me
    generation.py                # POST /chat/ (SSE) + /chat/histories CRUD + rag_traces persistence
    health.py                    # GET /health
    ingestion.py                 # POST /ingest
    retrieval.py                 # POST /retrieve
core/                            # framework-agnostic — NO FastAPI imports, reusable + testable
  orchestration/
    query_processor.py           # classify / rewrite / decompose / HyDE / passthrough
    rag_chain.py                 # RAG pipeline + RAGTraceBuilder (per-request observability)
  generation/                    # llm_client, prompt_builder, response_parser
  retrieval/                     # vector_store (ChromaDB + EnsembleRetriever) + reranker
  pipeline/                      # document_loader, chunker, text_splitter, embedder, ranker (RRF)
entity/                          # SQLAlchemy ORM (Base, User, ChatSession, ChatMessage, RAG_traces)
models/                          # Pydantic v2 request/response + Document models
services/                        # thin: singletons (VectorStore, Reranker, RAGChain) + DB-aware fns
                                 # all fns take AsyncSession via DI; chat_history_service, auth_service
utils/                           # logger (setup_logging from main.py), exceptions, telemetry
```

**Layer rule:** `core/` is framework-agnostic and never imports `api/`, `entity/`, or `models/`. `services/` owns the session-bound/singleton side. `api/routes/` is the only seam that ties them together (and is where `RAGTraceBuilder` → `RAG_traces` persistence happens — `api/routes/generation.py:record_rag_trace`).

### Singletons

`VectorStore`, `Reranker`, and `RAGChain` are factored as lazy singletons in `services/`. They are materialised by `lifespan` in `main.py` (reranker is loaded in a background task so it doesn't block startup) and reset on shutdown.

### Frontend layout (`src/frontend/src/`)

```
App.tsx + main.tsx               # BrowserRouter, AuthProvider, theme provider
contexts/AuthContext.tsx         # token + user state, localStorage persistence
lib/api.ts                       # apiFetch — prepends http://localhost:8000, injects Bearer token
components/Navbar.tsx, ProtectedRoute.tsx, ui/ (shadcn)
pages/Chat.tsx, Files.tsx, Home.tsx, Login.tsx, Register.tsx
```

Tailwind CSS v4 (`@tailwindcss/vite`), React 19, shadcn/ui primitives.

## Key Data Flows

**Chat (`POST /chat/`, SSE):**
1. Route validates auth (optional), resolves `chat_id` → int FK via `chat_history_service.get_internal_session_id`, saves user message.
2. `RAGChain.run(prompt, builder=RAGTraceBuilder())`:
   - `QueryProcessor` classifies topic (or rewrites / HyDE-decomposes if `QUERY_TRANSFORM_ENABLED=true`).
   - `VectorStore` runs hybrid (BM25 + cosine) ensemble → returns N×`top_k` candidates, then `Reranker` (BAAI/bge-reranker-v2-m3, lazy-loaded, optional via `RERANKER_ENABLED`) cuts to `top_k`.
   - `prompt_builder` injects chunks as context into the single generation system prompt.
   - `AsyncInferenceClient.chat.completions.create(stream=True)` yields tokens; route passes chunks through SSE.
3. After `[DONE]`: route writes one `RAG_traces` row from the populated `RAGTraceBuilder` (latencies + per-stage snapshots), then saves the assistant message. Trace-write failures are logged + swallowed — they never break the response.

**Ingestion (`POST /ingest`):** file → extension validation → text extraction (pdfminer.six for PDFs) → `RecursiveCharacterTextSplitter` (tiktoken, `chunk_size=1024`, 20% overlap) → embed (`sentence-transformers/all-MiniLM-L6-v2`) → ChromaDB.

**Auth:** JWT (HS256) issued by `auth/service.py`, validated by `api/dependencies.py:AuthenticatedUser` / `OptionalAuthenticatedUser`. Passwords hashed with argon2 via `pwdlib`.

## Architectural decisions worth knowing

- **Two models in the system, set via env-overridable fields in `config.py`:**
  - `router_model` (default `meta-llama/Llama-3.2-1B-Instruct`, alias `ROUTER_MODEL`) — used by `QueryProcessor` for classify / rewrite / decompose / HyDE.
  - `generation_model` (default `google/gemma-4-31B-it`, alias `GENERATION_MODEL`) — used by `generate_stream` for the final answer and by `llm_client.healthcheck`.
- **Hybrid retrieval:** EnsembleRetriever weights BM25 0.3 / vector 0.7. `HYBRID_CANDIDATE_MULTIPLIER` (default 4) controls the pre-rerank fan-out.
- **DB:** SQLite via `aiosqlite` (async). Schema created via `entity/base.py:init_db` on startup. Chat sessions use soft-delete + auto-title.
- **History persistence:** Messages stored as JSON column on `ChatMessage`, keyed by string UUID `session_uuid` (clients see strings; FK to `chat_sessions.id` is the int PK).
- **Observability:** Inline RAG trace persistence (`api/routes/generation.py:record_rag_trace`) — no new service module, by design. Anonymous queries (no `chat_id`, no user) still emit a `rag_traces` row with both FKs `NULL` (nullable + `ON DELETE SET NULL`).
- **Embedding / reranker models:** Pre-downloaded on first use; reranker load is deferred to a `lifespan` background task so it never blocks server start.

## Working in this codebase

### CodeGraph is enabled

A `.codegraph/` index exists at the repo root — reach for `codegraph_explore` (with `mcp__codegraph__codegraph_explore` if available) **before** reading files or grepping when you need to understand or locate code. ONE call returns the verbatim source of the relevant symbols grouped by file plus the call path between them — treat returned source as already Read.

### CODE_REVIEW.md is the source of design truth

`CODE_REVIEW.md` at the repo root records the refactor's bug-fix history, layer-boundary rationale, and **three remaining open items**:

1. `SUGGESTION #4` — drop `src/app/py.typed` (PEP 561 marker) so type checkers honour inline annotations.
2. `SUGGESTION #5` — replace `# type: ignore` comments in `core/retrieval/vector_store.py` and `core/retrieval/reranker.py` with thin typed wrappers / `.pyi` stubs.
3. `SUGGESTION #8` — wire `tenacity` retry logic around the three LLM call sites (`core/generation/llm_client.py`, `core/orchestration/query_processor.py:_llm_call`, `core/orchestration/rag_chain.py:generate_stream`). `tenacity` is already installed.

Before touching code that touches these areas, check CODE_REVIEW.md for the prior reasoning.

### Practical reminders

- `setup_logging()` is called once from `main.py` — don't call it elsewhere.
- `core/` raises `utils/exceptions.py` framework-agnostic error types; only `api/routes/` translates them to `HTTPException`.
- Entity models use the `Mapped[...]` / `mapped_column(...)` declarative style (SQLAlchemy 2).
- Pydantic is v2 throughout (`ConfigDict`, `model_validator`, `model_dump`, `Field` with `alias=`).
- Frontend ↔ backend auth is a Bearer token from `localStorage.auth_token`, sent by `lib/api.ts:apiFetch` (set `requireAuth: false` for `/auth/*` and `/health`).
