# Project Overview

RAG Chatbot — a FastAPI backend + React/Vite frontend. Primary users are general people who will try to chat to the system.
The product optimizes for:
- Factual, grounded answers
- Speed of response
- Fluidity of the system
- Easily traceable results

## Common Commands

### Backend
Run from repo root unless noted:

```bash
# Start dev server
cd src/app && uv run fastapi dev

# Print parsed config (env vars + defaults)
cd src/app && uv run python config.py

# Run one Python file
cd src/app && uv run python -m app.api.routes.generation

# Code quality (hook runs `uvx ruff check .; uvx ruff format` on every Edit/Write)
uvx ruff check src/app
uvx ruff format src/app

# Reindex codegraph
codegraph index .
```

### Frontend
```bash
cd src/frontend
npm run dev       # Vite dev server (http://localhost:5173)
npm run build     # Production build
npm run lint      # ESLint
npm run typecheck # tsc --noEmit
```

### Tools
```bash
# Batch ingestion (runs outside FastAPI — direct VectorStore construction)
cd src/app
uv run python ../../tools/batch_ingest.py path/to/docs/ --language en
```

## Architecture

### Layer map

Routes (`api/routes/`) are thin. Business logic lives in `services/` (singletons) and `core/` (framework-agnostic). `entity/` is SQLAlchemy ORM. `models/` is Pydantic request/response schemas.

```
api/routes/          HTTP/WS interface
├── services/        Singleton lifecycle + business logic entrypoints
│   ├── vector_db.py     (VectorStore, Reranker)
│   ├── rag_chain.py     (RAGChain)
│   ├── chat_history_service.py
│   └── auth_service.py
├── core/              Framework-agnostic (NO FastAPI imports, NO DB imports)
│   ├── orchestration/   (RAGChain, QueryProcessor)
│   ├── retrieval/       (VectorStore, CrossEncoderReranker)
│   ├── pipeline/        (chunker, document_loader, NER, embedder)
│   └── generation/      (llm_client, prompt_builder, response_parser)
├── entity/            SQLAlchemy ORM models (async aiosqlite)
└── models/            Pydantic v2 request/response schemas
```

**Rule**: `core/` never imports from `api/`, `entity/`, or `services/`.

### Singletons & startup lifecycle

`main.py` `lifespan()` calls `get_vector_store()` and `get_reranker()` to trigger creation, then loads the reranker & initializes the DB in background tasks. Routes depend on them via `api/dependencies.py` (`get_rag_chain_dep`, `get_vector_db`, `get_reranker_dep`). `services/rag_chain.py` wires these together lazily.

### Data flow (chat request)

`POST /chat` → `generation.py:chat()` → `RAGChain.run()`:

1. **Retrieve**: `RAGChain.retrieve()`
   - `QueryProcessor.transform()` → classify/rewrite/decompose/HyDE (router model)
   - `VectorStore.hybrid_search()` → BM25 + vector ensemble via `EnsembleRetriever`
   - `CrossEncoderReranker.rerank()` → rerank by cross-encoder score
   - Threshold filter (`settings.rag_min_relevance`, default 0.5)
2. **Generate**: `RAGChain.generate_stream()`
   - `build_rag_prompt()` assembles context + current query + optional `previous_query` (last user turn, used for conversational grounding only, no impact on retrieval)
   - `AsyncInferenceClient` streams the LLM response
3. **Stream**: SSE tokens to the frontend, terminated by `[DONE]`
4. **Persist**: `RAGTraceBuilder` is populated throughout the chain, then written to `rag_traces` by the route layer (`record_rag_trace`). Then the assistant message is saved to `messages`.

## Important Conventions

- **`core/` is framework-agnostic**: never import FastAPI, SQLAlchemy, or any route/entity/service module here. `core/` raises `AppException` subclasses (from `utils/exceptions.py`) for domain errors. Keep it pure so it is unit-testable.
- **Singletons**: `services/vector_db.py` owns `VectorStore` + `Reranker`, `services/rag_chain.py` owns `RAGChain`. Access them via the dependency functions in `api/dependencies.py` if inside a route; direct `get_*()` calls are fine too since they cache.
- **Async DB**: `services/` functions take `AsyncSession` as the first param (DI). Use `chat_history_service` and `auth_service` for DB access.
- **SSE protocol**: `POST /chat` streams SSE. Each event is a raw text chunk. The client (`frontend/src/pages/Chat.tsx`) listens for `data: ` lines and appends them. The stream ends with a `[DONE]` event.
- **Chroma DB**: `VectorStore.hybrid_search()` can return `[]` when the collection is empty — the RAG chain turns this into a friendly message to the user.

# Coding Conventions
- Helper functions do not need to have try except clause. Keep the try except clause in the worker function.
- Keep components focused and composable
- Extract repeated logics into helper functions
- prefer descriptive variable names over abbreviations
- Do not leave dead code or commented out blocks
- Do not use __future__ annotations
- Add comments only when intent is non obvious
- Keep API endpoint functions slim
- Prefer pydantic BaseModel over dataclass
- Export all public apis to __init__.py
- do not create a function inside a function
- do not create a class inside a class
- prefer creating custom exception class and catch using the exception class

# Files and component placements
- Add services into src/app/services folder
- Only create a new file if there are no file that is associated with the new module
- Do not create new abstraction for one off usage
- Prefer editing existing components over creating near duplicates

