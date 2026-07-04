# Self-feedback loop design for the RAG backend

## Flow

```
Query processor  --------------------------+
  (rewrite query on retry)                  |
       |                                     |
       v                                     |
Hybrid retrieval                             |
  (BM25 + vector + rerank)                   |
       |                                     |
       v                                     |
Draft generation                             |
  (RAGChain composes answer)                 |
       |                                     |
       v                                     |
Self-critic                                  |
  (score groundedness, relevance)            |
       |                                     |
       +--- fail, retries left --- retry ----+   (max 1-2 retries)
       |
       v (pass, or max retries hit)
Return to user
  (or flagged as low-confidence)
```

## Where each piece lives

| Component | Location | Why |
|---|---|---|
| `SelfCorrectionLoop` (controller) | `core/orchestration/` | Framework-agnostic, orchestrates the retry cycle — same tier as `RAGChain`/`QueryProcessor` |
| `Critic` (groundedness + relevance scorer) | `core/generation/` | It's an LLM call like `llm_client`, just with a scoring prompt instead of a chat prompt |
| `SelfCheckResult`, `FeedbackRecord` schemas | `models/` | Pydantic v2, same as other request/response schemas |
| `feedback_logs` table | `entity/` | Async SQLAlchemy, persists every iteration for offline analysis |
| Wiring/lifecycle | `services/rag_chain.py` | `RAGChain` service becomes the DI entrypoint that owns a `SelfCorrectionLoop` instance |
| `/chat` route change | `api/routes/` | None needed — the loop is internal to `RAGChain.generate()`; the API contract stays the same |

## Design notes specific to this stack

**Critic prompt, not a separate model.** Use the existing dual-LLM routing — route the critic call to the cheaper/faster model in the pair. It only needs to output a small JSON object: `{grounded: bool, relevant: bool, confidence: float, reason: str}`. Feed it the retrieved chunks plus the draft answer, and ask it to flag unsupported claims. This keeps latency bounded since no new infra is needed.

**Retry ≠ always re-retrieve.** Two distinct failure modes need different fixes:
- **Ungrounded** (answer states things the context doesn't support) → regenerate with the critic's `reason` injected into the prompt as a correction instruction. No new retrieval needed.
- **Irrelevant** (retrieved context doesn't match query intent) → reformulate the query and re-retrieve. `QueryProcessor` already exists for this — add a `rewrite_with_feedback(original_query, critic_reason)` method.

**Cap retries at 1–2.** Each retry costs a full retrieval + generation + critique round trip. Beyond 2, diminishing returns and latency blow past what's tolerable for a chat interface — return the best attempt with a low-confidence flag instead of looping indefinitely.

**Log every iteration, not just the final one.** The `feedback_logs` table should capture:

```
session_id, iteration, query_used, retrieved_chunk_ids,
draft_answer, critic_score, critic_reason
```

for every pass. This becomes a dataset for evaluating whether the reranker or embedder needs tuning — patterns like "queries mentioning BI rate decisions consistently fail groundedness on iteration 1" become visible over time.

**Keep it optional per-request.** Add a `self_check: bool = False` flag on the chat request schema so the latency cost vs. answer quality tradeoff can be A/B tested before turning it on by default.
