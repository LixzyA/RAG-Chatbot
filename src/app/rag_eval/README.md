# RAGAS evaluation

This runner evaluates the application's live `RAGChain`, not a mock or a separate
RAG client. It uses the exact chunks captured in `RAGTraceBuilder` as the Ragas
retrieved context.

Run it from `src/app`:

```powershell
$env:OPENROUTER_API_KEY = "your-openrouter-key"
uv run python rag_eval/evals.py
```

Alternatively, put `OPENROUTER_API_KEY` in `rag_eval/.env`; the runner loads it.

The CSV needs a `question` column. Add a `reference` answer derived from a
stored Chroma chunk for the reference-dependent metrics. Set
`expect_no_context` to `true` for a deliberate out-of-corpus control:

```csv
question,reference,expect_no_context,reference_chunk_id,reference_source
Which god do Geri and Freki accompany?,Geri and Freki are two wolves that accompany Odin.,false,sha256:85f12e5f4953f3c5,https://en.wikipedia.org/wiki/Geri_and_Freki
What is the RAG Chatbot refund policy?,,true,,
```

Every run writes per-question answers, retrieved context, and scores to
`rag_eval/evals/experiments/ragas_results.json`. Without a `reference`, it
reports only faithfulness. With one, it also reports answer correctness,
context precision, and context recall. An `expect_no_context` row records
`no_context_retrieval` as 1 when retrieval returns no chunks.
