"""RAG trace builder — populated by the chain, persisted by the route."""

import time
from typing import Any

from pydantic import BaseModel, Field


class RAGTraceBuilder(BaseModel):
    """Mutable per-request trace state.

    The chain's job is to populate this object during ``run()``. The route
    layer reads the populated builder after the SSE stream completes and
    writes one ``rag_traces`` row into the database.
    """

    # Query pipeline
    original_query: str = ""
    transformation_technique: str | None = None
    transformed_query: str | None = None

    # Stage outputs (list[dict] — each dict mirrors one Document snapshot)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    reranked_chunks: list[dict[str, Any]] = Field(default_factory=list)
    context_passed_to_llm: list[dict[str, Any]] = Field(default_factory=list)

    # LLM results
    llm_response: str = ""

    # Model identifiers (best-effort — empty if unknown)
    llm_model_name: str | None = None
    embedding_model_name: str | None = None

    # Per-stage latencies (milliseconds)
    retrieval_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    llm_latency_ms: float | None = None

    # Token accounting (filled when known; left None otherwise).
    input_tokens: int | None = None
    output_tokens: int | None = None

    # --- Self-Feedback Loop ---
    feedback_iterations: int = 0
    feedback_scores: list[dict[str, Any]] = Field(default_factory=list)
    feedback_final_score: float | None = None
    feedback_refinement_strategies: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 3)

    def start_retrieval(self) -> float:
        return time.perf_counter()

    def stop_retrieval(self, start: float) -> None:
        self.retrieval_latency_ms = self._ms(start)

    def start_rerank(self) -> float:
        return time.perf_counter()

    def stop_rerank(self, start: float) -> None:
        self.rerank_latency_ms = self._ms(start)

    def start_llm(self) -> float:
        return time.perf_counter()

    def stop_llm(self, start: float) -> None:
        self.llm_latency_ms = self._ms(start)
