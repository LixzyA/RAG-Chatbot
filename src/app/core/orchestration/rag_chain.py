"""RAG Chain — the main orchestrator.

Wires together retrieval, reranking, and generation into a single callable chain.
Also captures per-request trace data via ``RAGTraceBuilder`` so the route layer
can persist one ``rag_traces`` row per query.

Source: backend/chat/service.py (hybrid search + prompt building + LLM call flow).
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.core.generation.llm_client import get_llm_client
from app.core.generation.prompt_builder import build_rag_prompt, get_system_prompt
from app.core.generation.response_parser import parse_sse_chunk
from app.core.orchestration.query_processor import QueryProcessor
from app.core.retrieval.vector_store import VectorStore
from app.utils.exceptions import LLMException

import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Trace builder — populated by the chain, persisted by the route
# ------------------------------------------------------------------

@dataclass
class RAGTraceBuilder:
    """Mutable per-request trace state.

    The chain's job is to populate this object during ``run()``. The route
    layer reads the populated builder after the SSE stream completes and
    writes one ``rag_traces`` row into the database. Keeping the shape
    here (next to the producer) avoids leaking ORM concerns into ``core/``.
    """

    # Query pipeline
    original_query: str = ""
    transformation_technique: str | None = None
    transformed_query: str | None = None

    # Stage outputs (list[dict] — each dict mirrors one Document snapshot)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    reranked_chunks: list[dict[str, Any]] = field(default_factory=list)
    context_passed_to_llm: list[dict[str, Any]] = field(default_factory=list)

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


# ------------------------------------------------------------------
# Chain
# ------------------------------------------------------------------

class RAGChain:
    """Orchestrates the full RAG pipeline.

    Usage::

        chain = RAGChain(vector_store=store, use_reranker=True)
        async for chunk in chain.run("What is Python?", builder=builder):
            print(chunk, end="")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        use_reranker: bool = True,
        reranker: Any | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.use_reranker = use_reranker
        # Allow callers (e.g. tests) to inject a pre-built reranker; fall
        # back to whatever the vector store already has attached.
        self.reranker = reranker or getattr(vector_store, "reranker", None)
        self.query_processor = QueryProcessor()
        self.llm_client = get_llm_client()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        builder: RAGTraceBuilder | None = None,
    ) -> tuple[list, list]:
        """Retrieve, optionally rerank, and return ``(pre_rerank_docs, final_docs)``.

        If a ``builder`` is supplied it is populated with:
          * ``original_query``                — the raw query
          * ``transformation_technique``      — e.g. ``"rewrite"``, ``"hyde"``, ``"passthrough"``
          * ``transformed_query``             — the rewritten/decomposed/HyDE text(s)
          * ``retrieved_chunks``              — pre-rerank docs (BM25 + vector) as dicts
          * ``reranked_chunks``               — docs after the cross-encoder reranker as dicts
          * ``context_passed_to_llm``         — the dedup'd + sliced list that actually
                                                 reached the prompt, as dicts
          * ``retrieval_latency_ms``          — total wall-clock for hybrid retrieval
          * ``rerank_latency_ms``             — wall-clock for cross-encoder rerank stage
          * ``embedding_model_name``          — best-effort embedding model identifier
        """
        # 1. Transform the query (classification + expansion)
        ret_start = builder.start_retrieval() if builder else None
        transform_result = await self.query_processor.transform(query)

        if builder:
            builder.original_query = query
            builder.transformation_technique = transform_result["strategy"]
            builder.transformed_query = " || ".join(transform_result["transformed_queries"])

        transformed_queries = transform_result["transformed_queries"]
        logger.info(
            "Query transform: type=%s strategy=%s confidence=%.2f",
            transform_result["query_type"],
            transform_result["strategy"],
            transform_result["confidence"],
        )

        # Try to surface the embedding model name (best-effort; doesn't load weights).
        if builder and builder.embedding_model_name is None:
            try:
                from app.core.pipeline.embedder import Embedder  # local to avoid cycles
                builder.embedding_model_name = Embedder().model_name
            except Exception:  # noqa: BLE001
                pass

        def _doc_to_dict(doc) -> dict[str, Any]:
            return {
                "content": doc.page_content,
                "metadata": dict(doc.metadata or {}),
            }

        # 2. For each transformed query, run hybrid search (pre-rerank) and rerank (post-rerank)
        all_pre_rerank: list = []
        all_final: list = []
        seen_pre: set[str] = set()
        seen_final: set[str] = set()
        candidate_multiplier = settings.hybrid_candidate_multiplier

        for t_query in transformed_queries:
            candidates = self.vector_store.hybrid_search(t_query, k=top_k * candidate_multiplier)

            # Track pre-rerank (dedup across transformed queries so a chunk
            # that appears in multiple rewrites is logged only once).
            for doc in candidates:
                if doc.page_content not in seen_pre:
                    seen_pre.add(doc.page_content)
                    all_pre_rerank.append(doc)

            if self.use_reranker:
                rerank_start = builder.start_rerank() if builder else None
                if self.reranker:
                    reranked = self.reranker.rerank(t_query, candidates, top_k=top_k)
                else:
                    reranked = candidates[:top_k]
                if builder is not None and rerank_start is not None:
                    builder.stop_rerank(rerank_start)
                final_iter = reranked
            else:
                final_iter = candidates[:top_k]

            for doc in final_iter:
                if doc.page_content not in seen_final:
                    seen_final.add(doc.page_content)
                    all_final.append(doc)

        # Cap to requested top_k for both lists.
        pre_rerank = all_pre_rerank[:top_k]
        final = all_final[:top_k]

        if builder is not None:
            builder.retrieved_chunks = [_doc_to_dict(d) for d in pre_rerank]
            builder.reranked_chunks = [_doc_to_dict(d) for d in final]
            builder.context_passed_to_llm = [_doc_to_dict(d) for d in final]
            if ret_start is not None:
                builder.stop_retrieval(ret_start)

        return pre_rerank, final

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_context(self, documents: list) -> str:
        return "\n\n".join(doc.page_content for doc in documents if doc.page_content)

    async def generate_stream(
        self,
        query: str,
        context: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        builder: RAGTraceBuilder | None = None,
    ) -> AsyncIterable[str]:
        """Yield text chunks from the LLM.

        When a ``builder`` is supplied, ``llm_model_name`` is set up front and
        ``llm_latency_ms`` is filled in at the end. ``llm_response`` is
        accumulated chunk-by-chunk so the caller can use it for persistence.
        """
        model = model or settings.generalist_model
        system = system_prompt or get_system_prompt("generalist")
        full_prompt = build_rag_prompt(query, context)

        if builder is not None:
            builder.llm_model_name = model

        llm_start = builder.start_llm() if builder else None
        try:
            response = await self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": full_prompt},
                ],
                stream=True,
            )
            async for chunk in response:
                text = parse_sse_chunk(chunk)
                if text:
                    if builder is not None:
                        builder.llm_response += text
                    yield text
        except Exception as exc:
            raise LLMException(str(exc)) from exc
        finally:
            if builder is not None and llm_start is not None:
                builder.stop_llm(llm_start)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        *,
        top_k: int = 10,
        builder: RAGTraceBuilder | None = None,
    ) -> AsyncIterable[str]:
        """End-to-end RAG pipeline: retrieve → prompt → stream.

        ``builder``, if provided, is populated in-place with everything the
        route layer needs to persist a ``rag_traces`` row.
        """
        docs_pre, docs_final = await self.retrieve(query, top_k=top_k, builder=builder)
        logger.info(
            "Retrieved %d candidate docs, %d final for query: %s",
            len(docs_pre),
            len(docs_final),
            query,
        )

        if not docs_final:
            msg = "I could not find any relevant information in the provided documents."
            if builder is not None:
                builder.llm_response = msg
            yield msg
            return

        context = self._build_context(docs_final)
        async for chunk in self.generate_stream(query, context, builder=builder):
            yield chunk
