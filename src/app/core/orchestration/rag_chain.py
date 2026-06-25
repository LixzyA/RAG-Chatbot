import asyncio
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.core.generation.llm_client import get_llm_client
from app.core.generation.prompt_builder import (
    build_rag_prompt,
    get_generation_system_prompt,
)
from app.core.generation.response_parser import parse_sse_chunk
from app.core.orchestration.query_processor import QueryProcessor
from app.core.retrieval.vector_store import VectorStore
from app.core.pipeline.embedder import Embedder
from langchain_core.documents import Document
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
        threshold: float | None = None,
        builder: RAGTraceBuilder | None = None,
    ) -> tuple[list[Document], list[Document]]:
        """Retrieve, optionally rerank, and return ``(pre_rerank_docs, final_docs)``.

        * ``pre_rerank_docs`` — deduped union of every hybrid-search hit across all
          transformed queries; capped at ``top_k`` for the trace payload.
        * ``final_docs`` — post-rerank (or pre-rerank fallback) hits whose score
          passes ``threshold``; capped at ``top_k`` before being passed to the LLM.

        ``threshold`` defaults to ``settings.rag_min_relevance`` (env:
        ``RAG_MIN_RELEVANCE``). When the reranker is disabled or unavailable,
        threshold filtering is skipped — there are no real relevance scores to
        compare against.

        Sync CPU work (Chroma/BM25 + cross-encoder) runs on a worker thread via
        ``asyncio.to_thread`` so the FastAPI event loop stays responsive.

        If a ``builder`` is supplied it is populated with:
          * ``original_query``                — the raw query
          * ``transformation_technique``      — e.g. ``"rewrite"``, ``"hyde"``, ``"passthrough"``
          * ``transformed_query``             — the rewritten/decomposed/HyDE text(s)
          * ``retrieved_chunks``              — pre-rerank docs (BM25 + vector) as dicts
          * ``reranked_chunks``               — docs after threshold filtering as dicts
          * ``context_passed_to_llm``         — the dedup'd + sliced list that actually
                                                 reached the prompt, as dicts
          * ``retrieval_latency_ms``          — total wall-clock for hybrid retrieval
          * ``rerank_latency_ms``             — wall-clock for cross-encoder rerank stage
          * ``embedding_model_name``          — best-effort embedding model identifier
        """
        if threshold is None:
            threshold = settings.rag_min_relevance

        # 1. Transform the query (classification + expansion)
        ret_start = builder.start_retrieval() if builder else None
        transform_result = await self.query_processor.transform(query)

        if builder:
            builder.original_query = query
            builder.transformation_technique = transform_result["strategy"]
            builder.transformed_query = " || ".join(
                transform_result["transformed_queries"]
            )

        transformed_queries = transform_result["transformed_queries"]
        # Cap fan-out — `decompose` can emit many sub-queries; re-running the
        # cross-encoder on each is linear in N. Three covers the common case
        # without blowing the latency budget.
        if len(transformed_queries) > 3:
            logger.info(
                "Capping transformed_queries: %d -> 3",
                len(transformed_queries),
            )
            transformed_queries = transformed_queries[:3]

        logger.info(
            "Query transform: type=%s strategy=%s confidence=%.2f",
            transform_result["query_type"],
            transform_result["strategy"],
            transform_result["confidence"],
        )

        # Embedding model name: cached class-level constant — no model load.
        if builder and builder.embedding_model_name is None:
            builder.embedding_model_name = Embedder.default_model_name()

        def _doc_to_dict(doc: Document) -> dict[str, Any]:
            return {
                "content": doc.page_content,
                "metadata": dict(doc.metadata or {}),
            }

        # 2. For each transformed query, run hybrid search (pre-rerank) and rerank (post-rerank).
        # Hybrid candidates are (doc, score) tuples; reranker output is also (doc, score).
        all_pre_rerank: list[Document] = []
        all_reranked: list[Document] = []
        seen_pre: set[str] = set()
        seen_reranked: set[str] = set()
        candidate_multiplier = settings.hybrid_candidate_multiplier
        # `False` when reranker is off / unavailable — skip threshold filtering
        # since the scores in that path are not real relevance values.
        reranker_active = bool(self.use_reranker and self.reranker)

        for t_query in transformed_queries:
            candidates = await asyncio.to_thread(
                self.vector_store.hybrid_search,
                t_query,
                top_k * candidate_multiplier,
            )

            # Track pre-rerank (dedup across transformed queries so a chunk
            # that appears in multiple rewrites is logged only once).
            for doc, _score in candidates:
                if doc.page_content not in seen_pre:
                    seen_pre.add(doc.page_content)
                    all_pre_rerank.append(doc)

            if reranker_active:
                rerank_start = builder.start_rerank() if builder else None
                # Reranker takes plain Documents — strip the tuples first.
                rerank_input: list[Document] = [doc for doc, _s in candidates]
                reranked_pairs: list[tuple[Document, float]] = await asyncio.to_thread(
                    self.reranker.rerank,
                    t_query,
                    rerank_input,
                    top_k,
                )
                if builder is not None and rerank_start is not None:
                    builder.stop_rerank(rerank_start)
                iter_pairs: list[tuple[Document, float]] = reranked_pairs
            else:
                # No real relevance signal — trust ordering, threshold disabled.
                iter_pairs = [(doc, 1.0) for doc, _s in candidates[:top_k]]

            for doc, score in iter_pairs:
                if reranker_active and score < threshold:
                    continue
                if doc.page_content in seen_reranked:
                    continue
                seen_reranked.add(doc.page_content)
                all_reranked.append(doc)

        # Both lists respect top_k — pre_rerank for downstream trace size, final
        # for LLM context size.
        pre_rerank_capped = all_pre_rerank[:top_k]
        final = all_reranked[:top_k]

        if builder is not None:
            builder.retrieved_chunks = [_doc_to_dict(d) for d in pre_rerank_capped]
            builder.reranked_chunks = [_doc_to_dict(d) for d in final]
            builder.context_passed_to_llm = [_doc_to_dict(d) for d in final]
            if ret_start is not None:
                builder.stop_retrieval(ret_start)

        return pre_rerank_capped, final

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
        previous_query: str | None = None,
    ) -> AsyncIterable[str]:
        """Yield text chunks from the LLM.

        When a ``builder`` is supplied, ``llm_model_name`` is set up front and
        ``llm_latency_ms`` is filled in at the end. ``llm_response`` is
        accumulated chunk-by-chunk so the caller can use it for persistence.

        ``previous_query`` is purely an LLM-context signal: it is injected into
        the user-turn prompt so the model can ground follow-up turns, but it
        does NOT influence retrieval (retrieval uses the current ``query``
        only — see :meth:`run`).
        """
        model = model or settings.generation_model
        system = system_prompt or get_generation_system_prompt()
        full_prompt = build_rag_prompt(query, context, previous_query=previous_query)

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
        threshold: float | None = None,
        builder: RAGTraceBuilder | None = None,
        previous_query: str | None = None,
    ) -> AsyncIterable[str]:
        """End-to-end RAG pipeline: retrieve → prompt → stream.

        ``builder``, if provided, is populated in-place with everything the
        route layer needs to persist a ``rag_traces`` row. ``threshold`` falls
        back to ``settings.rag_min_relevance`` when not supplied.

        ``previous_query`` is the user's immediately-prior turn (sourced by
        the route from chat history). It is forwarded into the user-turn
        prompt only — it does NOT influence retrieval, classification, or
        query rewriting, by design. Pass ``None`` for first-turn / anonymous
        requests and the prompt builder will simply omit the history block.
        """
        if threshold is None:
            threshold = settings.rag_min_relevance

        docs_pre, docs_final = await self.retrieve(
            query, top_k=top_k, threshold=threshold, builder=builder
        )
        logger.info(
            "Retrieved %d candidate docs, %d final for query: %s",
            len(docs_pre),
            len(docs_final),
            query,
        )

        if not docs_final:
            # ``.count()`` is a sync Chroma call — hand it to a worker thread
            # so the event loop stays responsive. ``self.vector_store.client``
            # is unconditionally built in ``VectorStore.__init__``, so the
            # null-check is dead and has been removed.
            collection = self.vector_store.client.get_or_create_collection(
                self.vector_store.collection_name
            )
            retrieval_total = await asyncio.to_thread(collection.count)
            if retrieval_total == 0:
                msg = (
                    "Your knowledge base is empty — please upload some documents "
                    "first, then ask your question again."
                )
            else:
                msg = (
                    "I could not find any relevant information in the provided "
                    "documents for that question."
                )
            if builder is not None:
                builder.llm_response = msg
            yield msg
            return

        context = self._build_context(docs_final)
        async for chunk in self.generate_stream(
            query,
            context,
            builder=builder,
            previous_query=previous_query,
        ):
            yield chunk
