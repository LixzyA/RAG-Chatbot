import asyncio
import logging
import time
from collections.abc import AsyncIterable
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.core.generation.llm_client import get_llm_client
from app.core.generation.prompt_builder import (
    build_rag_prompt,
    get_generation_system_prompt,
)
from app.core.generation.response_parser import parse_sse_chunk
from app.core.orchestration.query_processor import QueryProcessor
from app.core.pipeline.embedder import Embedder
from app.core.retrieval.vector_store import VectorStore
from app.models.rag_trace import RAGTraceBuilder
from app.utils.exceptions import LLMException
from app.core.orchestration.feedback import SelfFeedbackLoop
from huggingface_hub import ChatCompletionInputStreamOptions

logger = logging.getLogger(__name__)

# Query-type and strategy constants used in the transform result and feedback loop.
QUERY_TYPE_FEEDBACK_REWRITE = "feedback_rewrite"
STRATEGY_PASSTHROUGH = "passthrough"
STRATEGY_RE_RETRIEVE_PREFIX = "re_retrieve:"


def _doc_to_dict(doc: Document) -> dict[str, Any]:
    """Convert a LangChain Document to a plain dict for trace persistence."""
    return {
        "id": doc.id,
        "content": doc.page_content,
        "metadata": dict(doc.metadata or {}),
    }


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
        self.reranker = reranker
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
        bypass_transform: bool = False,
        filter: dict[str, Any] | None = None,
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

        When ``bypass_transform`` is ``True`` (used by the self-feedback re-retrieval
        path), the query is used as-is without classification or rewriting.

        ``filter`` is an optional Chroma ``where`` clause forwarded to the
        vector leg of hybrid search (BM25 cannot apply metadata filters). It
        is captured verbatim onto the trace for downstream analysis.

        Sync CPU work (Chroma/BM25 + cross-encoder) runs on a worker thread via
        ``asyncio.to_thread`` so the FastAPI event loop stays responsive.

        If a ``builder`` is supplied it is populated with:
          * ``original_query``                — the raw query
          * ``transformation_technique``      — e.g. ``"rewrite"``, ``"hyde"``, ``"passthrough"``
          * ``transformed_query``             — the rewritten/decomposed/HyDE text(s)
          * ``filters_applied``               — the metadata filter dict (or ``None``)
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

        # Trace-level: capture the filter verbatim so analysts can correlate
        # retrieval behaviour with scoped queries. ``hybrid_search`` only
        # applies it to the vector leg (BM25 has no native metadata filter).
        if builder is not None:
            builder.filters_applied = filter

        # 1. Transform the query (classification + expansion)
        ret_start = builder.start_retrieval() if builder else None

        if bypass_transform:
            transform_result = {
                "original_query": query,
                "query_type": QUERY_TYPE_FEEDBACK_REWRITE,
                "strategy": STRATEGY_PASSTHROUGH,
                "transformed_queries": [query],
                "confidence": 1.0,
            }
        else:
            transform_result = await self.query_processor.transform(query)

        if builder:
            if not bypass_transform:
                builder.original_query = query
            builder.transformation_technique = transform_result["strategy"]
            builder.transformed_query = " || ".join(
                transform_result["transformed_queries"]
            )

        transformed_queries = transform_result["transformed_queries"]
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

        # 2. For each transformed query, run hybrid search (pre-rerank) and rerank (post-rerank).
        # Hybrid candidates are (doc, score) tuples; reranker output is also (doc, score).
        all_pre_rerank: list[Document] = []
        all_reranked: list[Document] = []
        seen_pre: set[str] = set()
        seen_reranked: set[str] = set()
        candidate_multiplier = settings.hybrid_candidate_multiplier

        for single_query in transformed_queries:
            candidates = await asyncio.to_thread(
                self.vector_store.hybrid_search,
                single_query,
                top_k * candidate_multiplier,
                filter=filter,
            )

            # Track pre-rerank (dedup across transformed queries so a chunk
            # that appears in multiple rewrites is logged only once).
            for doc, _score in candidates:
                if doc.page_content not in seen_pre:
                    seen_pre.add(doc.page_content)
                    all_pre_rerank.append(doc)

            if self.use_reranker and self.reranker:
                rerank_start = builder.start_rerank() if builder else None
                # Reranker takes plain Documents — strip the tuples first.
                rerank_input: list[Document] = [doc for doc, _score in candidates]
                reranked_pairs: list[tuple[Document, float]] = await asyncio.to_thread(
                    self.reranker.rerank,
                    single_query,
                    rerank_input,
                    top_k,
                )
                if builder is not None and rerank_start is not None:
                    builder.stop_rerank(rerank_start)
                iter_pairs: list[tuple[Document, float]] = reranked_pairs
            else:
                # No real relevance signal — trust ordering, threshold disabled.
                iter_pairs = [(doc, 1.0) for doc, _score in candidates[:top_k]]

            for doc, score in iter_pairs:
                if self.use_reranker and self.reranker and score < threshold:
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
        stream_opt = ChatCompletionInputStreamOptions()
        stream_opt.include_usage = True
        usage = None
        try:
            response = await self.llm_client.chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": full_prompt},
                ],
                stream=True,
                stream_options=stream_opt,
            )
            async for chunk in response:
                text = parse_sse_chunk(chunk)
                if chunk.usage is not None:
                    usage = chunk.usage
                if text:
                    if builder is not None:
                        builder.llm_response += text
                    yield text
        except Exception as exc:
            raise LLMException(str(exc)) from exc
        finally:
            if builder is not None and llm_start is not None and usage is not None:
                builder.stop_llm(llm_start)
                builder.input_tokens = usage.prompt_tokens
                builder.output_tokens = usage.completion_tokens

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
        self_feedback_enabled: bool = False,
        filter: dict[str, Any] | None = None,
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

        ``filter`` is an optional metadata filter (Chroma ``where`` clause)
        passed through to retrieval. It is recorded on the trace so analysts
        can correlate scoped queries with retrieval behaviour. The same
        filter is reused for any self-feedback re-retrieval.
        """
        if threshold is None:
            threshold = settings.rag_min_relevance

        docs_pre_rerank, docs_final = await self.retrieve(
            query,
            top_k=top_k,
            threshold=threshold,
            builder=builder,
            filter=filter,
        )
        logger.info(
            "Retrieved %d candidate docs, %d final for query: %s",
            len(docs_pre_rerank),
            len(docs_final),
            query,
        )

        if not docs_final:
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

        #
        if not self_feedback_enabled:
            async for chunk in self.generate_stream(
                query,
                context,
                builder=builder,
                previous_query=previous_query,
            ):
                yield chunk
            return

        loop = SelfFeedbackLoop()
        iteration = 0
        best_answer = ""
        best_score = -1.0
        system_prompt: str | None = None
        start_time = time.perf_counter()

        while iteration <= settings.self_feedback_max_iterations:
            chunks: list[str] = []
            # gather the chunk for evaluation
            async for chunk in self.generate_stream(
                query,
                context,
                builder=builder if iteration == 0 else None,
                previous_query=previous_query,
                system_prompt=system_prompt,
            ):
                chunks.append(chunk)
            answer = "".join(chunks)

            try:
                result = await loop.evaluate(answer, docs_final, query)
            except Exception as exc:
                logger.exception(
                    "Feedback evaluation failed at iteration %d: %s", iteration, exc
                )
                best_answer = answer
                break

            if builder is not None:
                builder.feedback_scores.append(
                    {
                        "iteration": iteration,
                        "faithfulness": result.faithfulness,
                        "relevance": result.relevance,
                        "completeness": result.completeness,
                        "groundedness": result.groundedness,
                        "truthfulness": result.truthfulness,
                        "context_relevance": result.context_relevance,
                        "critique": result.critique,
                        "hallucination_detected": result.hallucination_detected,
                        "missing_from_context": result.missing_from_context,
                    }
                )
                builder.feedback_iterations = iteration + 1

            if result.passes:
                strategy_label = "passed"
                best_score = result.composite_score
                best_answer = answer

                total_latency_ms = (time.perf_counter() - start_time) * 1000
                if total_latency_ms > settings.self_feedback_max_latency_ms:
                    logger.warning(
                        "Self-feedback latency budget exceeded (%.0f ms > %d ms), "
                        "returning best answer (score=%.2f)",
                        total_latency_ms,
                        settings.self_feedback_max_latency_ms,
                        best_score,
                    )
                    break
            else:
                rewritten = await self.query_processor.rewrite_with_feedback(
                    query, result.critique
                )
                new_query = rewritten[0]
                strategy_label = f"{STRATEGY_RE_RETRIEVE_PREFIX} {new_query}"

                logger.debug(
                    "Self-feedback: re-retrieving with rewritten query: %s", new_query
                )
                _, docs_final = await self.retrieve(
                    new_query,
                    top_k=top_k,
                    threshold=threshold,
                    builder=builder,
                    bypass_transform=True,
                    filter=filter,
                )

                if not docs_final:
                    logger.warning(
                        "Self-feedback: re-retrieval returned no results for: %s",
                        new_query,
                    )
                    break

                context = self._build_context(docs_final)
                system_prompt = None

            if builder is not None and builder.feedback_scores:
                builder.feedback_scores[-1]["strategy"] = strategy_label

            iteration += 1
        else:
            # Max iterations reached without breaking
            logger.warning(
                "Self-feedback max iterations reached, returning best answer (score=%.2f)",
                best_score,
            )
            total_latency_ms = (time.perf_counter() - start_time) * 1000

        if builder is not None:
            builder.feedback_final_score = best_score if best_score >= 0 else None
            builder.llm_response = best_answer

        # Simulate streaming by yielding in small slices
        CHUNK_SIZE = 20
        for i in range(0, len(best_answer), CHUNK_SIZE):
            yield best_answer[i : i + CHUNK_SIZE]
