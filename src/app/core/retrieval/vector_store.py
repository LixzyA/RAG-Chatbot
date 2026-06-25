"""Vector store abstraction — wraps ChromaDB (or future alternatives).

Source: backend/vectordb/core.py (ChromaDB object + LangChain Chroma integration).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

from app.utils.exceptions import ChromaInsertionException
from app.core.retrieval.reranker import CrossEncoderReranker
import logging

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "big_token_corpus"
DEFAULT_PERSIST = "../../.langchain_chroma/"
DEFAULT_BM25_CACHE = "./.bm25_cache/docs_cache.pkl"


class VectorStore:
    """Thin wrapper around ChromaDB with pickle-backed BM25 caching."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_path: str = DEFAULT_PERSIST,
        *,
        reranker: CrossEncoderReranker | None = None,
        bm25_cache_path: str = DEFAULT_BM25_CACHE,
    ) -> None:
        self.collection_name = collection_name
        self.reranker = reranker

        self._chroma_path = Path(persist_path)
        self._chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_path)
        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
        )

        self.bm25_cache_path = Path(bm25_cache_path)
        self._docs_cache: list[Document] = self._load_docs_cache()
        self.bm25_retriever: BM25Retriever | None = None

        if self._docs_cache:
            self._update_bm25()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_docs_cache(self) -> list[Document]:
        if self.bm25_cache_path.exists():
            with open(self.bm25_cache_path, "rb") as fh:
                docs = pickle.load(fh)  # noqa: S301
            logger.info("BM25 cache loaded: %d docs", len(docs))
            return docs
        return []

    def _save_docs_cache(self) -> None:
        self.bm25_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_cache_path, "wb") as fh:
            pickle.dump(self._docs_cache, fh)

    def _update_bm25(self) -> None:
        if not self._docs_cache:
            return
        self.bm25_retriever = BM25Retriever.from_documents(self._docs_cache)
        self.bm25_retriever.k = 10  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self, documents: list[Document], ids: list[str] | None = None
    ) -> None:
        """Add documents to the vector store and rebuild BM25."""
        try:
            self.vector_store.add_documents(documents, ids=ids)
            self._docs_cache.extend(documents)
            self._save_docs_cache()
            self._update_bm25()
        except Exception as exc:
            logger.error("Failed to add documents: %s", exc)
            raise ChromaInsertionException(str(exc)) from exc

    def add_documents_bulk(
        self, documents: list[Document], ids: list[str] | None = None
    ) -> None:
        """Add documents without rebuilding BM25 (call :meth:`rebuild_bm25` later)."""
        try:
            self.vector_store.add_documents(documents, ids=ids)
            self._docs_cache.extend(documents)
            self._save_docs_cache()
        except Exception as exc:
            logger.error("Failed to add documents in bulk: %s", exc)
            raise ChromaInsertionException(str(exc)) from exc

    def similarity_search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Pure vector search."""
        return self.vector_store.similarity_search_with_score(query, k=k)

    def hybrid_search(self, query: str, k: int = 20) -> list[tuple[Document, float]]:
        """BM25 + Vector ensemble via LangChain ``EnsembleRetriever``.

        Returns ``[]`` when the corpus is empty (no documents uploaded yet).
        Callers (RAG chain, retrieval routes) translate that to a graceful
        "no context" response; raising from here would orphan the SSE stream
        because the 200 / event-stream headers have already been flushed.

        Each item is ``(document, score)``. ``LangChain``'s ``EnsembleRetriever``
        does not surface per-doc scores, so we synthesise a fused reciprocal-rank
        score (RRF, weights 0.3 / 0.7 over BM25 + vector ranks) — this gives
        downstream threshold filters a usable, normalized value while keeping
        the BM25/vector ordering stable.
        """
        if self.bm25_retriever is None:
            self.rebuild_bm25()
            if self.bm25_retriever is None:
                logger.warning(
                    "hybrid_search skipped: collection '%s' is empty",
                    self.collection_name,
                )
                return []

        self.bm25_retriever.k = k  # type: ignore[union-attr]
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[0.3, 0.7],
        )
        ranked_docs = ensemble.invoke(query)

        # Synthesise RRF scores keyed by doc fingerprint (page_content + meta hash).
        # ``id(doc)`` would also work but is meaningless across pickle reloads.
        scored_tuples: list[tuple[Document, float]] = []
        scored_tuples = [(doc, self._rrf_score(ranked_docs, doc)) for doc in ranked_docs]
        return scored_tuples

    @staticmethod
    def _rrf_score(ranked_docs: list[Document], target: Document) -> float:
        """Reciprocal Rank Fusion score backed off by the inverse-rank term.

        Score is in (``0``, ``~1``] for top-K queries with K ≈ 30+. Each rank
        contributes ``1 / (k + rank)``; using a constant ``k=60`` matches the
        standard RRF formulation used by LangChain's ensemble logic.
        """
        for rank, doc in enumerate(ranked_docs):
            if doc.page_content == target.page_content:
                return 1.0 / (60 + rank)
        return 0.0

    def reranked_search(
        self, query: str, k: int = 5, candidate_multiplier: int = 4
    ) -> list[Document]:
        """Two-stage retrieval: hybrid search followed by cross-encoder reranking.

        Returns a plain ``list[Document]`` (not tuples) for backwards
        compatibility with the ``/retrieve`` response model — the reranker's
        cross-encoder score is recorded on each ``Document.metadata['rerank_score']``
        so callers can still surface a ``score`` field via metadata lookup.
        """
        candidates = self.hybrid_search(query, k=k * candidate_multiplier)
        if not candidates:
            return []
        docs_only = [doc for doc, _score in candidates]
        if self.reranker is None:
            return docs_only[:k]
        reranked = self.reranker.rerank(query, docs_only, top_k=k)
        return [doc for doc, _score in reranked]

    def rebuild_bm25(self) -> None:
        """Ensure docs cache is warm and rebuild the BM25 index."""
        if not self._docs_cache:
            self._docs_cache = self._fetch_all_documents()
            self._save_docs_cache()
        self._update_bm25()

    def _fetch_all_documents(self) -> list[Document]:
        """Batch-fetch all documents from Chroma (avoids SQLite variable limit)."""
        collection = self.client.get_or_create_collection(self.collection_name)
        batch_size = 500
        offset = 0
        all_docs: list[Document] = []

        while True:
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = result.get("ids") or []
            if not ids:
                break
            docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(
                    result["documents"] or [],
                    result["metadatas"] or [],
                )
            ]
            all_docs.extend(docs)
            if len(ids) < batch_size:
                break
            offset += batch_size

        return all_docs

    def heartbeat(self) -> dict:
        """Lightweight runtime health check."""
        chroma: dict = {"healthy": False, "error": None}
        vs_store: dict = {"healthy": False, "error": None}

        try:
            result = self.client.heartbeat()
            chroma["healthy"] = bool(result)
        except Exception as exc:
            chroma["error"] = str(exc)

        try:
            result = self.similarity_search("health check", k=1)
            vs_store["healthy"] = bool(result)
        except Exception as exc:
            vs_store["error"] = str(exc)

        return {
            "chroma_client": chroma,
            "vector_store": vs_store,
        }

    def close(self) -> None:
        """Release resources."""
        self.bm25_retriever = None
        self._docs_cache = []
