"""RAG chain service — singleton lifecycle for the ``RAGChain`` orchestrator.

The ``RAGChain`` is stateless (it only calls LLMs and reads from the live
``VectorStore``) so it's safe — and cheaper — to construct once at startup
and reuse across requests. This avoids rebuilding the ``QueryProcessor``
instance on every chat turn.

Routes should obtain the chain through the FastAPI dependency
``get_rag_chain_dep`` (in ``app.api.dependencies``), which delegates here.
"""
from __future__ import annotations

import logging

from app.core.orchestration.rag_chain import RAGChain
from app.services.vector_db import get_vector_store

logger = logging.getLogger(__name__)

_rag_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    """Return the global :class:`RAGChain` singleton.

    Lazily constructs the chain on first access, wiring it to the shared
    ``VectorStore`` and enabling the reranker.
    """
    global _rag_chain  # noqa: PLW0603
    if _rag_chain is None:
        _rag_chain = RAGChain(
            vector_store=get_vector_store(),
            use_reranker=True,
        )
        logger.info("RAGChain singleton created")
    return _rag_chain


def reset_rag_chain() -> None:
    """Drop the cached chain (useful for tests or graceful shutdown)."""
    global _rag_chain  # noqa: PLW0603
    if _rag_chain is not None:
        logger.info("RAGChain singleton reset")
    _rag_chain = None
