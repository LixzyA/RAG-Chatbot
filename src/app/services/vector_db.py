"""Vector DB service — factory / adapter for the concrete VectorStore.

Manages singleton lifecycle for the ``VectorStore`` + ``CrossEncoderReranker``
combo so routes and orchestration can both access the same instance via
FastAPI dependency injection or direct import.
"""

import logging
from app.core.retrieval.reranker import CrossEncoderReranker
from app.core.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_vector_store: VectorStore | None = None
_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """Return the global :class:`CrossEncoderReranker` singleton."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
        logger.info("Reranker singleton created")
    return _reranker


def get_vector_store(
    *,
    collection_name: str | None = None,
    persist_path: str | None = None,
) -> VectorStore:
    """Return the global :class:`VectorStore` singleton.

    If you need a different collection or path, pass them on the first call.
    Subsequent calls ignore these arguments and return the cached instance.
    """
    global _vector_store
    if _vector_store is None:
        kwargs: dict = {}
        if collection_name:
            kwargs["collection_name"] = collection_name
        if persist_path:
            kwargs["persist_path"] = persist_path
        kwargs["reranker"] = get_reranker()
        _vector_store = VectorStore(**kwargs)
        logger.info(
            "VectorStore singleton created (collection=%s path=%s)",
            _vector_store.collection_name,
            persist_path or "(default)",
        )
    return _vector_store


def reset_store() -> None:
    """Teardown — useful for tests or graceful shutdown."""
    global _vector_store, _reranker
    if _vector_store:
        _vector_store.close()
    if _reranker:
        _reranker.close()
    _vector_store = None
    _reranker = None
    logger.info("VectorStore and Reranker singletons reset")
