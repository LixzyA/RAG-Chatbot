"""Embedding step — convert text chunks into vector embeddings.

Source: backend/vectordb/custom_embeddings.py (sentence-transformers adapter).
"""
from __future__ import annotations

import asyncio

from sentence_transformers import SentenceTransformer

import logging

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """LangChain-compatible embeddings backed by ``sentence_transformers``."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or _DEFAULT_MODEL
        self._model: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text."""
        self._load()
        assert self._model is not None
        embeddings = self._model.encode(texts, convert_to_numpy=True).tolist()
        return [list(e) for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Return a single embedding vector for *text*."""
        self._load()
        assert self._model is not None
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async wrapper around :meth:`embed_documents`."""
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Async wrapper around :meth:`embed_query`."""
        return await asyncio.to_thread(self.embed_query, text)
