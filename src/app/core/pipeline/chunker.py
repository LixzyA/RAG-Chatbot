"""Chunk orchestration — converts a large text into chunked ``Document`` objects.

Delegates the mechanical splitting to `core.pipeline.text_splitter`.
Source: backend/file_mgt/service.py (chunking + metadata assignment).
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.core.pipeline.text_splitter import split_text
import logging

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    *,
    filename: str = "<unknown>",
    chunk_size: int = 1024,
    chunk_overlap: float = 0.2,
) -> list[Document]:
    """Split *text* into chunks and wrap each as a LangChain ``Document``.

    Args:
        text: Raw text content.
        filename: Original file name (stored in metadata).
        chunk_size: Target token count per chunk.
        chunk_overlap: Fractional overlap (0.0–1.0).

    Returns:
        List of ``Document`` objects with ``page_content`` and ``metadata``.
    """
    if not text.strip():
        logger.warning("chunk_text called with empty text for '%s'", filename)
        return []

    overlap = int(chunk_size * chunk_overlap)
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=overlap)

    total = len(chunks)
    documents: list[Document] = []
    for i, content in enumerate(chunks):
        doc = Document(
            page_content=content,
            metadata={
                "filename": filename,
                "total_chunk": total,
                "chunk_num": i,
                "chunk_size": chunk_size,
            },
        )
        documents.append(doc)

    logger.info(
        "Chunked '%s' into %d pieces (size=%d, overlap=%d)",
        filename,
        total,
        chunk_size,
        overlap,
    )
    return documents
