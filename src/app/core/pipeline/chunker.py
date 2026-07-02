from langchain_core.documents import Document

from app.core.pipeline.text_splitter import split_text
import logging

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    *,
    filename: str,
    chunk_size: int = 1024,
    chunk_overlap: float = 0.2,
    page_number: int | None = None,
) -> list[Document]:
    """Split *text* into chunks and wrap each as a LangChain ``Document``.

    Args:
        text: Raw text content.
        filename: Original file name (stored in metadata).
        chunk_size: Target token count per chunk.
        chunk_overlap: Fractional overlap (0.0–1.0).
        page_number: Optional 1-based PDF page number — stamped into metadata
            as ``"page"`` for page-aware retrieval filters.

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
        metadata: dict = {
            "file_name": filename,
            "total_chunk": total,
            "chunk_num": i,
            "chunk_size": chunk_size,
        }
        if page_number is not None:
            metadata["page"] = page_number
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    logger.info(
        "Chunked '%s' into %d pieces (size=%d, overlap=%d)",
        filename,
        total,
        chunk_size,
        overlap,
    )
    return documents
