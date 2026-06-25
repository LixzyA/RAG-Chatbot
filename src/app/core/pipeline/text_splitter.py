"""Text splitting utilities — chunk raw text into smaller pieces.

Source: backend/file_mgt/service.py (RecursiveCharacterTextSplitter logic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging

if TYPE_CHECKING:
    from langchain_core.documents import Document as LCDocument

logger = logging.getLogger(__name__)


def split_text(
    text: str,
    *,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    model_name: str = "gpt-4",
) -> list[str]:
    """Split *text* into overlapping chunks using tiktoken encoding.

    Args:
        text: The raw text to split.
        chunk_size: Maximum token count per chunk.
        chunk_overlap: Number of tokens to overlap between consecutive chunks.
        model_name: Tiktoken encoder to use (default ``gpt-4``).

    Returns:
        List of chunk strings.
    """
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name,
    )
    docs: list[LCDocument] = splitter.create_documents([text])
    return [doc.page_content for doc in docs]
