"""Document loading — extract raw text from files (PDF, TXT, etc.).

Source: backend/file_mgt/service.py (pdfminer / text reading logic).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pdfminer.high_level import extract_text as pdf_extract_text  # noqa: N813

from app.utils.exceptions import PDFProcessingException, FileTypeNotSupportedException
import logging

logger = logging.getLogger(__name__)

_ALLOWED_SUFFIXES = frozenset({".txt", ".md", ".pdf"})


def load_document(file_path: str | Path) -> str:
    """Extract plain text from *file_path*.

    Currently supports ``.txt``, ``.md``, and ``.pdf``.

    Args:
        file_path: Path to the file.

    Returns:
        UTF-8 decoded text content.

    Raises:
        FileTypeNotSupportedException: If the suffix is not supported.
        PDFProcessingException: If pdfminer fails on a PDF file.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in _ALLOWED_SUFFIXES:
        raise FileTypeNotSupportedException(f"File type {suffix} not supported")

    if suffix == ".pdf":
        try:
            return pdf_extract_text(path)
        except Exception as exc:
            logger.error("PDF extraction failed for %s: %s", path, exc)
            raise PDFProcessingException(str(exc)) from exc

    return path.read_text(encoding="utf-8")


def load_document_bytes(filename: str, content: bytes) -> str:
    """Same as :func:`load_document` but accepts raw bytes.

    Useful when the file has already been read into memory (e.g. from an
    HTTP upload).
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise FileTypeNotSupportedException(f"File type {suffix} not supported")

    if suffix == ".pdf":
        try:
            return pdf_extract_text(BytesIO(content))
        except Exception as exc:
            logger.error("PDF extraction failed for %s: %s", filename, exc)
            raise PDFProcessingException(str(exc)) from exc

    return content.decode("utf-8")
