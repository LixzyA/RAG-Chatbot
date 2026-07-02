"""Document loading — extract raw text from files (PDF, TXT, JSONL, etc.).

Source: backend/file_mgt/service.py (pdfminer / text reading logic).
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from pdfminer.high_level import extract_text as pdf_extract_text  # noqa: N813

from app.utils.exceptions import PDFProcessingException, FileTypeNotSupportedException
import logging

logger = logging.getLogger(__name__)

_ALLOWED_SUFFIXES = frozenset({".txt", ".md", ".pdf", ".jsonl"})


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


def load_pdf_pages(filename: str, content: bytes) -> list[tuple[int, str]]:
    """Iterate ``pdfminer`` extraction per page so each carries its page number.

    Returns ``[(page_number_1_based, page_text), ...]``. Empty pages are skipped;
    pure-image or empty PDFs still anchor ``page=1`` so downstream has the field.
    """
    if Path(filename).suffix.lower() != ".pdf":
        raise FileTypeNotSupportedException(
            f"load_pdf_pages expects .pdf, got {filename}"
        )
    try:
        # form-feed heuristic: pdfminer uses \x0c between pages in flat extraction.
        full = pdf_extract_text(BytesIO(content))
        parts = full.split("\x0c")
        pages: list[tuple[int, str]] = []
        for idx, text in enumerate(parts, start=1):
            if text.strip():
                pages.append((idx, text))
        if not pages:
            pages = [(1, full or "")]
        return pages
    except FileTypeNotSupportedException:
        raise
    except Exception as exc:
        logger.error("PDF per-page extraction failed for %s: %s", filename, exc)
        raise PDFProcessingException(str(exc)) from exc


def load_jsonl_bytes(
    filename: str,
    content: bytes,
) -> list[tuple[str, dict[str, Any]]]:
    """Parse a ``.jsonl`` byte stream into ``(text, metadata)`` records.

    Each line must be a JSON object with a ``"text"`` field. Every other key
    is treated as per-record metadata and merged into chunk metadata downstream.

    Args:
        filename: Original file name (for error messages).
        content: Raw bytes of the JSONL file.

    Returns:
        List of ``(text, metadata)`` tuples — one per valid JSON line.

    Raises:
        FileTypeNotSupportedException: If *filename* suffix is not ``.jsonl``.
        ValueError: If a line is not valid JSON or misses a ``"text"`` field.
    """
    suffix = Path(filename).suffix.lower()
    if suffix != ".jsonl":
        raise FileTypeNotSupportedException(f"Expected .jsonl, got {suffix}")

    records: list[tuple[str, dict[str, Any]]] = []
    decoded = content.decode("utf-8")
    for line_no, line in enumerate(decoded.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.warning("%s line %d: invalid JSON — %s", filename, line_no, exc)
            continue

        if not isinstance(record, dict):
            logger.warning(
                "%s line %d: expected dict, got %s — skipping",
                filename,
                line_no,
                type(record).__name__,
            )
            continue

        text = record.get("text", "") or record.get("page_content", "")
        if not text:
            logger.warning("%s line %d: no 'text' field — skipping", filename, line_no)
            continue

        meta = {k: v for k, v in record.items() if k not in ("text", "page_content")}
        records.append((text, meta))

    logger.info("Parsed %d records from %s", len(records), filename)
    return records
