"""Pure, framework-agnostic metadata derivation for ingested chunks.

No LLM, no FastAPI, no DB. Safe to test in isolation.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def extract_base_metadata(
    *,
    filename: str,
    uploaded_at: datetime,
) -> dict[str, Any]:
    """Derive the per-document metadata shared by every chunk."""
    base = os.path.basename(filename)
    ext = Path(filename).suffix.lower().lstrip(".") or "unknown"
    return {
        "source": base,
        "file_name": filename,
        "file_type": ext,
        "uploaded_at": uploaded_at.replace(microsecond=0).isoformat(),
    }


def compute_content_hash(text: str) -> str:
    """Stable 16-hex-char SHA-256 fingerprint over chunk content."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def enrich_chunks(
    chunks: list[Document],
    base_meta: dict[str, Any],
) -> list[Document]:
    """Return new Documents with base metadata + per-chunk char_count/content_hash merged in.

    Does not mutate input chunks. Existing metadata wins on key collision so
    chunker-set fields (filename/total_chunk/page) are preserved verbatim.
    """
    out: list[Document] = []
    for chunk in chunks:
        derived = {
            **base_meta,
            "char_count": len(chunk.page_content),
            "content_hash": compute_content_hash(chunk.page_content),
        }
        merged = {**derived, **dict(chunk.metadata or {})}
        out.append(Document(page_content=chunk.page_content, metadata=merged))
    return out
