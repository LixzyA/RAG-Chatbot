"""Ingestion endpoints — upload documents and add them to the vector store.

POST /ingest
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from langchain_core.documents import Document

from app.config import settings
from app.core.orchestration.metadata_enrichment import (
    enrich_chunks,
    extract_base_metadata,
)
from app.core.orchestration.ner_service import enrich_chunks_route
from app.core.pipeline.chunker import chunk_text
from app.core.pipeline.document_loader import load_document_bytes, load_pdf_pages
from app.core.pipeline.language_detect import detect_chunk_languages
from app.models.responses import UploadFileResponse
from app.services.vector_db import get_vector_store
from app.utils.exceptions import FileTypeNotSupportedException, PDFProcessingException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/", response_model=UploadFileResponse)
async def ingest_file(
    file: UploadFile,
    chunk_size: int = 1024,
    chunk_overlap: float = 0.2,
):
    """Upload a single document and ingest its chunks into the vector store.

    Supports ``.txt``, ``.md``, and ``.pdf`` (PDFs chunked per-page so each
    chunk keeps its 1-based page number in ``metadata["page"]``).

    All chunks receive heuristic metadata (``source``, ``file_name``,
    ``file_type``, ``uploaded_at``, ``char_count``, ``content_hash``).
    Language detection via ``langdetect`` runs unconditionally — every
    non-blank chunk gets a ``language`` field (``"en"``, ``"id"``, or absent
    for unknown/unrecognised languages). When ``NER_ENABLED=true`` (default),
    EN/ID chunks additionally get ``entities``, ``entity_types``, and
    ``ner_model`` written by the NER pipeline.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    raw_bytes = await file.read()
    uploaded_at = datetime.now(timezone.utc)
    suffix = Path(file.filename).suffix.lower()
    chunks: list[Document] = []

    if suffix == ".pdf":
        try:
            pages = load_pdf_pages(file.filename, raw_bytes)
        except FileTypeNotSupportedException as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except PDFProcessingException as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        for page_num, page_text in pages:
            chunks.extend(
                chunk_text(
                    page_text,
                    filename=file.filename,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    page_number=page_num,
                )
            )
    else:
        try:
            text = load_document_bytes(file.filename, raw_bytes)
        except FileTypeNotSupportedException as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except PDFProcessingException as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        chunks = chunk_text(
            text,
            filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced zero chunks")

    # --- Heuristic enrichment (always on) ---
    base_meta = extract_base_metadata(filename=file.filename, uploaded_at=uploaded_at)
    chunks = enrich_chunks(chunks, base_meta)

    # --- Language detection (always on — sets ``language`` on every non-blank chunk) ---
    detect_chunk_languages(chunks)

    # --- NER enrichment (gated by env — adds entities/entity_types/ner_model) ---
    if settings.ner_enabled:
        enrich_chunks_route(chunks)  # mutates metadata in place

    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    vs = get_vector_store()
    vs.add_documents(chunks, ids=ids)

    logger.info(
        "Ingested %d chunks from '%s' (ner_enabled=%s)",
        len(chunks),
        file.filename,
        settings.ner_enabled,
    )
    return UploadFileResponse(status=200, num_chunk=len(chunks))
