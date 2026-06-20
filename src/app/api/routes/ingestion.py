"""Ingestion endpoints — upload documents and add them to the vector store.

POST /ingest
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, UploadFile

from app.core.pipeline.chunker import chunk_text
from app.core.pipeline.document_loader import load_document_bytes
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

    Supports ``.txt``, ``.md``, and ``.pdf``.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    raw_bytes = await file.read()

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

    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    vs = get_vector_store()
    vs.add_documents(chunks, ids=ids)

    logger.info("Ingested %d chunks from '%s'", len(chunks), file.filename)
    return UploadFileResponse(status=200, num_chunk=len(chunks))
