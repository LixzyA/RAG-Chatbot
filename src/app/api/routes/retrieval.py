"""Retrieval endpoints — plain document retrieval without LLM generation.

POST /retrieve
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from app.config import settings
from app.models.requests import RetrieveRequest
from app.models.responses import RetrieveResponse, RetrievedDocument
from app.services.vector_db import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@router.post("/", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Retrieve relevant documents for *query* without LLM generation.

    Supports both hybrid (BM25 + vector) and vector-only search, with
    optional cross-encoder reranking. Sync search/rerank work is delegated
    to a worker thread so the FastAPI event loop stays responsive.
    """
    vs = get_vector_store()

    if req.use_hybrid or req.use_reranker:
        docs = await asyncio.to_thread(
            vs.reranked_search,
            req.query,
            req.top_k,
            settings.hybrid_candidate_multiplier,
            filter=req.filter,
        )
    else:
        scored = await asyncio.to_thread(
            vs.similarity_search,
            req.query,
            req.top_k,
            filter=req.filter,
        )
        docs = [doc for doc, _score in scored]

    results = [
        RetrievedDocument(
            id=doc.metadata.get("_chroma_id", str(i)),
            content=doc.page_content,
            score=doc.metadata.get("rerank_score"),
            metadata=doc.metadata,
        )
        for i, doc in enumerate(docs)
    ]

    return RetrieveResponse(
        query=req.query,
        documents=results,
        total_results=len(results),
    )
