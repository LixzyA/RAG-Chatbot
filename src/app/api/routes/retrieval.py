"""Retrieval endpoints — plain document retrieval without LLM generation.

POST /retrieve
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from app.models.requests import RetrieveRequest
from app.models.responses import RetrieveResponse, RetrievedDocument
from app.services.vector_db import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@router.post("/", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Retrieve relevant documents for *query* without LLM generation.

    Supports both hybrid (BM25 + vector) and vector-only search, with
    optional cross-encoder reranking.
    """
    vs = get_vector_store()

    if req.use_hybrid or req.use_reranker:
        docs = vs.reranked_search(
            query=req.query,
            k=req.top_k,
            candidate_multiplier=4 if not req.use_hybrid else 4,
        )
    else:
        scored = vs.similarity_search(req.query, k=req.top_k)
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
