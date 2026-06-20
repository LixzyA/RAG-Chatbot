from __future__ import annotations

import logging

from fastapi import APIRouter

from app.services.vector_db import get_reranker, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Aggregate application health check."""
    db_healthy = True
    db_error = None
    try:
        vs = get_vector_store()
        db_healthy = vs.heartbeat()
    except Exception as exc:
        db_healthy = False
        db_error = str(exc)

    reranker_result = get_reranker().healthcheck()

    return {
        "status": "ok" if db_healthy else "degraded",
        "vector_db": {"healthy": db_healthy, "error": db_error},
        "reranker": reranker_result,
    }
