from __future__ import annotations

import logging

from fastapi import APIRouter

from app.core.generation import llm_client
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
        vs_result = vs.heartbeat()
        chroma = vs_result["chroma_client"]
        vstore = vs_result["vector_store"]

        db_healthy = chroma["healthy"] and vstore["healthy"]
        errors = [c["error"] for c in [chroma, vstore] if c["error"]]
        if errors:
            db_error = "; ".join(errors)
    except Exception as exc:
        db_healthy = False
        db_error = str(exc)

    reranker_result = get_reranker().healthcheck()

    llm_result = await llm_client.healthcheck()

    all_healthy = db_healthy and llm_result["healthy"]

    return {
        "status": "ok" if all_healthy else "degraded",
        "vector_db": {"healthy": db_healthy, "error": db_error},
        "reranker": reranker_result,
        "llm_client": llm_result,
    }
