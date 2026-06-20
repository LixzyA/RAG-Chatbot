
from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    AuthenticatedUser,
    OptionalAuthenticatedUser,
    get_db,
    get_rag_chain_dep,
)
from app.core.orchestration.rag_chain import RAGChain, RAGTraceBuilder
from app.entity.rag_traces import RAG_traces
from app.models.requests import ChatQueryRequest
from app.services import chat_history_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

async def record_rag_trace(
    db: AsyncSession,
    builder: RAGTraceBuilder,
    *,
    session_id: int | None,
    user_id: int | None,
) -> None:
    """Persist a ``RAG_traces`` row from the populated ``builder``.

    Silently no-ops when *both* FKs are ``None`` (no session, no user) — this
    is the schema's documented "anonymous query" case but the row would carry
    no causal attribution, so we skip it. Failures are logged and swallowed
    so a trace-write bug never breaks a chat response.
    """
    try:
        trace = RAG_traces(
            original_query=builder.original_query,
            transformation_technique=builder.transformation_technique,
            transformed_query=builder.transformed_query,
            retrieved_chunks=builder.retrieved_chunks or None,
            reranked_chunks=builder.reranked_chunks or None,
            context_passed_to_llm=builder.context_passed_to_llm or None,
            llm_response=builder.llm_response or "",
            session_id=session_id,
            user_id=user_id,
            llm_model_name=builder.llm_model_name,
            embedding_model_name=builder.embedding_model_name,
            input_tokens=builder.input_tokens,
            output_tokens=builder.output_tokens,
            retrieval_latency_ms=builder.retrieval_latency_ms,
            rerank_latency_ms=builder.rerank_latency_ms,
            llm_latency_ms=builder.llm_latency_ms,
        )
        db.add(trace)
        await db.commit()
        logger.debug("Recorded rag_traces row for query: %s", builder.original_query[:80])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to persist rag_traces row (non-fatal): %s", exc)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


@router.post("/")
async def chat(
    query_req: ChatQueryRequest,
    current_user: OptionalAuthenticatedUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    chain: Annotated[RAGChain, Depends(get_rag_chain_dep)],
):
    """Server-Sent Events (SSE) RAG chat endpoint.

    Streams LLM tokens as raw text events, terminated by a ``[DONE]`` event.
    A ``RAG_traces`` row is persisted for every successful request that we
    can attribute to a session or a user.
    """
    user_id = current_user.id if current_user else None

    # Resolve the int FK for chat_sessions (string UUID -> int PK).
    internal_session_id: int | None = None
    if query_req.chat_id and user_id is not None:
        await chat_history_service.create_or_get_history(db, query_req.chat_id, user_id=user_id)
        internal_session_id = await chat_history_service.get_internal_session_id(
            db, query_req.chat_id
        )

    # Save user message
    if query_req.chat_id and user_id is not None:
        await chat_history_service.add_message(
            db,
            query_req.chat_id,
            {"id": str(uuid.uuid4()), "role": "user", "content": query_req.prompt},
            user_id=user_id,
        )

    assistant_msg_id = str(uuid.uuid4())
    builder = RAGTraceBuilder()

    async def event_stream():
        async for chunk in chain.run(
            query_req.prompt, top_k=query_req.top_k, builder=builder
        ):
            yield chunk

        yield "[DONE]"

        if builder.original_query:
            await record_rag_trace(
                db,
                builder,
                session_id=internal_session_id,
                user_id=user_id,
            )

        # Save assistant message
        if query_req.chat_id and user_id is not None and builder.llm_response:
            await chat_history_service.add_message(
                db,
                query_req.chat_id,
                {"id": assistant_msg_id, "role": "assistant", "content": builder.llm_response},
                user_id=user_id,
            )

    return EventSourceResponse(event_stream())


# ------------------------------------------------------------------
# History management (authenticated)
# ------------------------------------------------------------------

@router.get("/histories")
async def list_histories(
    current_user: AuthenticatedUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """List chat session summaries for the current user."""
    return await chat_history_service.list_histories(db, user_id=current_user.id)


@router.get("/history/{chat_id}")
async def get_chat_history(
    chat_id: str,
    current_user: AuthenticatedUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get the full chat history for a specific session."""
    history = await chat_history_service.get_history(db, chat_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Chat session {chat_id} not found")
    if history.get("user_id") and history["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return history


@router.delete("/history/{chat_id}")
async def delete_chat_history(
    chat_id: str,
    current_user: AuthenticatedUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Soft-delete a chat session."""
    history = await chat_history_service.get_history(db, chat_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Chat session {chat_id} not found")
    if history.get("user_id") and history["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    await chat_history_service.delete_history(db, chat_id)
    return {"status": "deleted", "chat_id": chat_id}


@router.patch("/history/{chat_id}/title")
async def update_chat_title(
    chat_id: str,
    body: dict,
    current_user: AuthenticatedUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update the title of a chat session."""
    history = await chat_history_service.get_history(db, chat_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Chat session {chat_id} not found")
    if history.get("user_id") and history["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    title = body.get("title")
    if not title:
        raise HTTPException(status_code=400, detail="Field 'title' is required")
    updated = await chat_history_service.update_title(db, chat_id, title)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Chat session {chat_id} not found")
    return updated
