import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.sse import format_sse_event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    AuthenticatedUser,
    OptionalAuthenticatedUser,
    get_db,
    get_rag_chain_dep,
)
from app.config import settings
from app.core.orchestration.rag_chain import RAGChain
from app.entity.feedback_log import FeedbackLog
from app.entity.rag_traces import RAG_traces
from app.models.rag_trace import RAGTraceBuilder
from app.models.requests import ChatQueryRequest
from app.models.responses import _SSEStreamContext
from app.services import chat_history_service
from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


async def _safe_rollback(db: AsyncSession) -> None:
    """Rollback *db*, logging failure without raising."""
    try:
        await db.rollback()
    except Exception:
        logger.exception("Database rollback failed")


async def _sse_event_stream(ctx: _SSEStreamContext) -> AsyncIterator[bytes]:
    """SSE token stream: yield chunks, persist trace + assistant message on completion."""
    try:
        async for chunk in ctx.chain.run(
            ctx.prompt,
            top_k=ctx.top_k,
            builder=ctx.builder,
            previous_query=ctx.previous_query,
            self_feedback_enabled=ctx.self_feedback_enabled,
            filter=ctx.metadata_filter,
        ):
            yield format_sse_event(data_str=chunk)
    except AppException as exc:
        logger.warning(
            "SSE: RAG chain interrupted by %s: %s",
            exc.__class__.__name__,
            exc.message,
        )
        yield format_sse_event(data_str=f"[ERROR] {exc.message}")
    except Exception as exc:
        # SSE boundary safety-net: headers already flushed; emit error event.
        logger.exception("SSE: unexpected failure in chain.run")
        yield format_sse_event(data_str=f"[ERROR] Unexpected error: {exc}")
    finally:
        yield format_sse_event(data_str="[DONE]")

        if ctx.builder.original_query:
            await record_rag_trace(
                ctx.db,
                ctx.builder,
                session_id=ctx.internal_session_id,
                user_id=ctx.user_id,
            )

        # Save assistant message
        if ctx.chat_id and ctx.user_id is not None:
            try:
                await chat_history_service.add_message(
                    ctx.db,
                    ctx.chat_id,
                    {
                        "id": ctx.assistant_msg_id,
                        "role": "assistant",
                        "content": ctx.builder.llm_response or "",
                    },
                    user_id=ctx.user_id,
                )
            except SQLAlchemyError:
                logger.exception("Failed to save assistant message (non-fatal)")
                await _safe_rollback(ctx.db)


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
            filters_applied=builder.filters_applied,
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
        await db.refresh(trace)  # populate trace.id for FK

        # Persist per-iteration feedback logs.
        if builder.feedback_scores:
            for score_entry in builder.feedback_scores:
                db.add(
                    FeedbackLog(
                        rag_trace_id=trace.id,
                        session_id=session_id,
                        iteration=score_entry["iteration"],
                        query_used=builder.original_query,
                        draft_answer=builder.llm_response,
                        faithfulness=score_entry.get("faithfulness"),
                        relevance=score_entry.get("relevance"),
                        completeness=score_entry.get("completeness"),
                        composite_score=min(
                            score_entry.get("faithfulness", 1.0),
                            score_entry.get("relevance", 1.0),
                            score_entry.get("completeness", 1.0),
                        ),
                        critique=score_entry.get("critique"),
                        refinement_strategy=score_entry.get("strategy"),
                        passed=score_entry.get("strategy"),
                    )
                )
            await db.commit()

        logger.debug(
            "Recorded rag_traces row for query: %s", builder.original_query[:80]
        )
    except SQLAlchemyError as exc:
        logger.exception("Failed to persist rag_traces row (non-fatal): %s", exc)
        await _safe_rollback(db)


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
        await chat_history_service.create_or_get_history(
            db, query_req.chat_id, user_id=user_id
        )
        internal_session_id = await chat_history_service.get_internal_session_id(
            db, query_req.chat_id
        )

    # Fetch the previous user turn BEFORE saving the current message, so that
    # "last user message" really means the prior turn. The chain uses this as
    # conversational grounding only — it never influences retrieval.
    previous_query: str | None = None
    if query_req.chat_id and user_id is not None:
        previous_query = await chat_history_service.get_last_user_message(
            db, query_req.chat_id
        )

    # Save user message
    if query_req.chat_id and user_id is not None:
        try:
            await chat_history_service.add_message(
                db,
                query_req.chat_id,
                {"id": str(uuid.uuid4()), "role": "user", "content": query_req.prompt},
                user_id=user_id,
            )
        except SQLAlchemyError:
            logger.exception("Failed to save user message (non-fatal)")
            await _safe_rollback(db)

    assistant_msg_id = str(uuid.uuid4())
    builder = RAGTraceBuilder()
    ctx = _SSEStreamContext(
        chain=chain,
        prompt=query_req.prompt,
        top_k=query_req.top_k,
        metadata_filter=query_req.filter,
        builder=builder,
        previous_query=previous_query,
        self_feedback_enabled=settings.self_feedback_enabled or query_req.self_check,
        db=db,
        internal_session_id=internal_session_id,
        user_id=user_id,
        assistant_msg_id=assistant_msg_id,
        chat_id=query_req.chat_id,
    )
    return StreamingResponse(
        _sse_event_stream(ctx),
        media_type="text/event-stream",
    )


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
