import uuid
from fastapi import APIRouter, Request, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from typing import Optional
from . import service
from vectordb.core import VectorDBClient
from . import models
from . import core
from . import history as chat_history
from auth.dependencies import AuthenticatedUser, OptionalAuthenticatedUser
from entity.user import User
from logger import configure_logging

logger = configure_logging(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Streaming chat endpoints
# ---------------------------------------------------------------------------

@router.post("/", response_class=EventSourceResponse)
async def query_chat(
    query_req: models.ChatQueryRequest,
    db_client: VectorDBClient,
    current_user: OptionalAuthenticatedUser = None
):
    llm_client = core.llm_client
    user_id = current_user.id if current_user else None

    logger.info(f"Received chat query: {query_req.prompt}")

    # Ensure chat history exists
    if query_req.chat_id and user_id is not None:
        await chat_history.create_history(query_req.chat_id, user_id=user_id)

    # Save user message
    if query_req.chat_id and user_id is not None:
        await chat_history.add_message(query_req.chat_id, {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": query_req.prompt,
        }, user_id=user_id)

    # Stream the response
    assistant_msg_id = str(uuid.uuid4())
    full_response = ""
    try:
        async for chunk in service.query_chat(
            db_client,
            llm_client,
            prompt=query_req.prompt,
            top_k=query_req.top_k,
        ):
            full_response += chunk
            yield ServerSentEvent(raw_data=chunk)

        yield ServerSentEvent(raw_data="[DONE]", event="done")
        logger.info(f"Successfully finished streaming response for chat_id: {query_req.chat_id}")
    except Exception as e:
        logger.error(f"Failed to generate response for query '{query_req.prompt}': {str(e)}")
        raise e

    # Save assistant message to history
    if query_req.chat_id and user_id is not None and full_response:
        await chat_history.add_message(query_req.chat_id, {
            "id": assistant_msg_id,
            "role": "assistant",
            "content": full_response,
        }, user_id=user_id)


@router.post("/v2", response_class=EventSourceResponse)
async def query_chat_v2(
    query_req: models.ChatQueryRequest,
    db_client: VectorDBClient,
    current_user: OptionalAuthenticatedUser = None
):
    llm_client = core.llm_client
    user_id = current_user.id if current_user else None

    logger.info(f"Received chat_v2 query: {query_req.prompt}")

    # Ensure chat history exists
    if query_req.chat_id and user_id is not None:
        await chat_history.create_history(query_req.chat_id, user_id=user_id)

    # Save user message
    if query_req.chat_id and user_id is not None:
        await chat_history.add_message(query_req.chat_id, {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": query_req.prompt,
        }, user_id=user_id)

    files = service.get_relevant_files(db_client, query_req.prompt, query_req.top_k)
    docs = files.get("documents")
    ids_len = len(files.get('ids', [[]])[0]) if files and files.get('ids') else 0
    logger.info(f"Retrieved {ids_len} relevant files from vector database")
    context_text = "\n\n".join([res for res in docs[0]]) if docs and docs[0] else ""

    # Stream the response
    assistant_msg_id = str(uuid.uuid4())
    full_response = ""
    try:
        async for chunk in service.get_answers(llm_client, query_req.prompt, context_text):
            full_response += chunk
            yield ServerSentEvent(raw_data=chunk)

        yield ServerSentEvent(raw_data="[DONE]", event="done")
        logger.info(f"Successfully finished streaming response for chat_v2 chat_id: {query_req.chat_id}")
    except Exception as e:
        logger.error(f"Failed to generate response in chat_v2 for query '{query_req.prompt}': {str(e)}")
        raise e

    # Save assistant message to history
    if query_req.chat_id and user_id is not None and full_response:
        await chat_history.add_message(query_req.chat_id, {
            "id": assistant_msg_id,
            "role": "assistant",
            "content": full_response,
        }, user_id=user_id)


# ---------------------------------------------------------------------------
# History management endpoints (authenticated)
# ---------------------------------------------------------------------------

@router.get("/histories", response_model=list[models.ChatHistorySummary])
async def list_histories(current_user: AuthenticatedUser):
    """List chat history summaries for the current user."""
    return await chat_history.list_histories(user_id=current_user.id)


@router.get("/history/{chat_id}", response_model=models.ChatHistoryResponse)
async def get_chat_history(chat_id: str, current_user: AuthenticatedUser):
    """Get the full chat history for a specific chat session."""
    history = await chat_history.get_history(chat_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Chat history {chat_id} not found")
    # Ensure the history belongs to the current user (if it has an owner)
    file_uid = history.get("user_id")
    if file_uid and file_uid != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat history")
    return history


@router.delete("/history/{chat_id}")
async def delete_chat_history(chat_id: str, current_user: AuthenticatedUser):
    """Delete a specific chat history."""
    history = await chat_history.get_history(chat_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Chat history {chat_id} not found")
    # Ensure the history belongs to the current user (if it has an owner)
    file_uid = history.get("user_id")
    if file_uid and file_uid != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this chat history")
    deleted = await chat_history.delete_history(chat_id)
    return {"status": "deleted", "chat_id": chat_id}


@router.patch("/history/{chat_id}/title")
async def update_chat_title(chat_id: str, body: dict, current_user: AuthenticatedUser):
    """Update the title of a chat history."""
    history = await chat_history.get_history(chat_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Chat history {chat_id} not found")
    # Ensure the history belongs to the current user (if it has an owner)
    file_uid = history.get("user_id")
    if file_uid and file_uid != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this chat history")
    title = body.get("title")
    if not title:
        raise HTTPException(status_code=400, detail="Field 'title' is required")
    updated = await chat_history.update_title(chat_id, title)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Chat history {chat_id} not found")
    return updated
