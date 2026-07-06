"""Pydantic response models for all public API endpoints.

Covers both synchronous and streaming (SSE) response shapes.
"""

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.orchestration.rag_chain import RAGChain
from app.models.rag_trace import RAGTraceBuilder


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


class ChatMessage(BaseModel):
    id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str | None = None


class ChatSessionResponse(BaseModel):
    chat_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[ChatMessage]


class ChatSessionSummary(BaseModel):
    chat_id: str
    user_id: int | None = None
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ChatRouteInfo(BaseModel):
    """Internal routing metadata for logging / debugging."""

    topic: str
    confidence: float
    model_used: str  # "generation"


class _SSEStreamContext(BaseModel):
    """Bundle all state captured by the SSE event-stream generator.

    Avoids a nested closure inside the route so the generator can live
    at module level and remain testable outside the FastAPI context.
    """

    chain: RAGChain
    prompt: str
    top_k: int
    metadata_filter: dict[str, Any] | None
    builder: RAGTraceBuilder
    previous_query: str | None
    self_feedback_enabled: bool
    db: AsyncSession
    internal_session_id: int | None
    user_id: int | None
    assistant_msg_id: str
    chat_id: str | None


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


class UploadFileResponse(BaseModel):
    status: int = Field(description="HTTP-like status code: 200 = success, etc.")
    num_chunk: int = Field(description="Number of chunks produced from the document")


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


class RetrievedDocument(BaseModel):
    id: str
    content: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    query: str
    documents: list[RetrievedDocument]
    total_results: int
    transformed_queries: list[str] | None = None


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
