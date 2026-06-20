"""Pydantic response models for all public API endpoints.

Covers both synchronous and streaming (SSE) response shapes.
"""
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


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
    model_used: str  # "specialist" or "generalist"


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
