"""Pydantic models and schemas shared across the application."""

from app.models.documents import Document
from app.models.feedback import EvaluationResult
from app.models.rag_trace import RAGTraceBuilder
from app.models.requests import (
    ChatQueryRequest,
    IngestBatchRequest,
    IngestRequest,
    LoginRequest,
    RegisterRequest,
    RetrieveBatchRequest,
    RetrieveRequest,
)
from app.models.responses import (
    # Auth
    UserResponse,
    TokenResponse,
    # Chat
    ChatMessage,
    ChatSessionResponse,
    ChatSessionSummary,
    ChatRouteInfo,
    # Ingestion
    UploadFileResponse,
    # Retrieval
    RetrievedDocument,
    RetrieveResponse,
    # Health
    HealthResponse,
)

__all__ = [
    # documents
    "Document",
    # feedback
    "EvaluationResult",
    # rag trace
    "RAGTraceBuilder",
    # requests
    "ChatQueryRequest",
    "IngestBatchRequest",
    "IngestRequest",
    "LoginRequest",
    "RegisterRequest",
    "RetrieveBatchRequest",
    "RetrieveRequest",
    # responses
    "ChatMessage",
    "ChatRouteInfo",
    "ChatSessionResponse",
    "ChatSessionSummary",
    "HealthResponse",
    "RetrievedDocument",
    "RetrieveResponse",
    "TokenResponse",
    "UploadFileResponse",
    "UserResponse",
]
