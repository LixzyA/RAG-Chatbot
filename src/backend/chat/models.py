from pydantic import BaseModel
from typing import Optional


class ChatQueryRequest(BaseModel):
    prompt: str
    top_k: int = 10
    chat_id: Optional[str] = None


class ChatMessage(BaseModel):
    id: str
    role: str  # "user" | "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    chat_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[ChatMessage]


class ChatHistorySummary(BaseModel):
    chat_id: str
    user_id: Optional[int] = None
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ChatRouteInfo(BaseModel):
    """Internal routing metadata for logging/debugging."""
    topic: str
    confidence: float
    model_used: str  # "specialist" or "generalist"
