from pydantic import BaseModel


class ChatQueryRequest(BaseModel):
    prompt: str
    top_k: int | None = 10


class ChatRouteInfo(BaseModel):
    """Internal routing metadata for logging/debugging."""
    topic: str
    confidence: float
    model_used: str  # "specialist" or "generalist"