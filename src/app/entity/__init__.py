"""SQLAlchemy ORM entities.

Mirrors backend/entity/ but lives under app/ so the new tree is self-contained.
"""
from app.entity.base import Base
from app.entity.chat_message import ChatMessage
from app.entity.chat_session import ChatSession
from app.entity.rag_traces import RAG_traces
from app.entity.user import User

__all__ = ["Base", "ChatMessage", "ChatSession", "RAG_traces", "User"]
