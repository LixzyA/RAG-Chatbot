"""Chat session entity — stores chat sessions with metadata.

Renamed from backend/entity/chat_history.py → ChatSession / chat_sessions.
"""
from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Integer, Text, desc, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base
from .chat_message import ChatMessage

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (
        Index("idx_chat_sessions_user_id", "user_id"),
        Index("idx_chat_sessions_updated_at", desc("updated_at")),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    session_uuid: Mapped[str] = mapped_column(
        Text,
        unique=True,
        nullable=False,
        server_default=text("(lower(hex(randomblob(16))))"),
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("(datetime('now'))"),
    )
    updated_at: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("(datetime('now'))"),
    )
    deleted_at: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    user = relationship("User", back_populates="sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="chat_session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        title = self.title or "Untitled Chat"
        return f"<ChatSession {self.session_uuid[:8]} ({title[:30]})>"
