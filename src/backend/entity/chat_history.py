"""
Chat session entity - stores chat sessions with metadata.
"""

from sqlalchemy import ForeignKey, Index, Integer, Text, desc, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .chat_message import ChatMessage

from .base import Base


class ChatHistory(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        Index("idx_sessions_user_id", "user_id"),
        Index("idx_sessions_updated_at", desc("updated_at")),
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
        back_populates="chat_history",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        title = self.title or "Untitled Chat"
        return f"<ChatHistory {self.session_uuid[:8]} ({title[:30]})>"
