"""Chat message entity — stores individual messages within a chat session.

Source: backend/entity/chat_message.py
"""
from __future__ import annotations

from sqlalchemy import CheckConstraint, DDL, ForeignKey, Index, Integer, Text, event, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.entity.base import Base


class ChatMessage(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="ck_messages_role",
        ),
        Index("idx_messages_session_id", "session_id"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    created_at: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("(datetime('now'))"),
    )

    chat_session = relationship(
        "ChatSession",
        back_populates="messages",
    )

    def __repr__(self) -> str:
        content_preview = self.content[:30] + ("..." if len(self.content) > 30 else "")
        return f"<ChatMessage {self.id} ({self.role}: {content_preview})>"


event.listen(
    ChatMessage.__table__,
    "after_create",
    DDL(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_updated
        AFTER INSERT ON messages
        BEGIN
            UPDATE chat_sessions SET updated_at = datetime('now')
            WHERE id = NEW.session_id;
        END
        """
    ),
)
