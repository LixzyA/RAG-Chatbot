"""Chat session history — SQLite-backed CRUD using SQLAlchemy.

Each public function takes the ``AsyncSession`` as its first argument so
callers (routes) can inject the session via ``Depends(get_db)``. This keeps
DB access consistent across the app and lets tests swap in a fake session.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.entity.chat_message import ChatMessage
from app.entity.chat_session import ChatSession

logger = logging.getLogger(__name__)

MAX_HISTORIES = 100
DEFAULT_TITLE = "Untitled Chat"


def _now_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _history_summary(hist: ChatSession) -> dict:
    return {
        "chat_id": hist.session_uuid,
        "user_id": hist.user_id,
        "title": hist.title or DEFAULT_TITLE,
        "created_at": _as_text(hist.created_at),
        "updated_at": _as_text(hist.updated_at),
        "message_count": len(hist.messages),
    }


def _message_payload(msg: ChatMessage) -> dict:
    return {
        "id": str(msg.id),
        "role": msg.role,
        "content": msg.content,
        "timestamp": _as_text(msg.created_at),
    }


def _history_payload(hist: ChatSession) -> dict:
    return {
        "chat_id": hist.session_uuid,
        "user_id": hist.user_id,
        "title": hist.title or DEFAULT_TITLE,
        "created_at": _as_text(hist.created_at),
        "updated_at": _as_text(hist.updated_at),
        "messages": [_message_payload(m) for m in hist.messages],
    }


async def list_histories(db: AsyncSession, user_id: int | None = None) -> list[dict]:
    query = (
        select(ChatSession)
        .where(ChatSession.deleted_at.is_(None))
        .order_by(desc(ChatSession.updated_at))
        .limit(MAX_HISTORIES)
    )
    if user_id is not None:
        query = query.where(ChatSession.user_id == user_id)

    result = await db.execute(query)
    return [_history_summary(hist) for hist in result.scalars().all()]


async def get_history(db: AsyncSession, chat_id: str) -> dict | None:
    query = select(ChatSession).where(
        ChatSession.session_uuid == chat_id,
        ChatSession.deleted_at.is_(None),
    )
    result = await db.execute(query)
    hist = result.scalar_one_or_none()
    return _history_payload(hist) if hist else None


async def get_internal_session_id(db: AsyncSession, chat_id: str) -> int | None:
    """Return the integer PK for a chat session, looked up by its ``session_uuid``.

    The route layer receives a string UUID from the client, but downstream
    tables (e.g. ``rag_traces.session_id``) FK to ``chat_sessions.id`` (int).
    Returns ``None`` if the session is missing or soft-deleted.
    """
    result = await db.execute(
        select(ChatSession.id).where(
            ChatSession.session_uuid == chat_id,
            ChatSession.deleted_at.is_(None),
        )
    )
    return result.scalar_one_or_none()


async def create_or_get_history(
    db: AsyncSession, chat_id: str, *, user_id: int | None = None
) -> dict | None:
    if user_id is None:
        return None

    query = select(ChatSession).where(ChatSession.session_uuid == chat_id)
    result = await db.execute(query)
    existing = result.scalar_one_or_none()

    if existing:
        if existing.deleted_at is not None:
            existing.deleted_at = None
            await db.commit()
            await db.refresh(existing)
        return _history_payload(existing)

    new_history = ChatSession(
        session_uuid=chat_id,
        user_id=user_id,
        title=DEFAULT_TITLE,
    )
    db.add(new_history)
    await db.commit()
    await db.refresh(new_history)
    return _history_payload(new_history)


async def add_message(
    db: AsyncSession,
    chat_id: str,
    message: dict,
    *,
    user_id: int | None = None,
) -> dict | None:
    if user_id is None:
        return None

    query = select(ChatSession).where(ChatSession.session_uuid == chat_id)
    result = await db.execute(query)
    history = result.scalar_one_or_none()

    if history is None:
        history = ChatSession(
            session_uuid=chat_id,
            user_id=user_id,
            title=DEFAULT_TITLE,
        )
        db.add(history)
        await db.flush()
    elif history.user_id != user_id:
        return None

    role = message.get("role", "user")
    new_message = ChatMessage(
        session_id=history.id,
        role=role,
        content=message.get("content", ""),
        token_count=message.get("token_count"),
    )
    db.add(new_message)

    # Auto-title from first user message
    if (history.title or DEFAULT_TITLE) == DEFAULT_TITLE and role == "user":
        content = message.get("content", "")
        history.title = content[:80] + ("..." if len(content) > 80 else "")

    await db.commit()

    query = select(ChatSession).where(ChatSession.session_uuid == chat_id)
    result = await db.execute(query)
    updated = result.scalar_one()
    return _history_payload(updated)


async def delete_history(db: AsyncSession, chat_id: str) -> bool:
    """Soft-delete a chat session. Returns ``True`` if deleted."""
    query = select(ChatSession).where(
        ChatSession.session_uuid == chat_id,
        ChatSession.deleted_at.is_(None),
    )
    result = await db.execute(query)
    history = result.scalar_one_or_none()
    if history is None:
        return False
    history.deleted_at = _now_text()
    await db.commit()
    return True


async def update_title(db: AsyncSession, chat_id: str, title: str) -> dict | None:
    query = select(ChatSession).where(
        ChatSession.session_uuid == chat_id,
        ChatSession.deleted_at.is_(None),
    )
    result = await db.execute(query)
    history = result.scalar_one_or_none()
    if history is None:
        return None
    history.title = title
    history.updated_at = _now_text()
    await db.commit()
    await db.refresh(history)
    return _history_payload(history)
