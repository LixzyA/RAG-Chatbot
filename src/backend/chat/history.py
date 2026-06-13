"""
Chat history storage - SQLite database persistence.

Uses SQLAlchemy ORM to store and retrieve chat sessions with messages.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import desc, select

from entity import ChatHistory, ChatMessage
from entity.base import async_session_factory

MAX_HISTORIES = 100
DEFAULT_TITLE = "Untitled Chat"


def _now_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_text(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _history_summary(hist: ChatHistory) -> dict:
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


def _history_payload(hist: ChatHistory) -> dict:
    return {
        "chat_id": hist.session_uuid,
        "user_id": hist.user_id,
        "title": hist.title or DEFAULT_TITLE,
        "created_at": _as_text(hist.created_at),
        "updated_at": _as_text(hist.updated_at),
        "messages": [_message_payload(msg) for msg in hist.messages],
    }


async def list_histories(user_id: Optional[int] = None) -> list[dict]:
    """
    Return a list of chat history summaries for a user.
    Sorted by most recent first.
    """
    async with async_session_factory() as session:
        query = (
            select(ChatHistory)
            .where(ChatHistory.deleted_at.is_(None))
            .order_by(desc(ChatHistory.updated_at))
            .limit(MAX_HISTORIES)
        )

        if user_id is not None:
            query = query.where(ChatHistory.user_id == user_id)

        result = await session.execute(query)
        return [_history_summary(hist) for hist in result.scalars().all()]


async def get_history(chat_id: str) -> Optional[dict]:
    """
    Retrieve the full chat history for a given external session UUID.
    Returns None if not found.
    """
    async with async_session_factory() as session:
        query = select(ChatHistory).where(
            ChatHistory.session_uuid == chat_id,
            ChatHistory.deleted_at.is_(None),
        )
        result = await session.execute(query)
        hist = result.scalar_one_or_none()
        return _history_payload(hist) if hist else None


async def create_history(
    chat_id: str,
    title: str = DEFAULT_TITLE,
    user_id: Optional[int] = None,
) -> Optional[dict]:
    """
    Create a new empty chat session entry.
    If it already exists, the existing one is returned.
    """
    if user_id is None:
        return None

    async with async_session_factory() as session:
        query = select(ChatHistory).where(ChatHistory.session_uuid == chat_id)
        result = await session.execute(query)
        existing = result.scalar_one_or_none()

        if existing:
            if existing.deleted_at is not None:
                existing.deleted_at = None
                await session.commit()
                await session.refresh(existing)
            return _history_payload(existing)

        new_history = ChatHistory(
            session_uuid=chat_id,
            user_id=user_id,
            title=title,
        )
        session.add(new_history)
        await session.commit()
        await session.refresh(new_history)
        return _history_payload(new_history)


async def add_message(
    chat_id: str,
    message: dict,
    user_id: Optional[int] = None,
) -> Optional[dict]:
    """
    Append a message to an existing chat session.

    ``message`` should have keys: ``role`` and ``content``.
    ``token_count`` is optional.
    """
    if user_id is None:
        return None

    async with async_session_factory() as session:
        query = select(ChatHistory).where(ChatHistory.session_uuid == chat_id)
        result = await session.execute(query)
        history = result.scalar_one_or_none()

        if history is None:
            history = ChatHistory(
                session_uuid=chat_id,
                user_id=user_id,
                title=DEFAULT_TITLE,
            )
            session.add(history)
            await session.flush()
        elif history.user_id != user_id:
            return None

        role = message.get("role", "user")
        new_message = ChatMessage(
            session_id=history.id,
            role=role,
            content=message.get("content", ""),
            token_count=message.get("token_count"),
        )
        session.add(new_message)

        if (history.title or DEFAULT_TITLE) == DEFAULT_TITLE and role == "user":
            content = message.get("content", "")
            history.title = content[:80] + ("..." if len(content) > 80 else "")

        await session.commit()

        query = select(ChatHistory).where(ChatHistory.session_uuid == chat_id)
        result = await session.execute(query)
        updated_history = result.scalar_one()
        return _history_payload(updated_history)


async def delete_history(chat_id: str) -> bool:
    """
    Soft delete a chat history. Returns True if deleted, False if not found.
    """
    async with async_session_factory() as session:
        query = select(ChatHistory).where(
            ChatHistory.session_uuid == chat_id,
            ChatHistory.deleted_at.is_(None),
        )
        result = await session.execute(query)
        history = result.scalar_one_or_none()

        if history is None:
            return False

        history.deleted_at = _now_text()
        await session.commit()
        return True


async def update_title(chat_id: str, title: str) -> Optional[dict]:
    """Update the title of a chat history. Returns updated history or None."""
    async with async_session_factory() as session:
        query = select(ChatHistory).where(
            ChatHistory.session_uuid == chat_id,
            ChatHistory.deleted_at.is_(None),
        )
        result = await session.execute(query)
        history = result.scalar_one_or_none()

        if history is None:
            return None

        history.title = title
        history.updated_at = _now_text()
        await session.commit()
        await session.refresh(history)
        return _history_payload(history)
