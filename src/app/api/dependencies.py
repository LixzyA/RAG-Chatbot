"""Shared FastAPI dependency providers (DB sessions, vector store, RAG chain, auth, etc.).

These are injected into route handlers via ``Depends(...)`` so tests can
override them with fakes.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.entity.base import async_session
from app.entity.user import User
from app.services import auth_service
from app.services.rag_chain import get_rag_chain
from app.services.vector_db import get_vector_store, get_reranker
from app.core.orchestration.rag_chain import RAGChain
from app.core.retrieval.vector_store import VectorStore
from app.core.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Database
# ------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


async def _required_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    return await auth_service.get_current_user(db, token)


async def _optional_user(
    token: str | None = Depends(oauth2_scheme_optional),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    if not token:
        return None
    return await auth_service.get_current_user(db, token)


AuthenticatedUser = Annotated[User, Depends(_required_user)]
OptionalAuthenticatedUser = Annotated[User | None, Depends(_optional_user)]


# ------------------------------------------------------------------
# Vector store & reranker
# ------------------------------------------------------------------

def get_vector_db() -> VectorStore:
    """Return the singleton ``VectorStore``."""
    return get_vector_store()


def get_reranker_dep() -> CrossEncoderReranker:
    """Return the singleton ``CrossEncoderReranker``."""
    return get_reranker()


def get_rag_chain_dep() -> RAGChain:
    """Return the singleton :class:`RAGChain` orchestrator.

    Override this in tests via ``app.dependency_overrides[get_rag_chain_dep]``.
    """
    return get_rag_chain()
