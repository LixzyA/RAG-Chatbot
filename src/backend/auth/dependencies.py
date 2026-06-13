"""
FastAPI dependency injection for auth.

Provides ``AuthenticatedUser`` — a dependency that extracts the current
user from the ``Authorization: Bearer <token>`` header.
"""

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, Optional

from entity.base import get_session
from entity.user import User
from .service import get_current_user

# OAuth2PasswordBearer automatically extracts the "Authorization: Bearer <token>" header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


async def _authenticated_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_session),
) -> User:
    return await get_current_user(db, token)


async def _optional_user(
    token: Optional[str] = Depends(oauth2_scheme_optional),
    db: AsyncSession = Depends(get_session),
) -> Optional[User]:
    if not token:
        return None
    return await get_current_user(db, token)
    

AuthenticatedUser = Annotated[User, Depends(_authenticated_user)]
OptionalAuthenticatedUser = Annotated[Optional[User], Depends(_optional_user)]
