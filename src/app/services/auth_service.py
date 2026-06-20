"""Auth business logic — password hashing, JWT creation/decoding, user registration/login.

Source: backend/auth/service.py
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.entity.user import User
from app.models.requests import LoginRequest, RegisterRequest
from app.utils.exceptions import (
    DBInsertionException,
    EmailAlreadyExistException,
    InvalidTokenException,
    InvalidUsernamePasswordException,
    UserNotFoundException,
    UsernameAlreadyExistException,
)

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
_password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _password_hash.verify(plain, hashed)


def create_access_token(user_id: int) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": str(user_id),
        "exp": expires,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> int | None:
    """Return the user id from a JWT, or ``None`` if invalid/expired."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
        user_id: str | None = payload.get("sub")
        return int(user_id) if user_id is not None else None
    except (jwt.PyJWTError, ValueError):
        return None


async def register_user(db: AsyncSession, body: RegisterRequest) -> User:
    """Create a new user account. Raises on conflict."""
    existing = await db.execute(select(User).where(User.username == body.username))
    if existing.scalar_one_or_none():
        raise EmailAlreadyExistException

    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise UsernameAlreadyExistException

    user = User(
        username=body.username,
        email=body.email,
        password_hash=hash_password(body.password),
    )
    try:
        db.add(user)
        await db.commit()
        await db.refresh(user)
    except Exception as exc:
        await db.rollback()
        raise DBInsertionException(str(exc)) from exc

    logger.info("User registered: %s (%s)", user.username, user.id)
    return user


async def login_user(db: AsyncSession, body: LoginRequest) -> tuple[str, User]:
    """Authenticate a user, return ``(access_token, user)``."""
    result = await db.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(body.password, user.password_hash):
        raise InvalidUsernamePasswordException

    user.last_login_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    await db.commit()
    await db.refresh(user)

    token = create_access_token(user.id)
    logger.info("User logged in: %s (%s)", user.username, user.id)
    return token, user


async def get_current_user(db: AsyncSession, token: str) -> User:
    """Validate a JWT and return the corresponding ``User``."""
    user_id = decode_access_token(token)
    if user_id is None:
        raise InvalidTokenException

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise UserNotFoundException

    return user
