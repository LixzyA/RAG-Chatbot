from .models import RegisterRequest, LoginRequest, TokenResponse, UserResponse
from entity.user import User
from exception import (
    DBInsertionException, 
    EmailAlreadyExistException, 
    UsernameAlreadyExistException, 
    InvalidUsernamePasswordException,
    InvalidTokenException,
    UserNotFoundException)
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production-use-a-real-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 h

password_hash = PasswordHash.recommended()
def _hash_password(password: str) -> str:
    return password_hash.hash(password)

def _verify_password(plain: str, hashed: str) -> bool:
    return password_hash.verify(plain, hashed)

def _create_access_token(user_id: int) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "exp": expires,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM) #type: ignore

def decode_access_token(token: str) -> Optional[int]:
    """
    Decode a JWT and return the ``sub`` (user id).
    Returns ``None`` if the token is invalid or expired.
    """
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]) #type: ignore
    user_id: str | None = payload.get("sub")
    return int(user_id) if user_id is not None else None


async def register_user(db: AsyncSession, body: RegisterRequest) -> UserResponse:
    """Register a new user. Raises ``HTTPException`` on conflicts."""

    # Check username uniqueness
    existing = await db.execute(select(User).where(User.username == body.username))
    if existing.scalar_one_or_none():
        raise EmailAlreadyExistException

    # Check email uniqueness
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise UsernameAlreadyExistException

    user = User(
        username=body.username,
        email=body.email,
        password_hash=_hash_password(body.password),
    )
    try:
        db.add(user)
        await db.commit()
        await db.refresh(user)
    except Exception as e:
        await db.rollback()
        raise DBInsertionException(500, str(e))

    return UserResponse.model_validate(user)


async def login_user(db: AsyncSession, body: LoginRequest) -> TokenResponse:
    """Authenticate a user and return an access token."""

    result = await db.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()

    if not user or not _verify_password(body.password, user.password_hash):
        raise InvalidUsernamePasswordException

    user.last_login_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    await db.commit()
    await db.refresh(user)

    token = _create_access_token(user.id)
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


async def get_current_user(db: AsyncSession, token: str) -> User:
    """Return the ``User`` instance for a valid JWT token."""
    user_id = decode_access_token(token)
    if user_id is None:
        raise InvalidTokenException

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise UserNotFoundException

    return user
