"""Auth endpoints — register, login, and profile.

POST /auth/register
POST /auth/login
GET  /auth/me
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import AuthenticatedUser, get_db
from app.models.requests import LoginRequest, RegisterRequest
from app.models.responses import TokenResponse, UserResponse
from app.services import auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse)
async def register(body: RegisterRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    """Create a new user account."""
    user = await auth_service.register_user(db, body)
    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Authenticate and receive a JWT access token."""
    body = LoginRequest(username=form_data.username, password=form_data.password)
    token, user = await auth_service.login_user(db, body)
    return TokenResponse(
        access_token=token,
        user=UserResponse.model_validate(user),
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: AuthenticatedUser):
    """Return the currently authenticated user's profile."""
    return UserResponse.model_validate(current_user)
