from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from entity.base import get_session
from . import service
from .models import RegisterRequest, LoginRequest, TokenResponse, UserResponse
from .dependencies import AuthenticatedUser
from logger import configure_logging

logger = configure_logging(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_session)):
    """Create a new user account."""
    res = await service.register_user(db, body)
    logger.info(f"User registered: {res.username} ({res.id})")
    return res


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_session),
):
    """Authenticate and receive a JWT access token."""
    body = LoginRequest(username=form_data.username, password=form_data.password)
    res = await service.login_user(db, body)
    logger.info(f"User logged in: {res.user.username} ({res.user.id})")
    return res


@router.get("/me", response_model=UserResponse)
async def me(current_user: AuthenticatedUser):
    """Return the currently authenticated user's profile."""
    return UserResponse.model_validate(current_user)
