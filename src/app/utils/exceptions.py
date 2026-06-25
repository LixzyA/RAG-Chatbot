from typing import Optional, Dict, Any
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base application exception."""

    # Default attributes to be overridden by subclasses
    status_code: int = 500
    default_message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.status_code = status_code if status_code is not None else self.status_code
        self.message = message if message is not None else self.default_message
        self.details = details or {}
        super().__init__(self.message)


# --- Subclasses with specific defaults ---


class FileNameNotFound(AppException):
    """No filename was provided for an upload."""

    status_code = 400  # Bad Request
    default_message = "No filename was provided for the upload."


class FileTypeNotSupportedException(AppException):
    """The uploaded file type is not supported."""

    status_code = 400
    default_message = "The uploaded file type is not supported."


class PDFProcessingException(AppException):
    """Failed to extract text from a PDF."""

    status_code = 500
    default_message = "Failed to process the PDF file."


class CreateCollectionException(AppException):
    """Failed to create a vector DB collection."""

    status_code = 500
    default_message = "Failed to create the vector DB collection."


class CollectionNotFoundException(AppException):
    """The requested vector DB collection does not exist."""

    status_code = 404  # Not Found
    default_message = "The requested collection does not exist."


class ChromaInsertionException(AppException):
    """Failed to insert documents into ChromaDB."""

    status_code = 500
    default_message = "Failed to insert documents into ChromaDB."


class ChromaQueryException(AppException):
    """Failed to query ChromaDB."""

    status_code = 500
    default_message = "Failed to query ChromaDB."


class LLMException(AppException):
    """The LLM refused or failed to generate a response."""

    status_code = 502  # Bad Gateway (External API failure)
    default_message = "The AI model failed to generate a response."


class DBInsertionException(AppException):
    """Failed to insert a record into the relational database."""

    status_code = 500
    default_message = "Database operation failed."


class EmailAlreadyExistException(AppException):
    """A user with that email already exists."""

    status_code = 409  # Conflict
    default_message = "A user with this email already exists."


class UsernameAlreadyExistException(AppException):
    """A user with that username already exists."""

    status_code = 409  # Conflict
    default_message = "A user with this username already exists."


class InvalidUsernamePasswordException(AppException):
    """The provided credentials are incorrect."""

    status_code = 401  # Unauthorized
    default_message = "Invalid username or password."


class InvalidTokenException(AppException):
    """The JWT token is missing, expired, or invalid."""

    status_code = 401
    default_message = "Invalid or expired token."


class UserNotFoundException(AppException):
    """The requested user does not exist."""

    status_code = 404
    default_message = "The requested user does not exist."


class EmbeddingException(AppException):
    """The embedding model failed to encode text."""

    status_code = 500
    default_message = "Failed to generate text embeddings."


class RerankerException(AppException):
    """The cross-encoder reranker failed."""

    status_code = 500
    default_message = "The reranker model failed."


# --- Exception Handler ---


async def handle_app_exception(request: Request, exc: AppException):
    # Log server-side errors (5xx) for monitoring and debugging
    if exc.status_code >= 500:
        logger.error(
            f"AppException: {exc.__class__.__name__} - {exc.message}", exc_info=True
        )

    # FastAPI natively uses "detail" for error messages, so we align with that standard
    content: dict[str, Any] = {"detail": exc.message}

    # Optionally include extra context/details if provided
    if exc.details:
        content["extra"] = exc.details

    return JSONResponse(status_code=exc.status_code, content=content)
