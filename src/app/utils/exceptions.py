"""Domain-specific exceptions used across the application.

These replace the FastAPI HTTPException subclasses from backend/exception.py
with plain Exception classes so the core layer stays framework-agnostic.
Routes catch these and convert them to appropriate HTTP responses.
"""


class AppException(Exception):
    """Base application exception."""
    pass


class FileNameNotFound(AppException):
    """No filename was provided for an upload."""
    pass


class FileTypeNotSupportedException(AppException):
    """The uploaded file type is not supported."""
    pass


class PDFProcessingException(AppException):
    """Failed to extract text from a PDF."""
    pass


class CreateCollectionException(AppException):
    """Failed to create a vector DB collection."""
    pass


class CollectionNotFoundException(AppException):
    """The requested vector DB collection does not exist."""
    pass


class ChromaInsertionException(AppException):
    """Failed to insert documents into ChromaDB."""
    pass


class ChromaQueryException(AppException):
    """Failed to query ChromaDB."""
    pass


class LLMException(AppException):
    """The LLM refused or failed to generate a response."""
    pass


class DBInsertionException(AppException):
    """Failed to insert a record into the relational database."""
    pass


class EmailAlreadyExistException(AppException):
    """A user with that email already exists."""
    pass


class UsernameAlreadyExistException(AppException):
    """A user with that username already exists."""
    pass


class InvalidUsernamePasswordException(AppException):
    """The provided credentials are incorrect."""
    pass


class InvalidTokenException(AppException):
    """The JWT token is missing, expired, or invalid."""
    pass


class UserNotFoundException(AppException):
    """The requested user does not exist."""
    pass


class EmbeddingException(AppException):
    """The embedding model failed to encode text."""
    pass


class RerankerException(AppException):
    """The cross-encoder reranker failed."""
    pass
