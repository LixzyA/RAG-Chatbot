from typing import Annotated, Any, Mapping
from annotated_doc import Doc
from fastapi import HTTPException
 
class FileNameNotFound(HTTPException):
    def __init__(self):
        self.status_code = 404
        self.detail = "File name not found"
        super().__init__(status_code=self.status_code, detail=self.detail)

class FileTypeNotSupportedException(HTTPException):
    def __init__(self, file_type):
        self.status_code = 400
        self.detail = f"File type {file_type} not supported"
        super().__init__(status_code=self.status_code, detail=self.detail)

class PDFProcessingException(HTTPException):
    def __init__(self, message):
        self.status_code = 500
        self.detail = f"Failed to process PDF file: {message}"
        super().__init__(status_code=self.status_code, detail=self.detail)

class CreateCollectionException(HTTPException):
    def __init__(self, message):
        self.status_code = 500
        self.detail = f"Failed to create collection: {message}"
        super().__init__(status_code=self.status_code, detail=self.detail)

class CollectionNotFoundException(HTTPException):
    def __init__(self, message):
        self.status_code = 404
        self.detail = f"Collection not found: {message}"
        super().__init__(status_code=self.status_code, detail=self.detail)

class ChromaInsertionException(HTTPException):
    def __init__(self, message):
        self.status_code = 500
        self.detail = f"Failed to insert data into Chroma: {message}"
        super().__init__(status_code=self.status_code, detail=self.detail)

class ChromaQueryException(HTTPException):
    def __init__(self, message):
        self.status_code = 500
        self.detail = f"Failed to query collection: {message}"
        super().__init__(status_code=self.status_code, detail=self.detail)

class LLMException(HTTPException):
    def __init__(self, message):
        self.status_code = 500
        self.detail = f"Failed to generate response: {message}"
        super().__init__(status_code=self.status_code, detail=self.detail)

class DBInsertionException(HTTPException):
    def __init__(self, status_code: int, detail: Any = None, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(status_code, detail, headers)

class EmailAlreadyExistException(HTTPException):
    def __init__(self, status_code: int = 409, detail: Any = "Email already exists!", headers: Mapping[str, str] | None = None) -> None:
        super().__init__(status_code, detail, headers)

class UsernameAlreadyExistException(HTTPException):
    def __init__(self, status_code: int = 409, detail: Any = "Uername already exists!", headers: Mapping[str, str] | None = None) -> None:
        super().__init__(status_code, detail, headers)

class InvalidUsernamePasswordException(HTTPException):
    def __init__(self, status_code: int = 401, detail: Any = "Invalid username or password", headers: Mapping[str, str] | None = None) -> None:
        super().__init__(status_code, detail, headers)

class InvalidTokenException(HTTPException):
    def __init__(self, status_code: int = 401, detail: Any = "Invalid or expired token", headers: Mapping[str, str] | None = None) -> None:
        super().__init__(status_code, detail, headers)

class UserNotFoundException(HTTPException):
    def __init__(self, status_code: int = 404, detail: Any = "User not found", headers: Mapping[str, str] | None = None) -> None:
        super().__init__(status_code, detail, headers)

class EmbeddingException(HTTPException):
    def __init__(self, status_code: int=500, detail: Any = "Unexpected embedding exception!", headers: Mapping[str, str] | None = None) -> None:
        super().__init__(status_code, detail, headers)

class RerankerException(HTTPException):
    def __init__(self, status_code: int=500, detail: Any = "Unexpected reranker exception!", headers: Mapping[str, str] | None = None) -> None:
        super().__init__(status_code, detail, headers)