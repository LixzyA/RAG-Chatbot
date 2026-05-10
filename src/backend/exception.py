from fastapi import HTTPException
 
class FileNameNotFound(HTTPException):
    def __init__(self):
        self.status_code = 404
        self.detail = f"File name not found"
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
    
class OverlapException(HTTPException):
    def __init__(self, message):
        self.status_code = 400
        self.detail = f"Invalid overlap: {message}"
        super().__init__(status_code=self.status_code, detail=self.detail)

class LLMException(HTTPException):
    def __init__(self, message):
        self.status_code = 500
        self.detail = f"Failed to generate response: {message}"
        super().__init__(status_code=self.status_code, detail=self.detail)