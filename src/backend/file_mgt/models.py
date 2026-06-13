from fastapi import UploadFile
from pydantic import BaseModel

class UploadFileResponse(BaseModel):
    status: str
    num_chunk: int
    file_content: str