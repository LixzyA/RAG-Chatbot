from pydantic import BaseModel

class UploadFileResponse(BaseModel):
    status: int
    num_chunk: int
    