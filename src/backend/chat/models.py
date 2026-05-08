from pydantic import BaseModel

class ChatQueryRequest(BaseModel):
    prompt: str
    top_k: int | None = 10