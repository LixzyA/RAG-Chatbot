from fastapi import APIRouter, Request
from . import service

router = APIRouter(prefix="/chat", tags=["chat"])

# TODO: implement chat and sse
@router.post("/")
def query_chat(prompt: str):
    return service.query_chat(prompt)
