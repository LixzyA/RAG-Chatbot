from fastapi import APIRouter, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from . import service
from vectordb.core import VectorDBClient
from . import models
from . import core
router = APIRouter(prefix="/chat", tags=["chat"])

# TODO: implement chat and sse
@router.post("/", response_class=EventSourceResponse)
async def query_chat(   
    query_req: models.ChatQueryRequest,
    db_client: VectorDBClient):
    llm_client=core.llm_client

    async for chunk in service.query_chat(db_client, llm_client, **query_req.model_dump()):
        yield ServerSentEvent(data=chunk)
    yield ServerSentEvent(raw_data="[DONE]", event="done")