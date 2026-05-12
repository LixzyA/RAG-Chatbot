from fastapi import APIRouter, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from . import service
from vectordb.core import VectorDBClient
from . import models
from . import core
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_class=EventSourceResponse)
async def query_chat(   
    query_req: models.ChatQueryRequest,
    db_client: VectorDBClient):
    llm_client=core.llm_client

    async for chunk in service.query_chat(db_client, llm_client, **query_req.model_dump()):
        yield ServerSentEvent(raw_data=chunk)
    yield ServerSentEvent(raw_data="[DONE]", event="done")


@router.post("/v2", response_class=EventSourceResponse)
async def query_chat_v2(   
    query_req: models.ChatQueryRequest,
    db_client: VectorDBClient):
    llm_client=core.llm_client

    files = service.get_relevant_files(db_client, query_req.prompt, query_req.top_k)
    docs = files.get("documents")
    context_text = "\n\n".join([res for res in docs[0]]) if docs and docs[0] else ""

    async for chunk in service.get_answers(llm_client, query_req.prompt, context_text):
        yield ServerSentEvent(raw_data=chunk)
    yield ServerSentEvent(raw_data="[DONE]", event="done")
