from fastapi import APIRouter, Request

router = APIRouter(prefix="/chat", tags=["chat"])

# TODO: implement chat and sse
@router.post("/")
def query_chat(request: Request):
    pass
