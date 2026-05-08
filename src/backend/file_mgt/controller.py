from fastapi import APIRouter,UploadFile, Depends
from . import service
from .models import UploadFileResponse
from vectordb.core import VectorDBClient

router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload")
async def upload_file(file: UploadFile, client: VectorDBClient) -> UploadFileResponse:
   return await service.upload_file(file, client)
