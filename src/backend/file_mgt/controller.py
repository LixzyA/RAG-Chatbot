from fastapi import APIRouter,UploadFile, Depends
from . import service
from .models import UploadFileResponse
from vectordb.core import VectorDBClient

router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload", response_model=UploadFileResponse)
async def upload_file(file: UploadFile, client: VectorDBClient):
   return await service.upload_file(file, client)
