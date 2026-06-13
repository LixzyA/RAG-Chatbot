from fastapi import APIRouter, UploadFile, Depends
from . import service
from .models import UploadFileResponse
from vectordb.core import VectorDBClient
from logger import configure_logging

logger = configure_logging(__name__)

router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload", response_model=UploadFileResponse)
async def upload_file(file: UploadFile, client: VectorDBClient):
    logger.info(f"Received file upload request for file: {file.filename}")
    res = await service.upload_file(file, client)
    logger.info(f"Successfully processed and chunked file: {file.filename} into {res.num_chunk} chunks")
    return res
