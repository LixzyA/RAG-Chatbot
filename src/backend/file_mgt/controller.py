from fastapi import APIRouter, UploadFile
from . import service
from . import models
from vectordb.core import VectorDBClient
from auth.dependencies import AuthenticatedUser
from logger import configure_logging

logger = configure_logging(__name__)

router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload", response_model=models.UploadFileResponse)
async def upload_file(file: UploadFile, client: VectorDBClient, current_user: AuthenticatedUser):
    logger.info(f"User: {current_user}, Received file upload request for file: {file.filename}")
    res = await service.upload_file(file, client)
    logger.info(f"User: {current_user}, Successfully processed and chunked file: {file.filename} into {res.num_chunk} chunks")
    return res
