from fastapi import APIRouter,UploadFile
from . import service
from .models import UploadFileResponse

router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload")
async def upload_file(request: UploadFile) -> UploadFileResponse:
   return await service.upload_file(request)
