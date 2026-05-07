from fastapi import UploadFile
from . import models
import logging
import pathlib
from exception import FileNameNotFound, FileTypeNotSupportedException, PDFProcessingException
import os
from pdfminer.high_level import extract_text

ALLOWED_FILE_TYPE = ['.txt', '.pdf']

async def upload_file(file: UploadFile) -> models.UploadFileResponse:
    if file.filename is None:
        logging.warning("File name not found")
        raise FileNameNotFound()
    file_type = pathlib.Path(file.filename).suffix
    if file_type not in ALLOWED_FILE_TYPE:
        logging.warning(f"Invalid file type. Received file type: {file_type}")
        raise FileTypeNotSupportedException(file_type=file_type)

    content = await file.read()
    logging.info(f"File {file.filename} uploaded successfully")

    # Save file to local
    save_dir = os.getenv("LOCAL_DATA_PATH", "../../data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logging.info(f"Directory {save_dir} created successfully")
    file_path = os.path.join(save_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)
    logging.info(f"File {file.filename} saved to local at {file_path}")
    
    # Extract text
    text_content = ""
    if file_type == ".txt":
        text_content = content.decode("utf-8")
    elif file_type == ".pdf":
        logging.info(f"Processing PDF")
        try:
            text_content = extract_text(file_path)
            logging.info(f"File {file.filename} processed successfully\nFile content:")
            logging.info(text_content)
        except Exception as e:
            logging.error(f"Failed to process PDF file: {str(e)}")
            raise PDFProcessingException(message=str(e))

    # TODO: Chunking
    # TODO: Embedding
    # TODO: Uperst to Chroma

    return models.UploadFileResponse(
        status="success",
        num_chunk=1,
        file_content=text_content
    )
        