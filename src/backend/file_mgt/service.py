from typing import List
from fastapi import UploadFile
from . import models
import logging
import pathlib
from exception import FileNameNotFound, FileTypeNotSupportedException, PDFProcessingException, OverlapException
import os
from pdfminer.high_level import extract_text as pdf_extract_text
import asyncio
from vectordb.core import add_data_to_collection

ALLOWED_FILE_TYPE = ['.txt', '.pdf']
CHUNK_SIZE = 1024
OVERLAP = int(0.2 * CHUNK_SIZE)

async def upload_file(file: UploadFile, client) -> models.UploadFileResponse:
    if file.filename is None:
        logging.warning("File name not found")
        raise FileNameNotFound()
    file_type = pathlib.Path(file.filename).suffix
    if file_type not in ALLOWED_FILE_TYPE:
        logging.warning(f"Invalid file type. Received file type: {file_type}")
        raise FileTypeNotSupportedException(file_type=file_type)
    
    # Get chroma collection
    collection = client.get_collection(name="file_mgt")

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
    if file_type == ".pdf":
        text_content = extract_text(file_path)
    else:
        text_content = content.decode("utf-8")
    # Chunk text
    chunks = await asyncio.to_thread(chunk_text, text_content, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    # Upload to Chroma
    await add_data_to_collection(data={"documents": chunks, "metadatas": [{"file": file.filename}] * len(chunks), "ids": [f"{file.filename}_{i}" for i in range(len(chunks))]}, collection=collection)


    return models.UploadFileResponse(
        status="success",
        num_chunk=len(chunks),
        file_content=text_content
    )

def extract_text(file_path: str) -> str:
    try:
        text_content = pdf_extract_text(file_path)
        logging.info(f"File {file_path} processed successfully")
        return text_content
    except Exception as e:
        logging.error(f"Failed to process PDF file: {str(e)}")
        raise PDFProcessingException(message=str(e))

    
        
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
        Fixed size chunking with overlap
    """
    if overlap >= chunk_size:
        raise OverlapException(message="Overlap must be smaller than chunk size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # The 'Slide': Move the pointer forward by (size - overlap)
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return chunks