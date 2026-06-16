from vectordb.core import ChromaDB
from . import models
from exception import FileNameNotFound, FileTypeNotSupportedException
import asyncio
import pathlib
from io import BytesIO
from fastapi import UploadFile
from pdfminer.high_level import extract_text as pdf_extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

ALLOWED_FILE_TYPE = ['.txt', '.pdf']
CHUNK_SIZE = 1024
OVERLAP = int(0.2 * CHUNK_SIZE)

async def upload_file(file: UploadFile, client: ChromaDB) -> models.UploadFileResponse:
    if file.filename is None:
        raise FileNameNotFound()
    file_type = pathlib.Path(file.filename).suffix
    if file_type not in ALLOWED_FILE_TYPE:
        raise FileTypeNotSupportedException(file_type=file_type)

    content = await file.read()
    text_content = pdf_extract_text(BytesIO(content)) if file_type == '.pdf' else content.decode("utf-8")

    chunks = await asyncio.to_thread(recursive_chunk, text_content, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    total_chunk = len(chunks)
    documents = [
        Document(
            page_content=chunk,
            metadata={"filename": file.filename, 'total_chunk': len(chunks), 'chunk_num': i}
        )
        for i, chunk in enumerate(chunks)
    ]
    ids = [f"{file.filename}_{i}" for i in range(total_chunk)]
    client.vector_add_documents(documents, ids=ids)

    return models.UploadFileResponse(
        status=200,
        num_chunk=total_chunk,
    )

# Token-based chunking using tiktoken
def recursive_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        model_name="gpt-4",
    )
    documents = text_splitter.create_documents([text])
    chunked_texts = [doc.page_content for doc in documents]
    return chunked_texts 
