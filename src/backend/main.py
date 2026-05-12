from fastapi import FastAPI
from contextlib import asynccontextmanager
from logger import configure_logging, LogLevels
from dotenv import load_dotenv
from api import conf_routing
from chat.core import init_llm, llm_healthcheck
from vectordb.core import init_chroma_client
from fastapi.middleware.cors import CORSMiddleware
import os

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_llm()
    configure_logging(LogLevels.info)
    init_chroma_client()
    
    yield


app = FastAPI(lifespan=lifespan)
conf_routing(app)

origins = [
    "http://localhost:5173", # React Frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    vectordb_client = init_chroma_client()
    status = vectordb_client.heartbeat()

    llm_status = await llm_healthcheck()
    return {
    "chroma_status": "ok" if status else "error", 
    "llm_status": "ok" if llm_status else "error",
    "storage_status": "ok"}