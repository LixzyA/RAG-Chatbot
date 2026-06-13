from fastapi import FastAPI
from contextlib import asynccontextmanager
from logger import configure_logging
from dotenv import load_dotenv
from api import conf_routing
from chat.core import init_llm, llm_healthcheck
from vectordb.core import init_chroma_client
from entity.base import init_db
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os

logger = configure_logging(__name__)

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application...")
    
    init_llm()
    init_chroma_client()
    
    async def run_init_db():
        await init_db()
        logger.info("Database initialization completed.")
        
    asyncio.create_task(run_init_db())
    
    yield
    logger.info("Shutting down FastAPI application...")


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