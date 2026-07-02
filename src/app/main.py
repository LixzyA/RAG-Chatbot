import asyncio
import logging
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.generation import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.ingestion import router as ingest_router
from app.api.routes.retrieval import router as retrieval_router
from app.services.vector_db import reset_store
from app.services.rag_chain import reset_rag_chain
from app.entity.base import init_db
from app.services.vector_db import get_reranker, get_vector_store
from app.utils.logger import setup_logging
from app.utils.exceptions import handle_app_exception, AppException
from prometheus_fastapi_instrumentator import Instrumentator

setup_logging()
logger = logging.getLogger(__name__)

# Suppress noisy HuggingFace tokenizer warning
warnings.filterwarnings(
    "ignore",
    message="You're using a XLMRobertaTokenizerFast tokenizer.*",
)


async def _load_reranker(reranker):
    try:
        logger.info("Loading reranker in background...")
        await asyncio.to_thread(reranker._load)
        logger.info("Reranker loaded")
    except Exception:
        logger.exception("Failed to load reranker")


async def _init_db():
    try:
        logger.info("Initialising database...")
        await init_db()
        logger.info("Database initialised")
    except Exception:
        logger.exception("Failed to initialise database")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application...")

    # Triggers singleton creation
    get_vector_store()
    reranker = get_reranker()

    # Background tasks so startup isn't blocked
    asyncio.create_task(_load_reranker(reranker))
    asyncio.create_task(_init_db())
    instrumentator.expose(app)

    yield

    logger.info("Shutting down...")
    reset_store()
    reset_rag_chain()


app = FastAPI(
    title="RAG Chatbot",
    version="2.0.0",
    lifespan=lifespan,
)
origins = [
    "http://localhost:5173",  # React Frontend
]

# CORS (permissive for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(retrieval_router)
Instrumentator().instrument(app).expose(app)
instrumentator = Instrumentator().instrument(app)

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return await handle_app_exception(request, exc)
