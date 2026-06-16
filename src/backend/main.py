from dotenv import load_dotenv
from fastapi import FastAPI  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from logger import configure_logging  # noqa: E402
from api import conf_routing  # noqa: E402
from chat.core import init_llm, llm_healthcheck  # noqa: E402
from vectordb.core import ChromaDB  # noqa: E402
from vectordb.reranker import Reranker  # noqa: E402
from entity.base import init_db  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
import asyncio  # noqa: E402

load_dotenv()
logger = configure_logging(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application...")

    # Quick init that shouldn't block startup
    init_llm()
    app.state.reranker = Reranker()

    async def load_reranker_background():
        try:
            logger.info("Loading reranker in background...")
            await asyncio.to_thread(app.state.reranker._load)
            logger.info("Reranker loaded")
        except Exception as e:
            logger.exception("Failed to load reranker: %s", e)

    async def init_vector_db_background():
        try:
            logger.info("Initializing vector DB in background...")
            app.state.vector_db = await asyncio.to_thread(lambda: ChromaDB(reranker=app.state.reranker))
            logger.info("Vector DB initialized")
        except Exception as e:
            logger.exception("Failed to initialize vector DB: %s", e)

    # Schedule background tasks so startup isn't blocked
    asyncio.create_task(load_reranker_background())
    asyncio.create_task(init_vector_db_background())

    async def run_init_db():
        try:
            await init_db()
            logger.info("Database initialization completed.")
        except Exception as e:
            logger.exception("Database initialization failed: %s", e)

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
    try:
        vectordb_client = app.state.vector_db
        status = vectordb_client.heartbeat()
    except Exception as e:
        logger.error(f'Vector DB error: {e}')
        status = False

    llm_status = await llm_healthcheck()
    return {
        "chroma_status": "ok" if status else "error",
        "llm_status": "ok" if llm_status else "error",
        "storage_status": "ok"
    }