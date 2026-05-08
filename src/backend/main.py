from fastapi import FastAPI
from contextlib import asynccontextmanager
from logger import configure_logging, LogLevels
from dotenv import load_dotenv
from api import conf_routing
from chat.core import init_llm
from vectordb.core import create_collection, init_chroma_client
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_llm()
    configure_logging(LogLevels.info)
    client = init_chroma_client()
    # create_collection(client, name="file_mgt")
    yield


app = FastAPI(lifespan=lifespan)
conf_routing(app)

# TODO: get model, db, and storage status
@app.get("/health")
def health():
    return {"status": "ok"}