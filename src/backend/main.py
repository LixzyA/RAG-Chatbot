from fastapi import FastAPI
from contextlib import asynccontextmanager
from logger import configure_logging, LogLevels
from dotenv import load_dotenv
from api import conf_routing

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: init model
    configure_logging(LogLevels.info)
    yield


app = FastAPI(lifespan=lifespan)
conf_routing(app)

# TODO: get model, db, and storage status
@app.get("/health")
def health():
    return {"status": "ok"}