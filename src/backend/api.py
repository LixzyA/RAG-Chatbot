from fastapi import FastAPI
from file_mgt.controller import router as file_router
from chat.controller import router as chat_router

def conf_routing(app: FastAPI):
    app.include_router(file_router)
    app.include_router(chat_router)