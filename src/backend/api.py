from fastapi import FastAPI
from file_mgt.controller import router as file_router

def conf_routing(app: FastAPI):
    app.include_router(file_router)