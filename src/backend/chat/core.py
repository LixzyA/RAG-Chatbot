import logging
from huggingface_hub import AsyncInferenceClient
from fastapi import Depends
from typing import Annotated

llm_client = None

    
def init_llm():
    global llm_client
    if llm_client is None:
        logging.info("Initializing LLM...")
        llm_client = AsyncInferenceClient()
        logging.info("LLM initialized.")
    return llm_client
