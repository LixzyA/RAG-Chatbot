# import logging # [LOGGING REMOVED]
from huggingface_hub import AsyncInferenceClient
from chat.service import GENERALIST_MODEL

llm_client = None

def init_llm():
    global llm_client
    if llm_client is None:
        llm_client = AsyncInferenceClient()
    return llm_client

async def llm_healthcheck():
    global llm_client
    if llm_client is None:
        return False
    response = await llm_client.chat.completions.create(
        model=GENERALIST_MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.1
    )
    if not response:
        return False
    return True
    