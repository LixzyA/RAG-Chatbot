# import logging # [LOGGING REMOVED]
from huggingface_hub import AsyncInferenceClient
from chat.service import SPECIALIST_MODEL

llm_client = None

def init_llm():
    global llm_client
    if llm_client is None:
        # [LOGGING REMOVED]
        llm_client = AsyncInferenceClient()
        # [LOGGING REMOVED]
    return llm_client

async def llm_healthcheck():
    global llm_client
    if llm_client is None:
        # [LOGGING REMOVED]
        return False
    try:
        response = await llm_client.chat.completions.create(
            model=SPECIALIST_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.1
        )
        # [LOGGING REMOVED]
        return True
    except Exception as e:
        # [LOGGING REMOVED]
        return False