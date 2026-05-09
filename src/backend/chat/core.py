import logging
from huggingface_hub import AsyncInferenceClient
from chat.service import SPECIALIST_MODEL

llm_client = None

def init_llm():
    global llm_client
    if llm_client is None:
        logging.info("Initializing LLM...")
        llm_client = AsyncInferenceClient()
        logging.info("LLM initialized.")
    return llm_client

async def llm_healthcheck():
    global llm_client
    if llm_client is None:
        logging.warning("LLM not initialized")
        return False
    try:
        response = await llm_client.chat.completions.create(
            model=SPECIALIST_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.1
        )
        logging.info(f"LLM healthcheck passed: {response.choices[0].message.content}")
        return True
    except Exception as e:
        logging.warning(f"LLM healthcheck failed: {str(e)}")
        return False