from vectordb.core import query_collection
import logging
import os
from collections.abc import AsyncIterable

TOP_K = int(os.getenv("MAX_RETRIEVAL", 10))
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

async def query_chat(db_client, llm_client, prompt: str, top_k: int) -> AsyncIterable:
    logging.info(f"Received chat query: {prompt}")

    files: dict = query_collection(db_client, "file_mgt",query_text=prompt, top_k=top_k)
    logging.info(f"Retrieved {len(files['ids'][0])} relevant files from vector database")
    
    # menyusun prompt (context + question) untuk dikirim ke LLM
    context_text = "\n\n".join([res for res in files['documents'][0]])
    
    full_prompt = (
        "Context information is below.\n"
        "---------------------\n"
        f"{context_text}\n"
        "---------------------\n"
        f"Given the context information and not prior knowledge, answer the query: {prompt}"
    )

    # send prompt to LLM
    response = await llm_client.chat.completions.create(
        model = LLM_MODEL,
        messages = [
            {"role": "system", "content": "You are a helpful assistant for answering questions based on retrieved documents."},
            {"role": "user", "content": full_prompt}
        ],
        stream = True,
        max_tokens=1024,
    )
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content

