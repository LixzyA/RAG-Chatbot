from fastapi import Depends
from vectordb.core import query_collection, get_collection_by_id
from typing import List
import logging
import os
from FlagEmbedding import FlagReranker
from huggingface_hub import InferenceClient

reranker, llm_client = None, None
MAX_RETRIEVAL = int(os.getenv("MAX_RETRIEVAL", 10))
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

def query_chat(prompt: str, top_k: int = 5, reranker = Depends(init_reranker), llm_client = Depends(init_llm)):
    logging.info(f"Received chat query: {prompt}")

    files: List[dict] = query_collection(prompt, top_k=MAX_RETRIEVAL)
    logging.info(f"Retrieved {len(files)} relevant files from vector database")

    pairs = [(prompt, file['content']) for file in files]
    # reranking files for better context
    scores = reranker.compute_score(pairs)
    results = sorted(zip(pairs, scores), key=lambda x: x[1], reverse=True)
    
    # menyusun prompt (context + question) untuk dikirim ke LLM
    context_text = "\n\n".join([res[0][1] for res in results[:top_k]])
    
    full_prompt = (
        "Context information is below.\n"
        "---------------------\n"
        f"{context_text}\n"
        "---------------------\n"
        f"Given the context information and not prior knowledge, answer the query: {prompt}"
    )

    # kirim prompt ke LLM dan dapatkan response
    completion = llm_client.chat.completions.create(
        model = LLM_MODEL,
        messages = [
            {"role": "system", "content": "You are a helpful assistant for answering questions based on retrieved documents."},
            {"role": "user", "content": full_prompt}
        ]
    )
    answer = completion.choices[0].message.content
    logging.info(f"Generated answer: {answer}")
    # TODO: use sse to stream response from LLM

def init_reranker():
    global reranker
    if reranker is None:
        logging.info("Initializing reranker...")
        reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        logging.info("Reranker initialized.")
    return reranker
    
def init_llm():
    global llm_client
    if llm_client is None:
        logging.info("Initializing LLM...")
        client = InferenceClient()
        logging.info("LLM initialized.")
    return llm_client
