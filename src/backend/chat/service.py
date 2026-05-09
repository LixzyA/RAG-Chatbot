from vectordb.core import query_collection
from chat.router import classify_query, is_serious_topic
import logging
import os
from collections.abc import AsyncIterable

TOP_K = int(os.getenv("MAX_RETRIEVAL", 10))
SPECIALIST_MODEL = os.getenv("SPECIALIST_MODEL", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
GENERALIST_MODEL = os.getenv("GENERALIST_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SPECIALIST_SYSTEM_PROMPT = (
    "You are an expert assistant specializing in providing precise, thorough, "
    "and well-structured answers to complex questions. Use the retrieved context "
    "to support your response. Be accurate and cite relevant details from the "
    "context. If the context does not contain enough information, clearly state "
    "what you know and what is uncertain."
)

GENERALIST_SYSTEM_PROMPT = (
    "You are a friendly and helpful assistant. Use the provided context to "
    "inform your answer when relevant, but feel free to draw on general "
    "knowledge as well. Keep your responses conversational, clear, and helpful."
)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_specialist_prompt(query: str, context: str) -> str:
    """Build a detailed RAG prompt for serious/hard topics."""
    return (
        "Use the following retrieved documents to answer the question. "
        "Provide a thorough, precise, and well-structured response. "
        "Reference specific details from the context to support your answer. "
        "If the context is insufficient, clearly state what you can and cannot "
        "determine from the available information.\n"
        "---------------------\n"
        f"Context:\n{context}\n"
        "---------------------\n"
        f"Question: {query}"
    )


def _build_generalist_prompt(query: str, context: str) -> str:
    """Build a conversational RAG prompt for general topics."""
    return (
        "Here is some relevant context that may help answer the question. "
        "Use it if helpful, and supplement with your general knowledge.\n"
        "---------------------\n"
        f"Context:\n{context}\n"
        "---------------------\n"
        f"Question: {query}"
    )


# ---------------------------------------------------------------------------
# Main chat handler with model routing
# ---------------------------------------------------------------------------

async def query_chat(db_client, llm_client, prompt: str, top_k: int) -> AsyncIterable:
    logging.info(f"Received chat query: {prompt}")

    # 1. Classify the query using the router model
    classification = await classify_query(llm_client, prompt)
    use_specialist = is_serious_topic(classification)

    logging.info(
        f"Routing decision — topic: {classification['topic']}, "
        f"confidence: {classification['confidence']:.2f}, "
        f"model: {'specialist' if use_specialist else 'generalist'}"
    )

    # 2. RAG retrieval (both paths)
    files: dict = query_collection(db_client, "file_mgt", query_text=prompt, top_k=top_k)
    logging.info(f"Retrieved {len(files['ids'][0])} relevant files from vector database")

    context_text = "\n\n".join([res for res in files['documents'][0]])

    # 3. Route to the appropriate model with tailored prompts
    if use_specialist:
        model = SPECIALIST_MODEL
        system_prompt = SPECIALIST_SYSTEM_PROMPT
        full_prompt = _build_specialist_prompt(prompt, context_text)
    else:
        model = GENERALIST_MODEL
        system_prompt = GENERALIST_SYSTEM_PROMPT
        full_prompt = _build_generalist_prompt(prompt, context_text)

    logging.info(f"Using model: {model}")

    # 4. Stream the response
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ],
        stream=True,
        max_tokens=1024,
    )
    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# TODO: evaluasi RAGAS

# TODO: LORA Fine-tuning