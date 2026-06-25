"""Prompt builder — assembles system + context + user prompts.

The single system prompt is inlined below so this module is self-contained
and does not need to carry the old ``.txt`` files around.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Generation system prompt (used by rag_chain.generate_stream)
# ------------------------------------------------------------------

_GENERATION_SYSTEM = """You are a friendly, conversational, and helpful assistant. You will be provided with relevant background information to help you answer the user's query.

CRITICAL RULES:
1. Use ONLY the provided background information to answer factual questions. Do not add external knowledge unless the query is clearly opinion, creative, or casual (e.g., greetings, "tell me a joke").
2. If the background information does NOT contain enough information to fully answer the query, say so clearly and concisely. Example: "I don't have enough information to answer that fully, but here's what I know based on the available material."
3. Never invent names, dates, procedures, or causal relationships. Stick strictly to what is stated or directly implied.
4. Integrate the provided information seamlessly. Never use phrases like 'Based on the context provided,' 'According to the information given,' or acknowledge that you were handed background data. Treat the information as your own inherent knowledge — but only if it's actually present."""


def build_rag_prompt(
    query: str,
    context: str,
    previous_query: str | None = None,
) -> str:
    """Build a RAG prompt with context and query.

    This is the *user* portion of the message. The *system* portion comes from
    :data:`_GENERATION_SYSTEM` (used by rag_chain.generate_stream).

    ``previous_query``, when provided, is injected between the retrieved
    context and the current question as a short "Previous user question"
    block. It gives the LLM conversational grounding for follow-up turns
    ("tell me more about that") without polluting retrieval — retrieval is
    driven by the current ``query`` only, by design.
    """
    history_block = ""
    if previous_query:
        history_block = (
            "---------------------\n"
            "Previous user question (for conversational context):\n"
            f"{previous_query}\n"
            "---------------------\n"
        )
    return (
        "Here is some relevant context that may help answer the question. "
        "Use it if helpful, and supplement with your general knowledge.\n"
        "---------------------\n"
        f"Context:\n{context}\n"
        f"{history_block}"
        "---------------------\n"
        f"Question: {query}"
    )


def get_generation_system_prompt() -> str:
    """Return the system prompt used by the generation model."""
    return _GENERATION_SYSTEM
