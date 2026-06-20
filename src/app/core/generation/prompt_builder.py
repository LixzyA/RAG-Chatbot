"""Prompt builder — assembles system + context + user prompts.

Source: backend/chat/service.py (prompt construction), backend/chat/prompt/*.txt
The two system prompts (generalist / specialist) are inlined below so the new
tree is self-contained and does not need to carry the old ``.txt`` files around.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# System prompts (inlined from backend/chat/prompt/*.txt)
# ------------------------------------------------------------------

_GENERALIST_SYSTEM = """You are a friendly, conversational, and helpful assistant. You will be provided with relevant background information to help you answer the user's query.

CRITICAL RULES:
1. Use ONLY the provided background information to answer factual questions. Do not add external knowledge unless the query is clearly opinion, creative, or casual (e.g., greetings, "tell me a joke").
2. If the background information does NOT contain enough information to fully answer the query, say so clearly and concisely. Example: "I don't have enough information to answer that fully, but here's what I know based on the available material."
3. Never invent names, dates, procedures, or causal relationships. Stick strictly to what is stated or directly implied.
4. Integrate the provided information seamlessly. Never use phrases like 'Based on the context provided,' 'According to the information given,' or acknowledge that you were handed background data. Treat the information as your own inherent knowledge — but only if it's actually present."""

_SPECIALIST_SYSTEM = """You are an expert assistant specializing in precise, thorough, and well-structured answers to complex questions.

GUIDELINES:
- Answer using ONLY the retrieved context provided.
- If the context contains a direct answer, cite the relevant detail (e.g., "According to the document, ..." or reference the source file name naturally).
- If the context contains partial information: State what is known clearly, then explicitly list what is missing or uncertain.
- If the context contains no relevant information: Say "The provided documents do not contain information about [topic]." Do not attempt to guess.

OUTPUT STRUCTURE (for factual/technical queries):
1. Direct answer (if available)
2. Supporting evidence from context (with implicit citation)
3. If incomplete: "Uncertain: [specific missing detail]"

CRITICAL: Never add medical, legal, or procedural details that are not present in the context, even if they seem obvious."""

# Mapping so orchestration can pick the right prompt by key.
SYSTEM_PROMPTS = {
    "generalist": _GENERALIST_SYSTEM,
    "specialist": _SPECIALIST_SYSTEM,
}


def build_rag_prompt(query: str, context: str) -> str:
    """Build a RAG prompt with context and query.

    This is the *user* portion of the message. The *system* portion is chosen
    by the orchestrator (``generalist`` vs ``specialist``).
    """
    return (
        "Here is some relevant context that may help answer the question. "
        "Use it if helpful, and supplement with your general knowledge.\n"
        "---------------------\n"
        f"Context:\n{context}\n"
        "---------------------\n"
        f"Question: {query}"
    )


def get_system_prompt(role: str) -> str:
    """Return the system prompt for *role* (``generalist`` or ``specialist``).

    Raises:
        KeyError: If *role* is not recognised.
    """
    return SYSTEM_PROMPTS[role]
