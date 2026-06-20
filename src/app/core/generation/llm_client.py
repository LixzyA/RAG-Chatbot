"""Async LLM client — wraps HuggingFace Inference API (or compatible).

Source: backend/chat/core.py (AsyncInferenceClient singleton + healthcheck).
"""
from __future__ import annotations

import logging
from huggingface_hub import AsyncInferenceClient

from app.config import settings
logger = logging.getLogger(__name__)

# Global singleton — created on first import.
_llm_client: AsyncInferenceClient | None = None


def get_llm_client() -> AsyncInferenceClient:
    """Return the global :class:`AsyncInferenceClient` singleton."""
    global _llm_client  # noqa: PLW0603
    if _llm_client is None:
        _llm_client = AsyncInferenceClient()
        logger.info("AsyncInferenceClient initialised")
    return _llm_client


async def healthcheck() -> bool:
    """Lightweight probe: issue a single-turn chat to the generalist model."""
    client = get_llm_client()
    try:
        response = await client.chat.completions.create(
            model=settings.generalist_model,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.1,
        )
        return bool(response)
    except Exception:
        return False


def reset_client() -> None:
    """Force re-creation on next access (useful in tests)."""
    global _llm_client  # noqa: PLW0603
    _llm_client = None
