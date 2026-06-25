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


async def healthcheck() -> dict:
    """Lightweight probe: issue a single-turn chat to the generation model."""
    client = get_llm_client()
    result: dict = {"healthy": False, "error": None}
    try:
        response = await client.chat.completions.create(
            model=settings.generation_model,
            messages=[{"role": "user", "content": "healthcheck"}],
            temperature=0.1,
        )
        if response:
            result["healthy"] = True
        else:
            result["error"] = "Empty response from LLM"
    except Exception as exc:
        result["error"] = str(exc)
    return result


def reset_client() -> None:
    """Force re-creation on next access (useful in tests)."""
    global _llm_client  # noqa: PLW0603
    _llm_client = None
