"""Response parser — extracts text from streamed LLM chunks.

Source: scattered SSE / chunk parsing logic in backend/chat/service.py
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def parse_sse_chunk(chunk: object) -> str | None:
    """Extract text content from a single streamed LLM chunk.

    Handles the HuggingFace ``InferenceClient`` streaming format where each
    chunk is a ``ChatCompletionStreamOutput`` / OpenAI-compatible object.

    Args:
        chunk: A raw chunk yielded by the async generator.

    Returns:
        The text delta, or ``None`` if the chunk has no valid content.
    """
    if chunk is None:
        return None

    # OpenAI / HF compatible format
    if hasattr(chunk, "choices") and chunk.choices:
        delta = chunk.choices[0].delta
        return delta.content if delta and delta.content else None

    # Fallback for dict-like chunks (defensive, should not be needed with HF client)
    if isinstance(chunk, dict):
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            text = delta.get("content")
            if text:
                return text

    return None
