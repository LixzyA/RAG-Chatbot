"""Shared domain models.

Unified `Document` and related types used across pipeline, retrieval, and generation.
"""
from typing import Any

from pydantic import BaseModel, ConfigDict


class Document(BaseModel):
    """A single chunk / document in the vector store.

    This is the lingua-franca type shared by:
    - ``pipeline/chunker.py`` (produces chunks)
    - ``retrieval/vector_store.py`` (stores / retrieves chunks)
    - ``retrieval/reranker.py`` (re-scores chunks)
    - ``generation/prompt_builder.py`` (formats chunks into context)
    """

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = {}

    # Pydantic v2: allow extra keys inside `metadata` without declaring them.
    model_config = ConfigDict(extra="ignore")

    @property
    def page_number(self) -> int | None:
        return self.metadata.get("page")

    @property
    def source(self) -> str | None:
        return self.metadata.get("source")

    def __repr__(self) -> str:
        snippet = self.content[:60] + ("..." if len(self.content) > 60 else "")
        extra = f" [{self.source}]" if self.source else ""
        return f"<Document id={self.id}{extra}: {snippet}>"
