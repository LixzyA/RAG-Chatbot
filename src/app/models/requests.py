"""Pydantic request models for all public API endpoints.

Covers ingestion, retrieval, chat, and auth request bodies.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


ALLOWED_FILTER_KEYS: frozenset[str] = frozenset(
    {
        "file_type",
        "language",
        "file_name",
        "ner_model",
        "page",
        "source",
    }
)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., min_length=5, max_length=128)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


class ChatQueryRequest(BaseModel):
    prompt: str
    top_k: int = 10
    chat_id: str | None = None


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


class IngestRequest(BaseModel):
    file_path: str | None = Field(
        default=None, description="Optional absolute file path for local ingestion"
    )
    collection: str | None = Field(
        default=None, description="Target ChromaDB collection name"
    )
    chunk_size: int = Field(default=1024, ge=100, description="Token limit per chunk")
    chunk_overlap: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Fractional overlap between chunks"
    )


class IngestBatchRequest(BaseModel):
    file_paths: list[str] = Field(
        default_factory=list, description="List of files to ingest"
    )
    collection: str | None = Field(default=None)
    chunk_size: int = Field(default=1024, ge=100)
    chunk_overlap: float = Field(default=0.2, ge=0.0, le=1.0)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    use_hybrid: bool = Field(
        default=True, description="Use BM25 + vector ensemble vs. vector only"
    )
    use_reranker: bool = Field(
        default=True, description="Apply cross-encoder reranking"
    )
    collection: str | None = Field(
        default=None, description="Override default collection"
    )
    filter: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Metadata filter (curated allowlist). "
            f"Keys: {sorted(ALLOWED_FILTER_KEYS)}. "
            "Values: str | int | float | list[str]."
        ),
    )

    @field_validator("filter")
    @classmethod
    def _validate_filter(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return v
        for key, val in v.items():
            if key not in ALLOWED_FILTER_KEYS:
                raise ValueError(
                    f"Unknown filter key '{key}'. Allowed: {sorted(ALLOWED_FILTER_KEYS)}"
                )
            if isinstance(val, list):
                if not all(isinstance(x, str) for x in val):
                    raise ValueError(
                        f"List filter values for '{key}' must be list[str] (Chroma $in semantics)"
                    )
            elif not isinstance(val, (str, int, float)):
                raise ValueError(
                    f"Filter value for '{key}' must be str|int|float|list[str], got {type(val).__name__}"
                )
        return v


class RetrieveBatchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1, max_length=50)
    top_k: int = Field(default=10, ge=1, le=100)
    use_hybrid: bool = Field(default=True)
    use_reranker: bool = Field(default=True)
