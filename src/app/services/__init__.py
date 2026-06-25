"""External service integrations.

Submodules
----------
- ``auth_service``      — password hashing, JWT, user CRUD.
- ``chat_history_service`` — SQLite-backed chat session / message CRUD.
- ``vector_db``         — ``VectorStore`` + ``CrossEncoderReranker`` singletons.
- ``rag_chain``         — ``RAGChain`` singleton orchestrator.
- ``cache``             — In-memory TTL cache (``TTLCache``).
- ``storage``           — Local filesystem I/O (``StorageService``).
"""

from app.services.auth_service import (
    create_access_token,
    decode_access_token,
    hash_password,
    login_user,
    register_user,
    verify_password,
)
from app.services.cache import TTLCache, default_cache
from app.services.chat_history_service import (
    add_message,
    create_or_get_history,
    delete_history,
    get_history,
    list_histories,
    update_title,
)
from app.services.rag_chain import get_rag_chain, reset_rag_chain
from app.services.storage import StorageService
from app.services.vector_db import (
    get_reranker,
    get_vector_store,
    reset_store,
)

__all__ = [
    # auth
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "login_user",
    "register_user",
    "verify_password",
    # chat history
    "add_message",
    "create_or_get_history",
    "delete_history",
    "get_history",
    "list_histories",
    "update_title",
    # cache
    "TTLCache",
    "default_cache",
    # rag chain
    "get_rag_chain",
    "reset_rag_chain",
    # storage
    "StorageService",
    # vector db
    "get_reranker",
    "get_vector_store",
    "reset_store",
]
