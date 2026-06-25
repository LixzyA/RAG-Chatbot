"""RAG Chatbot Application.

A modular FastAPI backend organized by data step (pipeline, retrieval, generation, orchestration),
with thin API routes and thick core business logic.
"""

from .config import settings

__all__ = ["settings"]
