"""Orchestration — wires pipeline, retrieval, and generation into cohesive flows."""

from app.core.orchestration.feedback import EvaluationResult, SelfFeedbackLoop
from app.core.orchestration.metadata_enrichment import (
    compute_content_hash,
    enrich_chunks,
    extract_base_metadata,
)
from app.core.orchestration.ner_service import enrich_chunks_batch, enrich_chunks_route
from app.core.orchestration.query_processor import QueryProcessor
from app.core.orchestration.rag_chain import RAGChain

__all__ = [
    "compute_content_hash",
    "enrich_chunks",
    "enrich_chunks_batch",
    "enrich_chunks_route",
    "EvaluationResult",
    "extract_base_metadata",
    "QueryProcessor",
    "RAGChain",
    "SelfFeedbackLoop",
]
