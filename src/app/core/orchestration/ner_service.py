"""NER orchestration — batch entrypoint.

``enrich_chunks_batch`` — caller picks language, one model for all chunks.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.core.pipeline.ner_extractor import NERExtractor

logger = logging.getLogger(__name__)

# Lazy cache: language -> NERExtractor
_extractors: dict[str, NERExtractor] = {}


def _get_extractor(language: str) -> NERExtractor:
    if language not in _extractors:
        _extractors[language] = NERExtractor(language, device=settings.ner_device)
    return _extractors[language]


def _entities_to_metadata(
    entities: list[dict[str, Any]],
    model_id: str,
    language: str,
) -> dict[str, Any]:
    """Build the NER metadata fragment, or empty dict if no entities."""
    if not entities:
        return {"language": language, "ner_model": model_id}
    return {
        "language": language,
        "ner_model": model_id,
        "entities": entities,
        "entity_types": sorted({e["label"] for e in entities}),
    }


def enrich_chunks_batch(chunks: list[Document], language: str) -> None:
    """Batch path: caller declares language, one extractor for all chunks.

    Args:
        chunks: list of Document objects to enrich (mutated in place).
        language: ``"en"`` or ``"id"``.
    """
    if language not in ("en", "id"):
        logger.warning("Unsupported batch language: %s — skipping NER", language)
        return

    extractor = _get_extractor(language)

    # --- batch-extract all texts at once ---
    texts = [c.page_content for c in chunks]
    batch_results = extractor.extract_batch(texts, batch_size=settings.ner_batch_size)

    for chunk, entities in zip(chunks, batch_results, strict=True):
        ner_meta = _entities_to_metadata(entities, extractor.model_id, language)
        chunk.metadata.update(ner_meta)
