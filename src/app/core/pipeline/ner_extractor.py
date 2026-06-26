"""Transformer-based NER extraction, one pipeline per model, lazy-loaded.

Two models: ``dslim/bert-base-NER`` (EN, CoNLL-2003) and
``cahya/bert-base-indonesian-NER`` (ID, token-level BIO).

Thread-safety note: ``transformers`` pipeline is mostly thread-safe for
inference (no graph-building after warmup). Guard lazy-init with a lock so
the first concurrent call from batch code doesn't race.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from transformers import pipeline, Pipeline  # pyright: ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, str] = {
    "en": "dslim/bert-base-NER",
    "id": "cahya/bert-base-indonesian-NER",
}


class NERExtractor:
    """Lazy-per-model NER pipeline. Load once, reuse for all chunks.

    Usage::
        en_ner = NERExtractor("en", device="cpu")
        entities = en_ner.extract("Felix works at Google in Jakarta")
    """

    def __init__(self, language: str, *, device: str | int = "cpu") -> None:
        if language not in _MODEL_REGISTRY:
            msg = f"Unsupported NER language: {language!r} (choose from {set(_MODEL_REGISTRY)})"
            raise ValueError(msg)
        self.language = language
        self.model_id = _MODEL_REGISTRY[language]
        self.device = device
        self._pipeline: Pipeline | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Run NER on *text*, return list of entity dicts.

        Each dict: ``{"text": …, "label": …, "start": int, "end": int}``.
        Labels from CoNLL-2003: ``PER``, ``ORG``, ``LOC``, ``MISC``.
        Indonesian model uses BIO tags — prefix (``B-``/``I-``) is stripped
        and consecutive same-label tokens are merged into one entity span.
        """
        pipe = self._get_pipeline()
        raw = pipe(text, aggregation_strategy="simple")  # type: ignore[arg-type]
        out: list[dict[str, Any]] = []
        for ent in raw:
            out.append({
                "text": ent["word"],
                "label": ent["entity_group"],
                "start": ent["start"],
                "end": ent["end"],
            })
        return out

    def extract_batch(
        self, texts: list[str], *, batch_size: int = 32
    ) -> list[list[dict[str, Any]]]:
        """Run NER over a batch of texts. Returns one entity list per input."""
        pipe = self._get_pipeline()
        results = pipe(texts, aggregation_strategy="simple", batch_size=batch_size)
        return [
            [{"text": e["word"], "label": e["entity_group"], "start": e["start"], "end": e["end"]}
             for e in batch]
            for batch in results
        ]

    def supported_languages(self) -> frozenset[str]:
        return frozenset(_MODEL_REGISTRY)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:  # double-checked
                    logger.info("Loading NER model: %s (device=%s)", self.model_id, self.device)
                    self._pipeline = pipeline(
                        "ner",
                        model=self.model_id,
                        tokenizer=self.model_id,
                        device=self.device,
                    )
        return self._pipeline
