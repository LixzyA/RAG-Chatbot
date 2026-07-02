"""Transformer-based NER extraction, one pipeline per model, lazy-loaded.

Two models: ``dslim/bert-base-NER`` (EN, CoNLL-2003) and
``cahya/bert-base-indonesian-NER`` (ID, token-level BIO).

Thread-safety note: ``transformers`` pipeline is mostly thread-safe for
inference (no graph-building after warmup). Guard lazy-init with a lock so
the first concurrent call from batch code doesn't race.
"""

import logging
import threading
from typing import Any, cast

from transformers.pipelines import pipeline

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
        self._pipeline = None
        self._lock = threading.Lock()

    def _split_for_ner(
        self, chunk: str, max_chars: int = 350, overlap: int = 70
    ) -> list[tuple[int, str]]:
        """
        Split chunks into smaller chunks because the ner has smaller max_token limit.
        """
        if len(chunk) <= max_chars:
            return [(0, chunk)]

        chunks = []
        start = 0
        n = len(chunk)
        while start < n:
            end = min(start + max_chars, n)

            if end < n:
                search_start = max(start, end - 50)
                last_space = chunk.rfind(" ", search_start, end)
                if last_space > start:
                    end = last_space

            chunks.append((start, chunk[start:end]))

            if end >= n:
                break

            start = end - overlap
            if start <= chunks[-1][0]:
                start = end

        return chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Run NER on *text*, return list of entity dicts.

        Each dict: ``{"text": …, "label": …, "start": int, "end": int}``.
        Labels from CoNLL-2003: ``PER``, ``ORG``, ``LOC``, ``MISC``.
        Indonesian model uses BIO tags — prefix (``B-``/``I-``) is stripped
        and consecutive same-label tokens are merged into one entity span.

        Sub-chunks via ``_split_for_ner`` so BERT's 512-token positional
        table isn't exceeded (chunker emits 1024-char pieces that tokenise
        past 512 for many scripts, e.g. Chinese).
        """
        pipe = self._get_pipeline()
        out: list[dict[str, Any]] = []
        for offset, sub in self._split_for_ner(text):
            for ent in cast(list[dict[str, Any]], pipe(sub)):
                out.append(
                    {
                        "text": ent["word"],
                        "label": ent["entity_group"],
                        "start": int(ent["start"]) + offset,
                        "end": int(ent["end"]) + offset,
                    }
                )
        return out

    def extract_batch(self, texts: list[str], batch_size: int = 32) -> list[list[dict]]:
        """
        Extracts entities from a batch of texts.
        Automatically sub-chunks long texts to prevent exceeding the model's max_length.
        """
        if not texts:
            return []

        # 1. Sub-chunk all texts and keep track of original indices and character offsets
        sub_chunks = []  # List of (original_index, start_offset, chunk_text)
        for orig_idx, text in enumerate(texts):
            if not text:
                continue
            for offset, chunk in self._split_for_ner(text):
                sub_chunks.append((orig_idx, offset, chunk))

        if not sub_chunks:
            return [[] for _ in texts]

        pipe_inputs = [item[2] for item in sub_chunks]

        pipe = self._get_pipeline()
        results = pipe(pipe_inputs, batch_size=batch_size)

        # Ensure results is a list of lists (HF pipeline sometimes returns a flat list for single inputs)
        if len(pipe_inputs) == 1 and results and isinstance(results[0], dict):
            results = [results]

        # 3. Map the entities back to the original texts and adjust character offsets
        final_results: list[list[dict]] = [[] for _ in texts]

        for (orig_idx, offset, _), entities in zip(sub_chunks, results):
            if not entities:
                continue
            for ent in entities:
                # Cast torch/numpy scalars → native Python so downstream
                # json.dumps + Chroma metadata accept the values.
                # transformers NER with aggregation_strategy="average" emits
                # ``score`` as np.float32 and ``start``/``end`` as np.int64.
                ent_type = ent.get("entity_group", ent.get("entity", "UNKNOWN"))
                ent_copy = {
                    "text": ent.get("word", ""),
                    "label": ent_type,
                    "start": int(ent.get("start", 0)) + offset,
                    "end": int(ent.get("end", 0)) + offset,
                }
                final_results[orig_idx].append(ent_copy)

        # 4. Deduplicate entities that might have been detected in the overlapping regions
        for i in range(len(final_results)):
            seen = set()
            unique_ents = []
            for ent in final_results[i]:
                # Create a unique key based on entity type and exact character boundaries
                ent_type = ent.get("entity_group", ent.get("entity", "UNKNOWN"))
                key = (ent_type, ent["start"], ent["end"])
                if key not in seen:
                    seen.add(key)
                    unique_ents.append(ent)
            final_results[i] = unique_ents

        return final_results

    def supported_languages(self) -> frozenset[str]:
        return frozenset(_MODEL_REGISTRY)

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:  # double-checked
                    logger.info(
                        "Loading NER model: %s (device=%s)", self.model_id, self.device
                    )
                    self._pipeline = pipeline(
                        "ner",
                        model=self.model_id,
                        tokenizer=self.model_id,
                        device=self.device,
                        aggregation_strategy="average",
                    )
        return self._pipeline
