"""Cross-encoder reranker — re-order retrieved chunks by relevance to the query.

Source: backend/vectordb/reranker.py (FlagReranker wrapper).
"""

import gc
import time
import torch
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document


from app.config import settings
from app.utils.exceptions import RerankerException
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder reranker using ``FlagEmbedding``."""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    @property
    def model_name(self) -> str:
        return settings.reranker_model

    @property
    def enabled(self) -> bool:
        return settings.reranker_enabled

    @property
    def device(self) -> str:
        return settings.reranker_device

    def _load(self):
        if self._loaded:
            return
        try:
            logger.info("Loading reranker..")
            is_gpu = self.device != "cpu"
            self._model = FlagReranker(
                self.model_name,
                use_fp16=is_gpu,
                device=self.device,
            )
            logger.debug("Reranker init with %s", self.device)

            if is_gpu:
                logger.debug("Reranker warming up...")
                self._model.compute_score(
                    [("warmup query", "warmup document")],
                    normalize=True,
                    batch_size=1,
                    max_length=32,
                )
                logger.debug("Warm-up complete")

            self._loaded = True
            logger.info("Reranker loaded.")
        except Exception as e:
            logger.exception(f"Encountered unexpected error during loading: {e}")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Return the top-``top_k`` ``(document, score)`` pairs ordered by cross-encoder score.

        A score of ``1.0`` is emitted for bypass paths (reranker disabled, no
        documents, or model not loaded) so downstream threshold filters have
        a well-defined value. Active reranking uses FlagReranker's normalised
        score (range ≈ [0, 1]).
        """
        # Bypass paths — top-K by input order, score 1.0 (passes any sane threshold).
        if not self.enabled or not documents:
            return [(doc, 1.0) for doc in documents[:top_k]]

        self._load()
        if self._model is None:
            raise RerankerException

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.compute_score(
            pairs,
            normalize=True,
            batch_size=32,
            max_length=512,
        )

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        result: list[tuple[Document, float]] = []
        for doc, score in scored[:top_k]:
            doc.metadata["rerank_score"] = score  # type: ignore[index]
            result.append((doc, float(score)))
        return result

    def healthcheck(self) -> dict:
        """Return a health-check result dict."""
        result: dict = {
            "status": "unhealthy",
            "model": self.model_name,
            "device": self.device,
            "enabled": self.enabled,
            "loaded": self._loaded,
        }
        if not self.enabled:
            result["status"] = "disabled"
            return result

        if not self._loaded or self._model is None:
            result["error"] = "Model not loaded"
            return result

        try:
            start = time.perf_counter()
            score = self._model.compute_score(
                [("healthcheck", "probe")],
                normalize=True,
                batch_size=1,
                max_length=16,
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            if isinstance(score, list):
                if len(score) > 0:
                    score = score[0]  # Extract the first (and only) score
                else:
                    result["error"] = "Empty score list returned"
                    return result

            if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                result["error"] = f"Invalid score returned: {score}"
                return result

            result["status"] = "healthy"
            result["latency_ms"] = latency_ms
            result["warm"] = True

        except Exception as exc:
            logger.error("Reranker healthcheck failed: %s", exc, exc_info=True)
            result["error"] = str(exc)

        return result

    def close(self) -> None:
        self._model = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
