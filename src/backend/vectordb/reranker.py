import os
from typing import List, Optional, Annotated, Dict, Any
import time

from fastapi import Request, Depends
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document
from exception import RerankerException
from logger import configure_logging
from dotenv import load_dotenv
load_dotenv()
logger = configure_logging(__name__)

class Reranker:
    """Cross-encoder reranker backed by FlagReranker.

    The model is downloaded on first use and loaded lazily inside
    """

    def __init__(self):
        self._model = None
        self._loaded = False

    @property
    def model_name(self) -> str:
        return os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    @property
    def enabled(self) -> bool:
        return os.getenv("RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")

    @property
    def device(self) -> str:
        return os.getenv("RERANKER_DEVICE", "cpu")

    def _load(self) -> None:
        if self._loaded:
            return
        is_gpu = self.device != "cpu"
        self._model = FlagReranker(
            self.model_name,
            use_fp16=is_gpu,
            device=self.device
        )
        logger.debug(f"Reranker init with {self.device}")
        
        if is_gpu:
            logger.debug("Reranker warming up...")
            self._model.compute_score(
                [("warmup query", "warmup document")], 
                normalize=True, 
                batch_size=1, 
                max_length=32
            )
            logger.debug("✅ Warm-up complete")
        
        self._loaded = True

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """Rerank *documents* by relevance to *query*.

        Parameters
        ----------
        query:
            The user's search query.
        documents:
            Candidate documents from a first-stage retriever.
        top_k:
            Number of top documents to return. Returns all if ``None``.

        Returns
        -------
        Documents sorted by cross-encoder score (most relevant first).
        Returns the original list unchanged when the reranker is disabled
        or *documents* is empty.
        """
        if not self.enabled or not documents:
            return documents[:top_k] if top_k else documents

        self._load()
        if self._model is None:
            raise RerankerException

        # OPTIMIZED: Added batch_size and max_length
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.compute_score(
            pairs, 
            normalize=True, 
            batch_size=32, 
            max_length=512
        )

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        result = []
        limit = top_k if top_k else len(scored)
        for doc, score in scored[:limit]:
            doc.metadata["rerank_score"] = score
            result.append(doc)
        return result
    
    def healthcheck(self):
        result: Dict[str, Any] = {
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
            # Lightweight probe: single pair, minimal tokens
            score = self._model.compute_score(
                [("healthcheck", "probe")],
                normalize=True,
                batch_size=1,
                max_length=16,
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            # Validate output is sane (normalized score should be 0-1)
            if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                result["error"] = f"Invalid score returned: {score}"
                return result

            result["status"] = "healthy"
            result["latency_ms"] = latency_ms
            result["warm"] = True  # Distinguishes from cold-start probes

        except Exception as e:
            logger.error("Reranker healthcheck failed: %s", e, exc_info=True)
            result["error"] = str(e)

        return result



def get_reranker(request: Request) -> Reranker:
    """Get the reranker from app state."""
    return request.app.state.reranker


RerankerClient = Annotated[Reranker, Depends(get_reranker)]
