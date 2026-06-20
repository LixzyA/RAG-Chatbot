"""Score fusion — merge multiple ranked lists into a single ranked list.

Implements Reciprocal Rank Fusion (RRF) used by the legacy ensemble retriever.
Source: backend/vectordb/core.py (hybrid-ensemble weighting via LangChain EnsembleRetriever).
"""
from __future__ import annotations

from collections import defaultdict

import logging

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Merge several ordered lists into one ranked list using RRF.

    Args:
        ranked_lists: Each inner list is an ordered result set (best first).
        weights: Optional per-list weight.  Must match ``len(ranked_lists)``.
        k: RRF constant (default 60).  Higher values dampen rank differences.
        top_k: Number of final results to return.

    Returns:
        ``[(item, score), ...]`` sorted by descending score.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    scores: dict[str, float] = defaultdict(float)

    for weight, results in zip(weights, ranked_lists):
        for rank, item in enumerate(results, start=1):
            scores[item] += weight * (1.0 / (k + rank))

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_k]
