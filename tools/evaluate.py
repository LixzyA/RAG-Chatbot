#!/usr/bin/env python3
"""Offline RAG evaluation — run a test set through the full pipeline and score results.

Usage:
    uv run python tools/evaluate.py tests/eval_queries.json [--output report.json]

The input JSON file must contain a ``queries`` array of objects:

    {
      "queries": [
        {
          "query": "What is the refund policy?",
          "expected_answer": "Refunds are processed within 14 days.",
          "relevant_doc_ids": ["doc_abc", "doc_def"]
        }
      ]
    }

``expected_answer`` and ``relevant_doc_ids`` are optional. When provided they add
semantic-answer-similarity and retrieval-precision/recall to the report.

Outputs a JSON report with per-query metrics and aggregates to ``stdout`` (or
``--output`` if given). Also prints a human-readable summary table.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: add the app directory so we can import the application packages.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from app.config import settings
from app.core.orchestration import RAGChain
from app.core.retrieval import VectorStore
from app.core.retrieval import CrossEncoderReranker
from app.models import RAGTraceBuilder
from app.models import EvaluationResult
from app.core.orchestration import SelfFeedbackLoop
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_queries(path: Path) -> list[dict[str, Any]]:
    """Load eval queries from a JSON file, returning only those that pass validation."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries: list[dict[str, Any]] = raw.get("queries", [])

    valid: list[dict[str, Any]] = []
    for i, q in enumerate(queries):
        if not isinstance(q.get("query"), str) or not q["query"].strip():
            logger.warning("Query %d missing 'query' field — skipping", i)
            continue
        valid.append(q)
    return valid


def _recall_at_k(relevant_ids: set[str], retrieved_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(relevant_ids & set(list(retrieved_ids)[:k])) / len(relevant_ids)


def _precision_at_k(relevant_ids: set[str], retrieved_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(list(retrieved_ids)[:k])
    return len(relevant_ids & top_k) / min(k, len(top_k)) if top_k else 0.0


def _mrr(relevant_ids: set[str], ranked_ids: list[str]) -> float:
    if not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _semantic_overlap(a: str, b: str) -> float:
    """Token-overlap score between two strings (0.0–1.0).

    A cheap fallback when an embedding-based semantic similarity model isn't
    available. Not as accurate as a dedicated entailment/NLI scorer, but useful
    for spotting grossly-wrong answers in offline reports.
    """
    if not a or not b:
        return 0.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / max(len(tokens_a), len(tokens_b))


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


async def run_eval(
    queries: list[dict[str, Any]],
    *,
    top_k: int,
    threshold: float | None = None,
    self_feedback: bool = False,
) -> dict[str, Any]:
    """Execute all queries through the RAG chain and return a structured report."""

    # --- Boot singletons ---
    vs = VectorStore(
        collection_name=settings.chroma_collection_name,
        persist_path=str(settings.local_data_path),
    )
    reranker = CrossEncoderReranker()
    chain = RAGChain(vs, use_reranker=settings.reranker_enabled, reranker=reranker)

    # Warm the reranker
    if settings.reranker_enabled:
        logger.info("Loading reranker...")
        await asyncio.to_thread(reranker._load)

    per_query: list[dict[str, Any]] = []
    aggregates: dict[str, list[float]] = {
        "total_latency_ms": [],
        "retrieval_latency_ms": [],
        "rerank_latency_ms": [],
        "llm_latency_ms": [],
        "input_tokens": [],
        "output_tokens": [],
        "faithfulness": [],
        "relevance": [],
        "completeness": [],
        "composite_score": [],
        "hallucination_rate": [],
        "precision@5": [],
        "recall@5": [],
        "mrr": [],
        "semantic_overlap": [],
        "feedback_iterations": [],
    }

    total = len(queries)
    for idx, q in enumerate(queries):
        query_text: str = q["query"].strip()
        expected: str | None = q.get("expected_answer")
        relevant_ids = set(q.get("relevant_doc_ids") or [])

        logger.info("[%d/%d] Evaluating: %s", idx + 1, total, query_text[:80])

        builder = RAGTraceBuilder()
        start = time.perf_counter()

        # Collect the full answer (non-streaming accumulation).
        chunks: list[str] = []
        try:
            async for chunk in chain.run(
                query_text,
                top_k=top_k,
                threshold=threshold,
                builder=builder,
                self_feedback_enabled=self_feedback,
            ):
                chunks.append(chunk)
        except Exception as exc:
            logger.error("Chain failed for query '%s': %s", query_text[:60], exc)
            per_query.append({"query": query_text, "answer": "", "error": str(exc)})
            continue

        answer = "".join(chunks)
        total_ms = (time.perf_counter() - start) * 1000

        # --- Critic evaluation (always run, fast router model) ---
        docs_final = [
            Document(
                id=d["id"],
                page_content=d["content"],
                metadata=d.get("metadata", {}),
            )
            for d in builder.context_passed_to_llm
        ]
        eval_result: EvaluationResult | None = None
        if answer and docs_final:
            try:
                eval_result = await SelfFeedbackLoop().evaluate(
                    answer, docs_final, query_text
                )
            except Exception as exc:
                logger.warning("Critic eval failed for '%s': %s", query_text[:60], exc)

        # --- Retrieval metrics ---
        retrieved_ids = {d["id"] for d in builder.retrieved_chunks}
        ranked_ids = [d["id"] for d in builder.reranked_chunks]

        p_at_5 = _precision_at_k(relevant_ids, retrieved_ids, 5)
        r_at_5 = _recall_at_k(relevant_ids, retrieved_ids, 5)
        mrr_val = _mrr(relevant_ids, ranked_ids)
        overlap = _semantic_overlap(answer, expected) if expected else None

        entry: dict[str, Any] = {
            "query": query_text,
            "answer": answer,
            "total_latency_ms": round(total_ms, 1),
            "retrieval_latency_ms": (
                round(builder.retrieval_latency_ms, 1)
                if builder.retrieval_latency_ms is not None
                else None
            ),
            "rerank_latency_ms": (
                round(builder.rerank_latency_ms, 1)
                if builder.rerank_latency_ms is not None
                else None
            ),
            "llm_latency_ms": (
                round(builder.llm_latency_ms, 1)
                if builder.llm_latency_ms is not None
                else None
            ),
            "input_tokens": builder.input_tokens,
            "output_tokens": builder.output_tokens,
            "retrieved_doc_count": len(retrieved_ids),
            "reranked_doc_count": len(ranked_ids),
        }

        if eval_result is not None:
            entry.update(
                {
                    "faithfulness": eval_result.faithfulness,
                    "relevance": eval_result.relevance,
                    "completeness": eval_result.completeness,
                    "composite_score": eval_result.composite_score,
                    "hallucination_detected": eval_result.hallucination_detected,
                    "critique": eval_result.critique,
                    "missing_from_context": eval_result.missing_from_context,
                }
            )

        if expected:
            entry["expected_answer"] = expected
        if relevant_ids:
            entry["relevant_doc_ids"] = sorted(relevant_ids)
            entry["precision@5"] = round(p_at_5, 4)
            entry["recall@5"] = round(r_at_5, 4)
            entry["mrr"] = round(mrr_val, 4)
        if overlap is not None:
            entry["semantic_overlap"] = round(overlap, 4)

        if self_feedback:
            entry["feedback_iterations"] = builder.feedback_iterations
            entry["feedback_scores"] = builder.feedback_scores
            entry["feedback_final_score"] = builder.feedback_final_score

        if builder.feedback_scores:
            # Use last iteration's scores for aggregate.
            last = builder.feedback_scores[-1]
        elif eval_result is not None:
            last = {
                "faithfulness": eval_result.faithfulness,
                "relevance": eval_result.relevance,
                "completeness": eval_result.completeness,
            }
        else:
            last = {}

        per_query.append(entry)

        # Collect aggregates.
        aggregates["total_latency_ms"].append(total_ms)
        if builder.retrieval_latency_ms is not None:
            aggregates["retrieval_latency_ms"].append(builder.retrieval_latency_ms)
        if builder.rerank_latency_ms is not None:
            aggregates["rerank_latency_ms"].append(builder.rerank_latency_ms)
        if builder.llm_latency_ms is not None:
            aggregates["llm_latency_ms"].append(builder.llm_latency_ms)
        if builder.input_tokens is not None:
            aggregates["input_tokens"].append(builder.input_tokens)
        if builder.output_tokens is not None:
            aggregates["output_tokens"].append(builder.output_tokens)
        if last.get("faithfulness") is not None:
            aggregates["faithfulness"].append(float(last["faithfulness"]))
        if last.get("relevance") is not None:
            aggregates["relevance"].append(float(last["relevance"]))
        if last.get("completeness") is not None:
            aggregates["completeness"].append(float(last["completeness"]))
        if eval_result is not None:
            aggregates["composite_score"].append(eval_result.composite_score)
            aggregates["hallucination_rate"].append(
                1.0 if eval_result.hallucination_detected else 0.0
            )
        if relevant_ids:
            aggregates["precision@5"].append(p_at_5)
            aggregates["recall@5"].append(r_at_5)
            aggregates["mrr"].append(mrr_val)
        if overlap is not None:
            aggregates["semantic_overlap"].append(overlap)
        if self_feedback:
            aggregates["feedback_iterations"].append(builder.feedback_iterations)

    # --- Build aggregate summary ---
    def _agg(lst: list[float]) -> dict[str, float]:
        if not lst:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(lst),
            "mean": round(sum(lst) / len(lst), 3),
            "min": round(min(lst), 3),
            "max": round(max(lst), 3),
        }

    summary = {key: _agg(vals) for key, vals in aggregates.items()}

    return {
        "config": {
            "top_k": top_k,
            "threshold": threshold or settings.rag_min_relevance,
            "self_feedback": self_feedback,
            "generation_model": settings.generation_model,
            "reranker_model": settings.reranker_model,
            "reranker_enabled": settings.reranker_enabled,
            "total_queries": total,
        },
        "queries": per_query,
        "aggregates": summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(report: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    cfg = report["config"]
    agg = report["aggregates"]

    print()
    print("=" * 60)
    print("  RAG Evaluation Report")
    print("=" * 60)
    print(f"  Queries evaluated : {cfg['total_queries']}")
    print(f"  Top-k             : {cfg['top_k']}")
    print(f"  Threshold         : {cfg['threshold']}")
    print(f"  Self-feedback     : {cfg['self_feedback']}")
    print(f"  Generation model  : {cfg['generation_model']}")
    print(
        f"  Reranker          : {cfg['reranker_model']} (enabled={cfg['reranker_enabled']})"
    )
    print()

    rows: list[tuple[str, str]] = [
        ("Total latency (ms)", _fmt_agg(agg.get("total_latency_ms"))),
        ("Retrieval latency (ms)", _fmt_agg(agg.get("retrieval_latency_ms"))),
        ("Rerank latency (ms)", _fmt_agg(agg.get("rerank_latency_ms"))),
        ("LLM latency (ms)", _fmt_agg(agg.get("llm_latency_ms"))),
        ("Input tokens", _fmt_agg(agg.get("input_tokens"))),
        ("Output tokens", _fmt_agg(agg.get("output_tokens"))),
    ]
    if agg.get("faithfulness", {}).get("count", 0) > 0:
        rows.append(("Faithfulness", _fmt_agg_score(agg["faithfulness"])))
        rows.append(("Relevance", _fmt_agg_score(agg["relevance"])))
        rows.append(("Completeness", _fmt_agg_score(agg["completeness"])))
        rows.append(("Composite score", _fmt_agg_score(agg["composite_score"])))
        rows.append(("Hallucination rate", _fmt_agg_pct(agg["hallucination_rate"])))
    if agg.get("precision@5", {}).get("count", 0) > 0:
        rows.append(("Precision@5", _fmt_agg_score(agg["precision@5"])))
        rows.append(("Recall@5", _fmt_agg_score(agg["recall@5"])))
        rows.append(("MRR", _fmt_agg_score(agg["mrr"])))
    if agg.get("semantic_overlap", {}).get("count", 0) > 0:
        rows.append(("Semantic overlap", _fmt_agg_score(agg["semantic_overlap"])))
    if agg.get("feedback_iterations", {}).get("count", 0) > 0:
        rows.append(("Feedback iterations", _fmt_agg(agg["feedback_iterations"])))

    label_w = max(len(r[0]) for r in rows)
    for label, val in rows:
        print(f"  {label:<{label_w}} : {val}")
    print()


def _fmt_agg(d: dict[str, Any] | None) -> str:
    if not d or d.get("count", 0) == 0:
        return "N/A"
    return f"mean={d['mean']:.1f}  min={d['min']:.1f}  max={d['max']:.1f}  (n={d['count']})"


def _fmt_agg_score(d: dict[str, Any] | None) -> str:
    if not d or d.get("count", 0) == 0:
        return "N/A"
    return f"mean={d['mean']:.3f}  min={d['min']:.3f}  max={d['max']:.3f}  (n={d['count']})"


def _fmt_agg_pct(d: dict[str, Any] | None) -> str:
    if not d or d.get("count", 0) == 0:
        return "N/A"
    pct = d["mean"] * 100
    return f"{pct:.1f}%  (n={d['count']})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline against a test set of queries."
    )
    parser.add_argument(
        "queries_file",
        type=Path,
        help="Path to JSON file with test queries (array under 'queries' key).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write JSON report to this file (default: stdout only).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of documents to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Relevance threshold (default: settings.rag_min_relevance = {settings.rag_min_relevance}).",
    )
    parser.add_argument(
        "--self-feedback",
        action="store_true",
        default=False,
        help="Enable self-feedback loop (re-retrieval on low-quality answers).",
    )
    args = parser.parse_args()

    if not args.queries_file.exists():
        print(f"Error: {args.queries_file} not found", file=sys.stderr)
        sys.exit(1)

    queries = _load_queries(args.queries_file)
    if not queries:
        print("Error: no valid queries found in the input file.", file=sys.stderr)
        sys.exit(1)

    report = asyncio.run(
        run_eval(
            queries,
            top_k=args.top_k,
            threshold=args.threshold,
            self_feedback=args.self_feedback,
        )
    )

    _print_table(report)

    if args.output:
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"Full report written to {args.output}")


if __name__ == "__main__":
    main()
