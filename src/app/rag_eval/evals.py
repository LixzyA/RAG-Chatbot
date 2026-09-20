"""Evaluate this application's RAG pipeline with Ragas.

Run from ``src/app`` so the evaluator uses the same dependencies and
configuration as the application:

    uv run python rag_eval/evals.py
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

EVAL_DIR = Path(__file__).parent
load_dotenv()

SRC_DIR = EVAL_DIR.parents[1]
sys.path.insert(0, str(SRC_DIR))

from app.config import settings

DEFAULT_DATASET = EVAL_DIR / "evals" / "datasets" / "test_dataset.csv"
DEFAULT_OUTPUT = EVAL_DIR / "evals" / "experiments" / "ragas_results.json"
RAGAS_MAX_TOKENS = 32_000
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[dict[str, str]]:
    """Read questions and optional reference answers from a CSV dataset."""
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Dataset must contain at least one row.")
    if any(not row.get("question", "").strip() for row in rows):
        raise ValueError("Every dataset row must have a non-empty 'question'.")
    return rows


async def run_rag(chain: Any, question: str, top_k: int) -> tuple[str, list[str]]:
    """Run one question and retain the exact context supplied to generation."""
    from app.models import RAGTraceBuilder

    trace = RAGTraceBuilder()
    answer_parts = [
        chunk async for chunk in chain.run(question, top_k=top_k, builder=trace)
    ]
    contexts = [item["content"] for item in trace.context_passed_to_llm]
    return "".join(answer_parts), contexts


async def score_response(
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
    scorers: dict[str, Any],
) -> dict[str, float]:
    """Score only metrics whose required Ragas inputs are available."""
    if not answer or not contexts:
        return {}

    scores = {
        "faithfulness": float(
            (
                await scorers["faithfulness"].ascore(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                )
            ).value
        )
    }
    if reference:
        scores.update(
            {
                "answer_correctness": float(
                    (
                        await scorers["answer_correctness"].ascore(
                            user_input=question,
                            response=answer,
                            reference=reference,
                        )
                    ).value
                ),
                "context_precision": float(
                    (
                        await scorers["context_precision"].ascore(
                            user_input=question,
                            reference=reference,
                            retrieved_contexts=contexts,
                        )
                    ).value
                ),
                "context_recall": float(
                    (
                        await scorers["context_recall"].ascore(
                            user_input=question,
                            reference=reference,
                            retrieved_contexts=contexts,
                        )
                    ).value
                ),
            }
        )
    return scores


def mean_scores(results: list[dict[str, Any]]) -> dict[str, float]:
    """Return the mean for every Ragas metric that produced a score."""
    values: dict[str, list[float]] = {}
    for result in results:
        for name, score in result.get("scores", {}).items():
            values.setdefault(name, []).append(score)
    return {
        name: round(sum(scores) / len(scores), 4) for name, scores in values.items()
    }


def no_context_retrieval_score(contexts: list[str]) -> float:
    """Score an intentional no-context case without asking the evaluator LLM."""
    return float(not contexts)


def build_scorers() -> dict[str, Any]:
    """Configure Ragas to judge with the same Hugging Face token as the app."""
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerCorrectness,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before running Ragas evals.")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    llm = llm_factory("z-ai/glm-5.3-flash", client=client, max_tokens=RAGAS_MAX_TOKENS)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_correctness": AnswerCorrectness(llm=llm, weights=[1.0, 0.0]),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def evaluate(dataset: list[dict[str, str]], top_k: int) -> dict[str, Any]:
    """Run the live RAG chain, then score its trace with Ragas."""
    from app.core.orchestration import RAGChain
    from app.core.retrieval import CrossEncoderReranker, VectorStore

    vector_store = VectorStore(
        collection_name=settings.chroma_collection_name,
        persist_path=str(settings.local_data_path),
    )
    reranker = CrossEncoderReranker()
    chain = RAGChain(
        vector_store,
        use_reranker=settings.reranker_enabled,
        reranker=reranker,
    )
    if settings.reranker_enabled:
        await asyncio.to_thread(reranker._load)

    scorers = build_scorers()
    results: list[dict[str, Any]] = []
    for row in dataset:
        question = row["question"].strip()
        reference = row.get("reference", "").strip()
        expects_no_context = row.get("expect_no_context", "").strip().lower() == "true"
        try:
            answer, contexts = await run_rag(chain, question, top_k)
            scores = (
                {"no_context_retrieval": no_context_retrieval_score(contexts)}
                if expects_no_context
                else await score_response(
                    question, answer, contexts, reference, scorers
                )
            )
        except Exception as error:
            logger.exception("Evaluation failed for question: %s", question)
            results.append({"question": question, "error": str(error)})
            continue

        results.append(
            {
                "question": question,
                "reference": reference or None,
                "expect_no_context": expects_no_context,
                "reference_chunk_id": row.get("reference_chunk_id", "").strip() or None,
                "reference_source": row.get("reference_source", "").strip() or None,
                "answer": answer,
                "contexts": contexts,
                "scores": scores,
            }
        )

    return {"results": results, "mean_scores": mean_scores(results)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline with Ragas."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    report = asyncio.run(evaluate(load_dataset(args.dataset), args.top_k))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["mean_scores"], indent=2))
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
