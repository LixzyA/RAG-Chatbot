"""Self-feedback loop — evaluate, critique, and refine RAG answers."""

import json
import logging
import re
from langchain_core.documents import Document

from app.config import settings
from app.core.generation.llm_client import get_llm_client
from app.models import EvaluationResult

logger = logging.getLogger(__name__)


_CRITIC_PROMPT = """You are an evaluator for a RAG (Retrieval-Augmented Generation) system.
Your job is to critique an AI-generated answer against the provided context and the user's query.

CONTEXT (retrieved documents):
{context}

USER QUERY:
{query}

GENERATED ANSWER:
{answer}

Evaluate the answer on SIX dimensions (each 0.0-1.0):
1. faithfulness - Are ALL claims in the answer supported by the context? (1.0 = every claim has a source, 0.0 = entirely fabricated)
2. relevance - Does the answer directly address the user's query? (1.0 = perfectly on-topic, 0.0 = completely irrelevant)
3. completeness - Does the answer cover all key information in the context? (1.0 = nothing important missed, 0.0 = misses everything)
4. groundedness - What fraction of the answer's claims can be traced back to specific passages in the context? (1.0 = every sentence traceable, 0.0 = none traceable)
5. truthfulness - Does the answer avoid factual contradictions with the context? (1.0 = no contradictions at all, 0.0 = contradicts the context)
6. context_relevance - How relevant are the retrieved documents to the user's query? (1.0 = all documents highly relevant, 0.0 = none relevant)

Also identify:
- hallucination_detected: true if any claim in the answer contradicts or cannot be found in the context
- missing_from_context: list of key topics present in the context but absent from the answer

Output JSON only:
{{"faithfulness": ..., "relevance": ..., "completeness": ..., "groundedness": ..., "truthfulness": ..., "context_relevance": ..., "critique": "...", "hallucination_detected": ..., "missing_from_context": [...]}}
"""


class SelfFeedbackLoop:
    """Post-generation evaluation + iterative refinement.

    Uses the router_model (small, fast) as critic to keep latency low.
    """

    def __init__(self, max_iterations: int | None = None) -> None:
        self.max_iterations = max_iterations or settings.self_feedback_max_iterations

    async def evaluate(
        self,
        answer: str,
        context_docs: list[Document],
        query: str,
    ) -> EvaluationResult:
        """Run the critic LLM over answer + context + query. Returns structured scores."""

        context = "\n\n".join(
            f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(context_docs)
        )
        prompt = _CRITIC_PROMPT.format(
            context=context,
            query=query,
            answer=answer,
            threshold=settings.self_feedback_threshold,
        )

        try:
            client = get_llm_client()
            response = await client.chat.completions.create(
                model=settings.router_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise evaluator. Output only valid JSON.\n",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("Critic LLM call failed: %s", exc)
            # Fail open
            return EvaluationResult(
                faithfulness=1.0,
                relevance=1.0,
                completeness=1.0,
                groundedness=1.0,
                truthfulness=1.0,
                context_relevance=1.0,
                critique="Critic LLM call failed, falling open.",
            )

        # Parse JSON
        result: dict = {}
        try:
            result = json.loads(raw.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    pass
            if not result:
                # Try loose JSON match
                match = re.search(r"\{[\s\S]*\}", raw)
                if match:
                    try:
                        result = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass

        if not result:
            logger.warning("Critic response not valid JSON: %s", raw[:200])
            return EvaluationResult(
                faithfulness=1.0,
                relevance=1.0,
                completeness=1.0,
                groundedness=1.0,
                truthfulness=1.0,
                context_relevance=1.0,
                critique="Critic response parsing failed.",
            )

        return EvaluationResult(
            faithfulness=float(result.get("faithfulness", 1.0)),
            relevance=float(result.get("relevance", 1.0)),
            completeness=float(result.get("completeness", 1.0)),
            groundedness=float(result.get("groundedness", 1.0)),
            truthfulness=float(result.get("truthfulness", 1.0)),
            context_relevance=float(result.get("context_relevance", 1.0)),
            critique=result.get("critique", ""),
            hallucination_detected=result.get("hallucination_detected", False),
            missing_from_context=result.get("missing_from_context", []),
        )
