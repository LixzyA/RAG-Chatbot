"""Query processor — classification, expansion, and routing.

Decides which model/prompt to use and optionally rewrites the user query
for better retrieval (HyDE, multi-query, etc.).

Source: backend/vectordb/query_transformer.py (classification + rewrite logic).
Uses the **router model** (small) for all LLM calls — classification,
rewriting, decomposition, and HyDE generation.
"""

from __future__ import annotations

import json
import re
from typing import Any
import logging

from app.config import settings
from app.core.generation.llm_client import get_llm_client

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prompts (inlined so the module is self-contained)
# ------------------------------------------------------------------

_CLASSIFY_PROMPT = """Classify the following user query. Output a JSON object with two keys: "query_type" and "confidence".

Available query types:
- "simple": Greetings, casual chat, or very simple factual questions.
- "factual": A direct question that can be answered with retrieved documents.
- "multi_part": A complex query with multiple distinct parts that should be decomposed.
- "domain_specific": A technical or specialized topic where a hypothetical document would help.
- "conversational": Small talk, opinions, or creative prompts (e.g., "tell me a joke").

Example output:
{{"query_type": "factual", "confidence": 0.92}}

Query: {query}
"""

_REWRITE_PROMPT = """Rewrite the following query to make it more effective for searching a document database.
Be concise and preserve the original meaning. Output only the rewritten query.

Original query: {query}

Rewritten query:"""

_DECOMPOSE_PROMPT = """Break the following complex query into 2-3 simpler, independent sub-queries.
Output them as a JSON array of strings.

Query: {query}
"""

_HYDE_PROMPT = """Write a short hypothetical passage that would answer the following question.
The passage should be detailed enough to be used as a search query in a document database.

Question: {query}

Passage:"""

QUERY_TYPE_TO_STRATEGY: dict[str, str] = {
    "factual": "rewrite",
    "multi_part": "decompose",
    "domain_specific": "hyde",
    "conversational": "rewrite",
    "simple": "passthrough",
}

VALID_QUERY_TYPES = set(QUERY_TYPE_TO_STRATEGY.keys())


class QueryProcessor:
    """LLM-based query classification and transformation for improved retrieval."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.router_model
        self._client = get_llm_client()

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------
    async def _llm_call(
        self, system_prompt: str, user_prompt: str, *, max_tokens: int = 256
    ) -> str:
        """Low-temperature LLM call returning raw text."""
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_classification(raw_text: str) -> dict[str, Any]:
        fallback: dict[str, Any] = {"query_type": "simple", "confidence": 0.0}
        text = raw_text.strip()

        # Direct JSON parse
        try:
            result = json.loads(text)
            qtype = str(result["query_type"]).lower().strip()
            if qtype in VALID_QUERY_TYPES:
                return {"query_type": qtype, "confidence": float(result["confidence"])}
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: extract JSON-like fragment
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            try:
                result = json.loads(json_match.group())
                qtype = str(result["query_type"]).lower().strip()
                if qtype in VALID_QUERY_TYPES:
                    return {
                        "query_type": qtype,
                        "confidence": float(result["confidence"]),
                    }
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        return fallback

    async def classify(self, query: str) -> dict[str, Any]:
        prompt = _CLASSIFY_PROMPT.format(query=query)
        try:
            raw = await self._llm_call(
                system_prompt="You are a precise query classifier. Output only valid JSON.",
                user_prompt=prompt,
                max_tokens=128,
            )
            return self._parse_classification(raw)
        except Exception as exc:
            logger.warning(
                "Query classification failed, falling back to 'simple': %s", exc
            )
            return {"query_type": "simple", "confidence": 0.0}

    # ------------------------------------------------------------------
    # Transformation strategies
    # ------------------------------------------------------------------

    async def rewrite(self, query: str) -> list[str]:
        prompt = _REWRITE_PROMPT.format(query=query)
        try:
            raw = await self._llm_call(
                system_prompt="You are a search query optimizer. Output only the rewritten query.",
                user_prompt=prompt,
                max_tokens=256,
            )
            rewritten = raw.strip().strip("\"'")
            if not rewritten:
                return [query]
            return [rewritten]
        except Exception as exc:
            logger.warning("Query rewrite failed, using original: %s", exc)
            return [query]

    async def decompose(self, query: str) -> list[str]:
        prompt = _DECOMPOSE_PROMPT.format(query=query)
        try:
            raw = await self._llm_call(
                system_prompt="You are a search query decomposer. Output only a valid JSON array of strings.",
                user_prompt=prompt,
                max_tokens=512,
            )
            return self._parse_decompose(raw, query)
        except Exception as exc:
            logger.warning("Query decomposition failed, using original: %s", exc)
            return [query]

    @staticmethod
    def _parse_decompose(raw_text: str, fallback_query: str) -> list[str]:
        text = raw_text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, list) and all(isinstance(s, str) for s in result):
                if 2 <= len(result) <= 5:
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if array_match:
            try:
                result = json.loads(array_match.group())
                if isinstance(result, list) and all(isinstance(s, str) for s in result):
                    if 2 <= len(result) <= 5:
                        return result
            except (json.JSONDecodeError, ValueError):
                pass

        return [fallback_query]

    async def hyde(self, query: str) -> list[str]:
        prompt = _HYDE_PROMPT.format(query=query)
        try:
            raw = await self._llm_call(
                system_prompt="You are a domain expert. Write a short hypothetical passage that answers the question.",
                user_prompt=prompt,
                max_tokens=512,
            )
            passage = raw.strip()
            if not passage:
                return [query]
            return [passage]
        except Exception as exc:
            logger.warning("HyDE generation failed, using original: %s", exc)
            return [query]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def transform(self, query: str) -> dict[str, Any]:
        """Classify *query* and apply the matching transformation strategy.

        Returns:
            {
                "original_query": str,
                "query_type": str,
                "strategy": str,
                "transformed_queries": list[str],
                "confidence": float,
            }
        """
        if not settings.query_transform_enabled:
            return {
                "original_query": query,
                "query_type": "disabled",
                "strategy": "passthrough",
                "transformed_queries": [query],
                "confidence": 1.0,
            }

        classification = await self.classify(query)
        query_type = classification["query_type"]
        confidence = classification["confidence"]
        strategy = QUERY_TYPE_TO_STRATEGY.get(query_type, "passthrough")

        if strategy == "rewrite":
            transformed = await self.rewrite(query)
        elif strategy == "decompose":
            transformed = await self.decompose(query)
        elif strategy == "hyde":
            transformed = await self.hyde(query)
        else:
            transformed = [query]

        result = {
            "original_query": query,
            "query_type": query_type,
            "strategy": strategy,
            "transformed_queries": transformed,
            "confidence": confidence,
        }

        logger.info(
            "Query transformed: type=%s strategy=%s confidence=%.2f | '%s' -> %s",
            query_type,
            strategy,
            confidence,
            query,
            transformed,
        )

        return result
