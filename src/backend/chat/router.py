"""
Standalone query classifier for model routing.

Uses a lightweight router model (Llama-3.2-1B-Instruct) to classify incoming
queries into topics. Serious/hard topics are routed to the specialist model,
while general topics go to the generalist model.
"""

import json
# import logging # [LOGGING REMOVED]
import os
import re

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

# ---------------------------------------------------------------------------
# Topic taxonomy
# ---------------------------------------------------------------------------
# Topics considered "serious" — requiring the specialist model.
# Add or remove entries to tune which queries get specialist treatment.
SERIOUS_TOPICS: dict[str, str] = {
    "legal": "Law, contracts, regulations, compliance, legal disputes",
    "medical": "Health, diseases, treatments, medications, anatomy",
    "financial": "Accounting, tax, investments, budgeting, financial planning",
    "technical": "Programming, engineering, system architecture, debugging",
    "scientific": "Research, physics, chemistry, biology, mathematics",
    "security": "Cybersecurity, encryption, vulnerabilities, data protection",
}

# Confidence threshold — below this the query falls back to generalist.
CONFIDENCE_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------
_CLASSIFICATION_PROMPT = """You are a query classifier. Classify the user's query into exactly one topic.

Serious topics (require deep expertise):
{topics_block}

General topics (everything else):
- small_talk: Greetings, casual conversation
- general_knowledge: Trivia, facts, history, geography
- creative: Story writing, poetry, brainstorming
- opinion: Subjective questions, recommendations
- other: Anything that doesn't fit the above

Respond with ONLY a valid JSON object on a single line, no markdown, no explanation:
{{"topic": "<topic_key>", "confidence": <float between 0.0 and 1.0>}}

User query: {query}"""


def _build_topics_block() -> str:
    """Format the serious topics dict into a readable block for the prompt."""
    lines = []
    for key, desc in SERIOUS_TOPICS.items():
        lines.append(f"- {key}: {desc}")
    return "\n".join(lines)


def _parse_classification(raw_text: str) -> dict:
    """
    Parse the router model's response into a classification dict.
    
    Attempts JSON parsing first, then falls back to regex extraction.
    Returns {"topic": "other", "confidence": 0.0} on failure.
    """
    fallback = {"topic": "other", "confidence": 0.0}
    
    text = raw_text.strip()
    
    # Try direct JSON parse
    try:
        result = json.loads(text)
        if "topic" in result and "confidence" in result:
            return {
                "topic": str(result["topic"]).lower().strip(),
                "confidence": float(result["confidence"]),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Fallback: extract JSON from within the text (model may wrap it)
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if "topic" in result and "confidence" in result:
                return {
                    "topic": str(result["topic"]).lower().strip(),
                    "confidence": float(result["confidence"]),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    # [LOGGING REMOVED]
    return fallback


async def classify_query(llm_client, query: str) -> dict:
    """
    Classify a user query into a topic using the router model.
    
    Returns:
        dict with keys:
            - topic (str): The classified topic key.
            - confidence (float): Confidence score 0.0–1.0.
    """
    prompt = _CLASSIFICATION_PROMPT.format(
        topics_block=_build_topics_block(),
        query=query,
    )
    
    try:
        response = await llm_client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise query classifier. Output only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,    # Low temperature for deterministic classification
            max_tokens=64,      # Short response — just a JSON object
        )
        
        raw = response.choices[0].message.content or ""
        classification = _parse_classification(raw)
        
        # [LOGGING REMOVED]
        
        return classification
    
    except Exception as e:
        # [LOGGING REMOVED]
        # On failure, default to generalist (safe fallback)
        return {"topic": "other", "confidence": 0.0}


def is_serious_topic(classification: dict) -> bool:
    """
    Determine whether a classification result should be routed to the
    specialist model.
    
    Returns True if the topic is in SERIOUS_TOPICS AND the confidence
    meets the threshold.
    """
    topic = classification.get("topic", "other")
    confidence = classification.get("confidence", 0.0)
    return topic in SERIOUS_TOPICS and confidence >= CONFIDENCE_THRESHOLD
