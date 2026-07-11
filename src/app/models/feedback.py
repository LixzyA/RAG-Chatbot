from pydantic import BaseModel, Field
from app.config import settings


class EvaluationResult(BaseModel):
    """Structured result of feedback evaluation."""

    faithfulness: float
    relevance: float
    completeness: float
    groundedness: float = 1.0
    truthfulness: float = 1.0
    context_relevance: float = 1.0
    critique: str
    hallucination_detected: bool = False
    missing_from_context: list[str] = Field(default_factory=list)

    @property
    def composite_score(self) -> float:
        return (
            0.25 * self.faithfulness
            + 0.20 * self.groundedness
            + 0.20 * self.truthfulness
            + 0.15 * self.relevance
            + 0.10 * self.completeness
            + 0.10 * self.context_relevance
        )

    @property
    def passes(self) -> bool:
        return (
            not self.hallucination_detected
            and self.composite_score >= settings.self_feedback_threshold
        )
