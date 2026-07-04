from pydantic import BaseModel, Field
from app.config import settings


class EvaluationResult(BaseModel):
    """Structured result of feedback evaluation."""

    faithfulness: float
    relevance: float
    completeness: float
    critique: str
    hallucination_detected: bool = False
    missing_from_context: list[str] = Field(default_factory=list)

    @property
    def composite_score(self) -> float:
        return min(self.faithfulness, self.relevance, self.completeness)

    @property
    def passes(self) -> bool:
        return (
            not self.hallucination_detected
            and self.composite_score >= settings.self_feedback_threshold
        )
