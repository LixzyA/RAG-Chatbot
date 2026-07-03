"""Feedback log entity — per-iteration critic scores and refinement strategy.

One row per self-feedback iteration. Links to ``rag_traces`` so the full
pipeline (retrieval → generation → critic → refinement) is queryable per
session for offline analysis.
"""

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.entity.base import Base


class FeedbackLog(Base):
    __tablename__ = "feedback_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # FK → rag_traces so feedback belongs to a specific query trace.
    rag_trace_id: Mapped[int | None] = mapped_column(
        ForeignKey("rag_traces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="FK → rag_traces.id. Links feedback to a specific query trace.",
    )

    # FK → chat_sessions for session-scoped queries.
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="FK → chat_sessions.id. Groups feedback by conversation.",
    )

    # Iteration counter (0-indexed; first generation).
    iteration: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Feedback iteration number (0 = first pass, 1 = first retry, …)",
    )

    # The query used for this iteration (original or feedback-rewritten).
    query_used: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="The query string used for retrieval in this iteration",
    )

    # Draft answer produced this iteration.
    draft_answer: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="The LLM answer before any further refinement",
    )

    # Critic scores (0.0–1.0).
    faithfulness: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Critic faithfulness score"
    )
    relevance: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Critic relevance score"
    )
    completeness: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Critic completeness score"
    )
    composite_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="min(faithfulness, relevance, completeness)",
    )

    # Critic narrative.
    critique: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Critic's free-text explanation"
    )

    # Refinement strategy applied after this iteration (if any).
    refinement_strategy: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Refinement instruction or 're_retrieve: <query>' label",
    )

    # Whether this iteration passed (needs_refinement = False).
    passed: Mapped[bool] = mapped_column(
        default=False, comment="True if critic deemed this answer acceptable"
    )

    # Timestamps.
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        comment="Timestamp when the feedback row was created",
    )

    def __repr__(self) -> str:
        return (
            f"<FeedbackLog(id={self.id}, iter={self.iteration}, "
            f"composite={self.composite_score}, passed={self.passed})>"
        )
