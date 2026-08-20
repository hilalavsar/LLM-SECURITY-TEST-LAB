"""Persistence · SQLAlchemy models.

Tables: test_runs, test_results. (judge_verdicts / model_configs come later.)
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Run(Base):
    __tablename__ = "test_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    model: Mapped[str] = mapped_column(String(160))
    configs: Mapped[str] = mapped_column(String(400))  # comma-joined config ids
    # Empty string means judge was disabled for this run.
    judge_model: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    results: Mapped[list["Result"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="Result.pk"
    )


class Result(Base):
    __tablename__ = "test_results"

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("test_runs.id"))
    config: Mapped[str] = mapped_column(String(60))
    case_id: Mapped[str] = mapped_column(String(30))
    category: Mapped[str] = mapped_column(String(40))
    owasp: Mapped[str] = mapped_column(String(10))
    detection: Mapped[str] = mapped_column(String(20))
    verdict: Mapped[str] = mapped_column(String(10))
    reason: Mapped[str] = mapped_column(String(240))
    prompt: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped["Run"] = relationship(back_populates="results")
