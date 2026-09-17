"""SQLAlchemy 2.x ORM models — runs and scores tables."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Run(Base):
    """Stores one RAG query execution."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    framework: Mapped[str] = mapped_column(String(50), nullable=False)
    vector_backend: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    latency_sec: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # One Run → one Score (optional, populated after RAGAS eval)
    score: Mapped[Optional["Score"]] = relationship(
        "Score", back_populates="run", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Run id={self.id} framework={self.framework!r} "
            f"backend={self.vector_backend!r} model={self.model!r}>"
        )


class Score(Base):
    """RAGAS evaluation scores linked to a Run."""

    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    faithfulness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    context_recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    run: Mapped["Run"] = relationship("Run", back_populates="score")

    def __repr__(self) -> str:
        f_str = f"{self.faithfulness:.3f}" if self.faithfulness is not None else "N/A"
        return f"<Score run_id={self.run_id} faithfulness={f_str}>"
