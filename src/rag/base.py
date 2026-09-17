"""Base interfaces and data structures for RAG pipelines.

Adds:
- Cache check before running the pipeline (via RedisCache)
- Automatic Postgres persistence of every RAGResponse (runs + scores tables)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

from src.ingestion.loader import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """Standardised response object returned by all RAG pipelines."""
    query: str
    answer: str
    sources: List[Chunk] = field(default_factory=list)
    latency_sec: float = 0.0
    model: str = ""
    framework: str = ""
    vector_backend: str = ""


# ---------------------------------------------------------------------------
# Base pipeline
# ---------------------------------------------------------------------------

class BaseRAGPipeline(ABC):
    """Abstract base class for all RAG pipelines.

    Subclasses must implement `ingest()` and `_query_impl()`.
    The public `query()` method wraps the implementation with:
      1. Redis cache lookup  → return cached hit immediately
      2. Call `_query_impl()` (the real pipeline)
      3. Persist the Run to Postgres
      4. Store result in Redis cache
    """

    # Subclasses expose these as properties / class attributes
    @property
    def framework_name(self) -> str:
        return ""

    @property
    def backend_name(self) -> str:
        return ""

    @property
    def model_name(self) -> str:
        return ""

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def ingest(self, file_or_dir: Optional[Union[str, Path]] = None) -> int:
        """Load, chunk, and index documents into the vector store. Returns chunk count."""

    @abstractmethod
    def _query_impl(self, question: str, top_k: int = 4) -> RAGResponse:
        """The actual pipeline logic — implemented by each subclass."""

    # ------------------------------------------------------------------
    # Public query entry point (cache + persistence)
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 4) -> RAGResponse:
        """Query with cache-check and automatic DB persistence."""
        from src.cache.redis_cache import cache
        from src.db.session import get_session, create_tables
        from src.db.models import Run

        # 1. Cache check
        cached = cache.get(question)
        if cached:
            logger.info("[CACHE HIT] %.60s…", question)
            # Rebuild a RAGResponse from the stored dict (no sources on cache hit)
            return RAGResponse(
                query=question,
                answer=cached["answer"],
                sources=[],
                latency_sec=cached.get("latency_sec", 0.0),
                model=cached.get("model", self.model_name),
                framework=cached.get("framework", self.framework_name),
                vector_backend=cached.get("vector_backend", self.backend_name),
            )

        # 2. Run the actual pipeline
        response = self._query_impl(question, top_k)

        # 3. Persist to Postgres (best-effort — never crash the pipeline)
        run_id: Optional[int] = None
        try:
            create_tables()
            with get_session() as session:
                run = Run(
                    query=response.query,
                    answer=response.answer,
                    framework=response.framework or self.framework_name,
                    vector_backend=response.vector_backend or self.backend_name,
                    model=response.model or self.model_name,
                    latency_sec=response.latency_sec,
                )
                session.add(run)
                session.flush()        # populate run.id before commit
                run_id = run.id
            logger.debug("Persisted Run id=%s to Postgres.", run_id)
        except Exception as exc:
            logger.warning("Postgres persistence failed (run skipped): %s", exc)

        # 4. Write to cache
        cache.set(question, {
            "answer": response.answer,
            "latency_sec": response.latency_sec,
            "model": response.model,
            "framework": response.framework,
            "vector_backend": response.vector_backend,
            "run_id": run_id,
        })

        return response

    # ------------------------------------------------------------------
    # Helper — save RAGAS scores for a run
    # ------------------------------------------------------------------

    @staticmethod
    def save_scores(run_id: int, scores: Dict[str, float]) -> None:
        """Attach RAGAS scores to an existing Run row."""
        from src.db.session import get_session
        from src.db.models import Score

        try:
            with get_session() as session:
                score = Score(
                    run_id=run_id,
                    faithfulness=scores.get("faithfulness"),
                    answer_relevancy=scores.get("answer_relevancy"),
                    context_precision=scores.get("context_precision"),
                    context_recall=scores.get("context_recall"),
                )
                session.add(score)
            logger.debug("Saved RAGAS scores for run_id=%s.", run_id)
        except Exception as exc:
            logger.warning("Failed to save scores: %s", exc)