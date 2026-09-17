"""Tests for Milestone 8 — PostgreSQL persistence and Redis cache.

Run with:
    pytest tests/test_persistence.py -v

Requirements:
    - Docker services running: `docker compose up -d postgres redis`
    - .env has correct DATABASE_URL and REDIS_URL
"""

import pytest

# ---------------------------------------------------------------------------
# Helpers — skip gracefully when services are unavailable
# ---------------------------------------------------------------------------

def _db_available() -> bool:
    try:
        from src.db.session import ping_db
        return ping_db()
    except Exception:
        return False


def _redis_available() -> bool:
    try:
        from src.cache.redis_cache import RedisCache
        rc = RedisCache()
        return rc.ping()
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _db_available(),
    reason="PostgreSQL not reachable — start Docker first"
)

requires_redis = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis not reachable — start Docker first"
)


# ---------------------------------------------------------------------------
# PostgreSQL tests
# ---------------------------------------------------------------------------

@requires_db
def test_db_ping():
    """Database is reachable."""
    from src.db.session import ping_db
    assert ping_db() is True


@requires_db
def test_create_tables():
    """Tables can be created without error."""
    from src.db.session import create_tables
    create_tables()  # idempotent


@requires_db
def test_run_insert_and_query():
    """Insert a Run row and retrieve it."""
    from src.db.session import get_session, create_tables
    from src.db.models import Run

    create_tables()

    with get_session() as session:
        run = Run(
            query="What is RAG?",
            answer="Retrieval-Augmented Generation.",
            framework="langchain",
            vector_backend="chroma",
            model="openai/gpt-4o-mini",
            latency_sec=1.23,
        )
        session.add(run)
        session.flush()
        run_id = run.id

    assert run_id is not None

    with get_session() as session:
        fetched = session.get(Run, run_id)
        assert fetched is not None
        assert fetched.query == "What is RAG?"
        assert fetched.framework == "langchain"
        assert fetched.latency_sec == pytest.approx(1.23)


@requires_db
def test_score_insert_linked_to_run():
    """Insert a Score linked to a Run and verify relationship."""
    from src.db.session import get_session, create_tables
    from src.db.models import Run, Score

    create_tables()

    with get_session() as session:
        run = Run(
            query="Explain embeddings.",
            answer="Embeddings are dense vector representations.",
            framework="llamaindex",
            vector_backend="qdrant",
            model="openai/gpt-4o-mini",
            latency_sec=0.95,
        )
        session.add(run)
        session.flush()

        score = Score(
            run_id=run.id,
            faithfulness=0.85,
            answer_relevancy=0.78,
            context_precision=0.60,
            context_recall=0.72,
        )
        session.add(score)
        session.flush()
        score_id = score.id
        run_id = run.id

    with get_session() as session:
        fetched_score = session.get(Score, score_id)
        assert fetched_score is not None
        assert fetched_score.run_id == run_id
        assert fetched_score.faithfulness == pytest.approx(0.85)
        assert fetched_score.context_precision == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Redis cache tests
# ---------------------------------------------------------------------------

@requires_redis
def test_redis_ping():
    """Redis is reachable."""
    from src.cache.redis_cache import RedisCache
    rc = RedisCache()
    assert rc.ping() is True


@requires_redis
def test_cache_set_and_get():
    """Cache set → get returns same value."""
    from src.cache.redis_cache import RedisCache
    rc = RedisCache()

    query = "test_cache_set_and_get unique query xyz"
    payload = {"answer": "42", "latency_sec": 0.5, "model": "test-model"}

    rc.delete(query)  # clean slate
    assert rc.get(query) is None

    rc.set(query, payload)
    result = rc.get(query)

    assert result is not None
    assert result["answer"] == "42"
    assert result["model"] == "test-model"


@requires_redis
def test_cache_delete():
    """Deleted key is gone from cache."""
    from src.cache.redis_cache import RedisCache
    rc = RedisCache()

    query = "test_cache_delete unique query abc"
    rc.set(query, {"answer": "to be deleted"})
    rc.delete(query)
    assert rc.get(query) is None


@requires_redis
def test_cache_miss_returns_none():
    """Non-existent key returns None."""
    from src.cache.redis_cache import RedisCache
    rc = RedisCache()
    assert rc.get("query that has never been cached zzzzzz") is None


# ---------------------------------------------------------------------------
# Integration — base pipeline writes to DB + cache
# ---------------------------------------------------------------------------

@requires_db
@requires_redis
def test_pipeline_persists_run(tmp_path):
    """Calling query() on a pipeline writes a Run to Postgres and cache."""
    from unittest.mock import MagicMock, patch
    from src.rag.base import BaseRAGPipeline, RAGResponse
    from src.db.session import get_session, create_tables
    from src.db.models import Run

    create_tables()

    # Minimal concrete subclass
    class MockPipeline(BaseRAGPipeline):
        framework_name = "langchain"
        backend_name = "chroma"
        model_name = "mock-model"

        def ingest(self, file_or_dir=None) -> int:
            return 0

        def _query_impl(self, question: str, top_k: int = 4) -> RAGResponse:
            return RAGResponse(
                query=question,
                answer="Mock answer",
                sources=[],
                latency_sec=0.1,
                model=self.model_name,
                framework=self.framework_name,
                vector_backend=self.backend_name,
            )

    pipeline = MockPipeline()
    question = "Integration test: does the pipeline persist runs?"

    # Clear cache so we don't get a hit
    from src.cache.redis_cache import cache as _cache
    _cache.delete(question)

    response = pipeline.query(question)

    assert response.answer == "Mock answer"

    # Verify Run was persisted
    with get_session() as session:
        from sqlalchemy import text
        result = session.execute(
            text("SELECT id, query, framework FROM runs WHERE query = :q LIMIT 1"),
            {"q": question}
        ).fetchone()
    assert result is not None, "Run row should exist in Postgres after query()"
    assert result[2] == "langchain"

    # Verify cache has the answer
    cached = _cache.get(question)
    assert cached is not None
    assert cached["answer"] == "Mock answer"
