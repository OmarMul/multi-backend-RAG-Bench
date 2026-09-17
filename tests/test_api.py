"""Tests for the FastAPI layer — Milestone 9."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures & lightweight test doubles (avoid heavy ML imports)
# ---------------------------------------------------------------------------

@dataclass
class DummyChunk:
    content: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class DummyRAGResponse:
    query: str
    answer: str
    sources: list = field(default_factory=list)
    latency_sec: float = 0.0
    model: str = ""
    framework: str = ""
    vector_backend: str = ""


@pytest.fixture()
def mock_rag_response():
    chunk = DummyChunk(content="RAG combines retrieval and generation.", metadata={"source": "doc.pdf"})
    return DummyRAGResponse(
        query="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation.",
        sources=[chunk],
        latency_sec=0.5,
        model="openai/gpt-4o",
        framework="langchain",
        vector_backend="chroma",
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /api/v1/query
# ---------------------------------------------------------------------------

def test_query_success(mock_rag_response):
    pipeline_instance = MagicMock()
    pipeline_instance.query.return_value = mock_rag_response

    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    with (
        patch("src.api.routes._get_pipeline", return_value=pipeline_instance),
        patch("src.db.session.get_session") as mock_session_ctx,
    ):
        mock_session_ctx.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_ctx.return_value.__exit__ = MagicMock(return_value=False)
        res = client.post("/api/v1/query", json={"question": "What is RAG?"})

    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "RAG stands for Retrieval-Augmented Generation."
    assert data["latency_sec"] == 0.5
    assert data["run_id"] is None


def test_query_pipeline_error():
    with patch("src.api.routes._get_pipeline", side_effect=RuntimeError("vector store down")):
        res = client.post("/api/v1/query", json={"question": "fail"})

    assert res.status_code == 500
    assert "vector store down" in res.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/v1/runs
# ---------------------------------------------------------------------------

def test_list_runs_empty():
    mock_session = MagicMock()
    mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = []

    with (
        patch("src.db.session.create_tables"),
        patch("src.db.session.get_session") as ctx,
    ):
        ctx.return_value.__enter__ = MagicMock(return_value=mock_session)
        ctx.return_value.__exit__ = MagicMock(return_value=False)
        res = client.get("/api/v1/runs")

    assert res.status_code == 200
    assert res.json() == []


# ---------------------------------------------------------------------------
# GET /api/v1/runs/{id}/score
# ---------------------------------------------------------------------------

def test_get_score_not_found():
    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = None

    with patch("src.db.session.get_session") as ctx:
        ctx.return_value.__enter__ = MagicMock(return_value=mock_session)
        ctx.return_value.__exit__ = MagicMock(return_value=False)
        res = client.get("/api/v1/runs/999/score")

    assert res.status_code == 404


def test_get_score_found():
    mock_score = MagicMock()
    mock_score.run_id = 1
    mock_score.faithfulness = 0.85
    mock_score.answer_relevancy = 0.90
    mock_score.context_precision = 0.80
    mock_score.context_recall = 0.75

    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = mock_score

    with patch("src.db.session.get_session") as ctx:
        ctx.return_value.__enter__ = MagicMock(return_value=mock_session)
        ctx.return_value.__exit__ = MagicMock(return_value=False)
        res = client.get("/api/v1/runs/1/score")

    assert res.status_code == 200
    data = res.json()
    assert data["faithfulness"] == 0.85
    assert data["answer_relevancy"] == 0.90
