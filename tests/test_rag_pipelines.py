"""Tests for LangChain RAG pipeline."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.ingestion.loader import Chunk
from src.rag.base import RAGResponse
from src.rag.langchain_pipeline import LangChainPipeline
from src.vectorstores.chroma_store import ChromaVectorStore


@pytest.fixture
def temp_chroma_store(tmp_path: Path) -> ChromaVectorStore:
    """Create an isolated Chroma store in a temporary directory."""
    return ChromaVectorStore(
        persist_dir=str(tmp_path / "test_chroma"),
        collection_name="test_collection"
    )


@pytest.fixture
def mock_openrouter_client() -> MagicMock:
    """Create a mock OpenRouter client to avoid real API costs during testing."""
    client = MagicMock()
    client.generate.return_value = "This is a verified test answer based on the context."
    return client


def test_langchain_pipeline_query(temp_chroma_store: ChromaVectorStore, mock_openrouter_client: MagicMock):
    """Test full LangChain pipeline query workflow."""
    chunks = [
        Chunk(
            content="Agentic AI systems use autonomous planning and memory retrieval.",
            metadata={"source": "agentic_ai.pdf", "page": 1, "chunk_index": 0}
        )
    ]
    temp_chroma_store.add_chunks(chunks)

    pipeline = LangChainPipeline(
        vector_store=temp_chroma_store,
        llm_client=mock_openrouter_client,
        model_name="openai/gpt-4o-mini"
    )

    response = pipeline.query("What does agentic AI use?", top_k=1)

    assert isinstance(response, RAGResponse)
    assert response.framework.lower() == "langchain"
    assert response.vector_backend.lower() == "chroma"
    assert len(response.sources) == 1
    assert "Agentic AI systems" in response.sources[0].content
    assert response.answer == "This is a verified test answer based on the context."
    assert response.latency_sec >= 0
