"""Tests for Chroma and Qdrant vector store implementations."""

from pathlib import Path
import pytest
from src.ingestion.loader import Chunk
from src.vectorstores.chroma_store import ChromaVectorStore
from src.vectorstores.qdrant_store import QdrantStore


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            content="Agentic AI systems use autonomous planning and memory retrieval.",
            metadata={"source": "agentic_doc.txt", "chunk_index": 0}
        ),
        Chunk(
            content="Vector databases index embeddings for high-dimensional cosine similarity search.",
            metadata={"source": "vector_doc.txt", "chunk_index": 1}
        ),
    ]


@pytest.mark.parametrize("store_cls,dir_name", [
    (ChromaVectorStore, "test_chroma_db"),
    (QdrantStore, "test_qdrant_db"),
])
def test_vector_store_crud(tmp_path: Path, sample_chunks, store_cls, dir_name):
    """Test chunk indexing, count, and similarity search across both vector stores."""
    db_path = str(tmp_path / dir_name)
    store = store_cls(persist_dir=db_path) if store_cls == ChromaVectorStore else store_cls(path=db_path)

    # 1. Add chunks
    added = store.add_chunks(sample_chunks)
    assert added == len(sample_chunks)
    assert store.count() >= 2

    # 2. Search similarity
    results = store.similarity_search("How do agentic systems plan?", k=1)
    assert len(results) == 1
    assert "autonomous planning" in results[0].content
    assert results[0].metadata["source"] == "agentic_doc.txt"

    # 3. Clear store
    store.clear()
    assert store.count() == 0
