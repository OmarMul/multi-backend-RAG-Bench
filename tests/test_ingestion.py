"""Tests for document loading and chunking."""

from pathlib import Path
import pytest
from src.ingestion.loader import load_and_chunk, Chunk


@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a temporary text file with sample content."""
    file = tmp_path / "sample.txt"
    content = "RAG benchmark test paragraph. " * 50
    file.write_text(content, encoding="utf-8")
    return file


def test_load_and_chunk_txt(sample_txt_file: Path):
    """Test loading and chunking a plain text file."""
    chunk_size = 200
    chunk_overlap = 20
    chunks = load_and_chunk(sample_txt_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].metadata["source"] == "sample.txt"
    assert chunks[0].metadata["chunk_index"] == 0
    assert len(chunks[0].content) <= chunk_size


def test_file_not_found():
    """Test handling of non-existent files."""
    with pytest.raises(FileNotFoundError):
        load_and_chunk("non_existent_file.pdf")


def test_unsupported_format(tmp_path: Path):
    """Test rejection of unsupported file extensions."""
    file = tmp_path / "sample.csv"
    file.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_and_chunk(file)
