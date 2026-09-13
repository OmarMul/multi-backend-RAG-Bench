"""Base interfaces and data structures for RAG pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from src.ingestion.loader import Chunk



@dataclass
class RAGResponse:
    """Standardized response object returned by all RAG pipelines."""
    query: str
    answer: str
    sources: List[Chunk] = field(default=list)
    latency_sec: float = 0.0
    model: str = ""
    framework: str = ""
    vector_backend: str = ""


class BaseRAGPipeline(ABC):
    """Abstract base class for RAG pipelines."""
    @abstractmethod
    def ingest(self, file_or_dir: Optional[Union[str, Path]] = None) -> int:
        """Load, chunk, and index documents into the vector store. Returns chunk count."""
        pass
    @abstractmethod
    def query(self, question: str, top_k: int = 4) -> RAGResponse:
        """Query the RAG pipeline with a question and retrieve answer + sources."""
        pass