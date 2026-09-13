"""Configuration settings for multi-backend RAG benchmark."""

from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrameworkType(str, Enum):
    LANGCHAIN = "langchain"
    LLAMAININDEX = "llamaindex"


class VectorBackendType(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Active switches
    RAG_FRAMEWORK: FrameworkType = FrameworkType.LANGCHAIN
    VECTOR_BACKEND: VectorBackendType = VectorBackendType.CHROMA

    # OpenRouter LLM Settings
    OPENROUTER_API_KEY: Optional[str] = None
    LLM_MODEL: str = "openai/gpt-oss-120b"

    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Ingestion & Chunking
    DATA_DIR: Path = Path("./data/raw_docs")
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Vector Stores
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "rag_bench_docs"

    # Redis & Postgres
    REDIS_URL: str = "redis://localhost:6379/0"
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/rag_bench"


settings = Settings()
