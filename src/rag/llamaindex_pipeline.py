"""LlamaIndex RAG Pipeline with swappable Chroma/Qdrant vector backends."""

import time
from pathlib import Path
from typing import List, Optional, Union
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import VectorStoreIndex, StorageContext, Settings as LlamaSettings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore as LlamaChromaStore
from llama_index.vector_stores.qdrant import QdrantVectorStore as LlamaQdrantStore
from qdrant_client import QdrantClient

from src.config import settings, VectorBackendType
from src.generation.openrouter_client import OpenRouterClient
from src.ingestion.loader import Chunk, load_and_chunk
from src.rag.base import BaseRAGPipeline, RAGResponse
from src.eval.tracing import traced

class LlamaIndexPipeline(BaseRAGPipeline):
    """LlamaIndex-based RAG pipeline supporting both Chroma and Qdrant."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_client: Optional[OpenRouterClient] = None,
        model_name: Optional[str] = None,
        backend: Optional[VectorBackendType] = None,
    ):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.embedding_model_name = embedding_model_name
        self.llm_client = llm_client or OpenRouterClient()
        self.model_name = model_name or settings.LLM_MODEL
        self.framework_name = "llamaindex"
        self.backend_type = backend or settings.VECTOR_BACKEND
        self.backend_name = self.backend_type.value

        # Initialize LlamaIndex embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)
        LlamaSettings.embed_model = self.embed_model
        LlamaSettings.llm = None  # Handled explicitly via OpenRouterClient

        # Initialize vector store based on active backend switch
        if self.backend_type == VectorBackendType.QDRANT:
            self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
            try:
                self.qdrant_client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    timeout=3.0
                )
                self.qdrant_client.get_collections()
            except Exception:
                local_dir = "./data/qdrant_db"
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                self.qdrant_client = QdrantClient(path=local_dir)

            self.vector_store = LlamaQdrantStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
            )
        else:
            self.collection_name = collection_name or "rag_bench_llamaindex"
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False)
            )
            self.chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)
            self.vector_store = LlamaChromaStore(chroma_collection=self.chroma_collection)

        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

    def ingest(self, file_or_dir: Optional[Union[str, Path]] = None) -> int:
        """Ingest documents, convert to TextNodes, and index in active vector store."""
        target_path = Path(file_or_dir or settings.DATA_DIR)
        all_chunks: List[Chunk] = []

        if target_path.is_file():
            all_chunks.extend(load_and_chunk(target_path))
        elif target_path.is_dir():
            valid_exts = {".pdf", ".docx", ".doc", ".txt", ".md"}
            files = [f for f in target_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
            for file in files:
                all_chunks.extend(load_and_chunk(file))
        else:
            raise FileNotFoundError(f"Path not found: {target_path}")

        if not all_chunks:
            return 0

        nodes = [
            TextNode(
                text=chunk.content,
                metadata=chunk.metadata,
                id_=str(uuid.uuid5(
                    uuid.NAMESPACE_DNS, 
                    f"{self.backend_name}_{chunk.metadata.get('source', 'doc')}_{chunk.metadata.get('chunk_index', i)}_{i}"
                )),
            )
            for i, chunk in enumerate(all_chunks)
        ]

        self.index.insert_nodes(nodes)
        return len(nodes)
    @traced("query")
    def _query_impl(self, question: str, top_k: int = 4) -> RAGResponse:
        """Retrieve matching nodes and generate answer via OpenRouter."""
        start_time = time.perf_counter()

        retriever = self.index.as_retriever(similarity_top_k=top_k)
        retrieved_nodes = retriever.retrieve(question)

        retrieved_chunks: List[Chunk] = [
            Chunk(content=node.node.get_content(), metadata=node.node.metadata)
            for node in retrieved_nodes
        ]

        context_blocks = [
            f"[Source: {c.metadata.get('source', 'unknown')} | Chunk {c.metadata.get('chunk_index', 0)}]\n{c.content}"
            for c in retrieved_chunks
        ]
        context_str = "\n\n---\n\n".join(context_blocks)

        answer = self.llm_client.generate(
            prompt=question,
            context=context_str,
            model=self.model_name,
        )

        latency = time.perf_counter() - start_time

        return RAGResponse(
            query=question,
            answer=answer,
            sources=retrieved_chunks,
            latency_sec=round(latency, 4),
            model=self.model_name,
            framework=self.framework_name,
            vector_backend=self.backend_name,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("📊 Testing Chroma vs Qdrant with LlamaIndex")
    print("=" * 60)

    test_query = "What are the core components of agentic AI architecture?"

    # 1. LlamaIndex + Chroma
    print("\n[1] Running LlamaIndex + Chroma...")
    llama_chroma = LlamaIndexPipeline(backend=VectorBackendType.CHROMA)
    res_chroma = llama_chroma.query(test_query, top_k=3)
    print(f"⏱️  Chroma Latency: {res_chroma.latency_sec}s | Cited Chunks: {len(res_chroma.sources)}")

    # 2. LlamaIndex + Qdrant
    print("\n[2] Ingesting & Running LlamaIndex + Qdrant...")
    llama_qdrant = LlamaIndexPipeline(backend=VectorBackendType.QDRANT)
    llama_qdrant.ingest()
    res_qdrant = llama_qdrant.query(test_query, top_k=3)
    print(f"⏱️  Qdrant Latency: {res_qdrant.latency_sec}s | Cited Chunks: {len(res_qdrant.sources)}")
