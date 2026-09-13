"""LangChain RAG Pipeline implementation."""

import time
from pathlib import Path
from typing import List, Optional, Union


from src.config import settings
from src.generation.openrouter_client import OpenRouterClient
from src.ingestion.loader import Chunk, load_and_chunk
from src.rag.base import BaseRAGPipeline, RAGResponse
from src.vectorstores.chroma_store import ChromaVectorStore




class LangChainPipeline(BaseRAGPipeline):
    """LangChain-based RAG pipeline using Chroma and OpenRouter."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        llm_client: Optional[OpenRouterClient] = None,
        model_name: Optional[str] = None,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.llm_client = llm_client or OpenRouterClient()
        self.model_name = model_name or settings.LLM_MODEL
        self.framework_name = "LangChain"
        self.backend_name = "Chroma"

    def ingest(self, file_or_dir: Optional[Union[str, Path]] = None) -> int:
        """Ingest document(s) from a file or folder into the Chroma vector store."""

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


        return self.vector_store.add_chunks(all_chunks)
    

    def query(self, question: str, top_k: int = 4) -> RAGResponse:
        """Query the vector store, build context, and generate an answer via OpenRouter."""

        start_time = time.perf_counter()

        # 1. Retrieve relevant chunks
        retrieved_chunks = self.vector_store.similarity_search(query=question, k=top_k)

        # 2. Format context
        context_blocks = [
            f"[Source: {c.metadata.get('source', 'unknown')} | Chunk {c.metadata.get('chunk_index', 0)}]\n{c.content}"
            for c in retrieved_chunks
        ]
        context_str = "\n\n---\n\n".join(context_blocks)

        # 3. Generate response via OpenRouter
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
    pipeline = LangChainPipeline()
    print("Ingesting raw documents...")
    chunk_count = pipeline.ingest()
    print(f"Ingested {chunk_count} chunks into Chroma.")
    test_query = "What are the core components of agentic AI architecture?"
    print(f"\nQuerying: '{test_query}'...")
    response = pipeline.query(test_query, top_k=3)
    print(f"\n--- Answer (Latency: {response.latency_sec}s | Model: {response.model}) ---")
    print(response.answer)
    print(f"\n--- Sources ({len(response.sources)} chunks cited) ---")
    for s in response.sources:
        print(f"- {s.metadata.get('source')} (page {s.metadata.get('page', 'N/A')})")
