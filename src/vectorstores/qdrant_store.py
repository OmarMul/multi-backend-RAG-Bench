"""Qdrant vector store implementation."""

import collections
from pathlib import Path
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore as LangchainQdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config import settings
from src.ingestion.loader import Chunk

class QdrantStore:
    """Manages embeddings and similarity search using Qdrant."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[str] = None,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=path)

        elif host and port:
            self.client = QdrantClient(host=host, port=port)

        else:
            try:
                # try docker
                self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=3.0)
                self.client.get_collections()
            except Exception:
                # Fallback to local persistent folder
                print("\nQdrant falling back to local\n")
                local_dir = "./data/qdrant_db"
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=local_dir)
        
        self._ensure_collection()


        self.vector_store = LangchainQdrant(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
    def _ensure_collection(self):
        """Ensure the collection exists in Qdrant with 384 dimensions (all-MiniLM-L6-v2)."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )
    
    
    def add_chunks(self, chunks: List[Chunk]) -> int:
        """add a list fo chunks to Qdrant"""
        if not chunks:
            return 0
        

        text = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self.vector_store.add_texts(texts=text, metadatas=metadatas)
        return len(chunks)

    def similarity_search(self, query: str, k: int = 4) -> List[Chunk]:
        """Perform similarity search and return matching Chunks."""

        docs = self.vector_store.similarity_search(query, k=k)
        return[
            Chunk(content=doc.page_content, metadata=doc.metadata)
            for doc in docs
        ]
    

    def count(self) -> int:
        """Return total vectors in the collection."""
        try:
            res = self.client.count(collection_name=self.collection_name, exact=True)
            return res.count
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete all points and reset the collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=models.Filter())
            )
        except Exception:
            try:
                self.client.delete_collection(self.collection_name)
                self._ensure_collection()
            except Exception:
                pass
        self.vector_store = LangchainQdrant(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )


if __name__ == "__main__":
    store = QdrantStore()
    sample_chunks = [
        Chunk(content="Qdrant is a vector database written in Rust for high performance.", metadata={"source": "qdrant_doc.txt", "chunk_index": 0}),
        Chunk(content="Chroma is a developer-friendly vector store for AI embeddings.", metadata={"source": "chroma_doc.txt", "chunk_index": 1}),
    ]
    store.add_chunks(sample_chunks)
    print(f"Total vectors in Qdrant: {store.count()}")
    res = store.similarity_search("What is Qdrant?", k=1)
    print(f"Search Result: {res[0].content}")
