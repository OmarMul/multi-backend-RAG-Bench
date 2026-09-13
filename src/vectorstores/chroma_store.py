"""ChromaDB vector store implementation."""

from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.ingestion.loader import Chunk



class ChromaVectorStore:
    """Manages document embeddings and similarity search using ChromaDB."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "rag_bench_chroma",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):

        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)


        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False)
        )


        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Add a list of Chunks to the Chroma vector store."""
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        ids = [
            f"{c.metadata.get('source', 'doc')}_{c.metadata.get('chunk_index', i)}_{i}"
            for i, c in enumerate(chunks)
        ]

        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return len(chunks)


    def similarity_search(self, query: str, k: int = 4) -> List[Chunk]:
        """Perform similarity search and return matching Chunks with metadata."""
        docs = self.vector_store.similarity_search(query, k=k)
        return [
            Chunk(content=doc.page_content, metadata=doc.metadata)
            for doc in docs
        ]


    def count(self) -> int:
        """Get the total number of documents in the collection."""
        collection = self.client.get_collection(self.collection_name)
        return collection.count()
    
    
    def clear(self) -> None:
        """Clear the collection."""
        self.client.delete_collection(self.collection_name)
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )


if __name__ == "__main__":
    store = ChromaVectorStore()
    sample_chunks = [
        Chunk(content="LangChain is an orchestration framework for LLM applications.", metadata={"source": "test.txt", "chunk_index": 0}),
        Chunk(content="Chroma is an open-source embedding database for AI applications.", metadata={"source": "test.txt", "chunk_index": 1}),
    ]
    store.add_chunks(sample_chunks)
    print(f"Total chunks in Chroma: {store.count()}")
    results = store.similarity_search("What is Chroma?", k=1)
    print(f"Search result: {results[0].content}")