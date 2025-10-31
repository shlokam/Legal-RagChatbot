from embedding import EmbeddingManager
from typing import List, Dict, Any
from vectorStore_AWS import VectorStore

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        print(dir(VectorStore))
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, score_threshold, top_k) -> List[Dict[str, Any]]:

        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        return self.vector_store.query(embedding=query_embedding.astype("float32").tolist(), n_results=top_k, score_threshold=score_threshold)
        


