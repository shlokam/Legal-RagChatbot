from typing import Any
import boto3
import json
import re

class VectorStore:
  """Manages document embeddings in vector store"""

  def __init__(self, collection_name: str = "pdf_doucments"):
    """Initalise vector store
    args :
    collection_name : name of the chromadb collecton
    persist_dir : dir to persis vector store
    """

    self.s3vectors = boto3.client("s3vectors",region_name="us-east-1")
    self.bucket_name = "vectorbuckettechxi"
    self.collection_name = collection_name

  def query(self, embedding: list, n_results, score_threshold) -> Any:
      try:
        vector_response = self.s3vectors.query_vectors(
                vectorBucketName=self.bucket_name,
                indexName="s3-vector-index",
                queryVector={"float32": embedding},
                topK=n_results,
                returnDistance=True,
                returnMetadata=True
            )
        print("Query successful. Results:")
        print(json.dumps(vector_response["vectors"], indent=2))
        return vector_response["vectors"]
      except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

  
  