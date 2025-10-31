import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingManager:
  """Handles document embedding geeneration using SenteceTransformer"""

  def __init__(self, model_name : str = "all-MiniLM-L6-v2"):
    """
      Initiakse the embedding manager

      args:
        model_name : hugging face model name for sentence embeddings
      """
    self.model_name = model_name
    self.model = None
    self._load_model()


  def _load_model(self):
    """Load the senetenceTransformer model"""
    try:
      print(f"Loading embedding model : {self.model_name}")
      self.model = SentenceTransformer(self.model_name)
      print(f"Model Loaded suceessfully. Embedding dimension : {self.model.get_sentence_embedding_dimension()}")
    except Exception as e:
      print(f"Error loading model : {e}")
      raise


  def generate_embeddings(self, texts : List[str]) -> np.ndarray:
    """ Generate embeddings for list of texts

    args: texts: list of text string to embed

    returns : numpy array of embeddings wiyth shape (len(texts) , embedding_dim)
    """

    if not self.model:
      raise ValueError("Model not loaded")

    print(f"Generatring embedddings for {len(texts)} texts...")
    embeddings = self.model.encode(texts, show_progress_bar = True)
    print(f"Embeddings generated successfully. Shape : {embeddings.shape}")
    return embeddings
